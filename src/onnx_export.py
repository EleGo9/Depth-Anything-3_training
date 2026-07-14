# Copyright (c) 2026 MSch8791.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

import torch
import onnx

from depth_anything_3.api import DepthAnything3


class DepthAnything3Wrapper(torch.nn.Module):
    def __init__(self, api_model: DepthAnything3, export_sky: bool = False, export_conf: bool = False) -> None:
        super().__init__()
        self._model = api_model
        self._export_sky = export_sky
        self._export_conf = export_conf
        # _process_mono_sky_estimation contains data-dependent branches (sum() <= 10)
        # that are incompatible with torch.export static tracing. Replace with identity.
        _noop = lambda output: output
        for mod in self._model.model.modules():
            if hasattr(type(mod), "_process_mono_sky_estimation"):
                mod._process_mono_sky_estimation = _noop

    # TODO HERE : you can add the others inputs (intrinsics matrix, extrinsics matrix, etc) and
    # return the others outputs given by the model
    def forward(self, image: torch.Tensor):
        model_in = image

        with torch.no_grad():
            dtype = torch.float32 if model_in.device.type == "cpu" else torch.float16
            with torch.autocast(device_type=model_in.device.type, dtype=dtype):
                # we use the internal model object
                output = self._model.model(
                    model_in,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[],
                    infer_gs=False,
                    use_ray_pose=False
                )

        depth = output["depth"]

        outputs = [depth]

        if self._export_conf:
            assert "depth_conf" in output, (
                "Model output does not contain 'depth_conf'. "
                "da3mono-large (output_dim=1) has no confidence head; use da3-large."
            )
            # depth_conf: (B*N, H, W) — confidence score per pixel, higher = more reliable.
            outputs.append(output["depth_conf"])

        if self._export_sky:
            assert "sky" in output, (
                "Model output does not contain 'sky'. "
                "Make sure the model was trained with use_sky_head=True."
            )
            # sky: (B*N, H_sky, W_sky) — raw relu activation, higher value = more sky.
            # Threshold at 0.3 to get a binary mask: sky_mask = sky >= 0.3
            outputs.append(output["sky"])

        return tuple(outputs) if len(outputs) > 1 else outputs[0]


def getArguments():
    parser = argparse.ArgumentParser(description='Replay tool for performance testing')
    parser.add_argument('--da3model', type=str, help='The Depth Anything 3 model to convert. It can be a Hugging Face project identifier or a path to the downloaded model.')
    parser.add_argument('--output', type=str, help='The path where to write the ONNX model file (.onnx).')
    parser.add_argument('--nviews', type=int, help='Number of views for the model\'s input')
    parser.add_argument('--batchsize', type=int, help='Batch size for the model\'s input')
    parser.add_argument('--conf', action='store_true', help='Export the depth confidence map alongside depth (da3-large only, not da3mono-large).')
    parser.add_argument('--sky',  action='store_true', help='Export the sky segmentation head output alongside depth (da3mono-large only).')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = getArguments()

    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading the model...")
    model = DepthAnything3.from_pretrained(args.da3model)
    model = model.to(device)
    model.eval()
    print(f"Model at {args.da3model} loaded on {device}")

    print("Converting...")
    wrapper = DepthAnything3Wrapper(model, export_sky=args.sky, export_conf=args.conf).to(device)

    # assert(args.batchsize > 0 and args.nviews > 0)

    B, N_views, C, H, W = args.batchsize, args.nviews, 3, 210, 210
    # TODO HERE : add the others dummy inputs necessary (dummy intrinsic/extrinsic/etc tensors)
    dummy_input = torch.zeros(B, N_views, C, H, W).to(device)

    output_names = ["depth"]
    if args.conf:
        output_names.append("conf")
    if args.sky:
        output_names.append("sky")


    dynamic_axes = {
        "image": {0: "batch"},
    }
    for out_name in output_names:
        dynamic_axes[out_name] = {0: "batch"}

    # TODO HERE : add the others input and output names you need
    with torch.no_grad():
        onnx_program = torch.onnx.export(
            wrapper,
            dummy_input,
            args.output,
            export_params=True,
            do_constant_folding=True,
            input_names=["image"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            training=torch.onnx.TrainingMode.EVAL
        )

    print(f"Convertion done successfully, model saved at {args.output}.")

    print("Checking the model...")
    # check the converted model
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    print("Job done.")
