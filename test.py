"""
DA3 multi-view inference con pose GT da proiezioni planari di un fisheye.

Formato .pose in input: C2W (camera-to-world) in convenzione OpenCV.
  - translation = centro ottico in world frame
  - colonna 2 di R = direzione forward della camera in world frame
  - output dal C++: pose = mTfs * tfC2W

DA3 vuole W2C (world-to-camera):
  W2C = inv(C2W)           ← unica conversione necessaria

Caso speciale: 6 proiezioni planari dello stesso fisheye hanno tutte lo
stesso centro ottico → baseline = 0 → Umeyama scale = 0 se le GT
extrinsics vengono passate a inference(). Soluzione: si fa girare
inference() senza extrinsics GT e si iniettano W2C dopo, prima del GLB.
"""

import numpy as np
import torch
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import export_to_glb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
USE_MONO   = False  # True = da3mono-large (ha sky mask), False = da3-large
OUTPUT_DIR = "output"

# DATA_DIR = "/home/elena/repos/Depth-Anything-3/samples_pp"
DATA_DIR = "/home/elena/repos/tkFerrari_pp/tkFerrariAdas/build/sfm"
# IMAGES = [f"{DATA_DIR}/cam_surround_r_0_{i}.jpg" for i in range(6)]
IMAGES = [f"{DATA_DIR}/cam_surround_r_0_1.jpg"] #, f"{DATA_DIR}/cam_surround_r_0_2.jpg", f"{DATA_DIR}/cam_surround_r_0_5.jpg", f"{DATA_DIR}/cam_surround_f_0_0.jpg", f"{DATA_DIR}/cam_surround_f_0_2.jpg", f"{DATA_DIR}/cam_surround_f_0_5.jpg", ]
DEPTH = ["/home/elena/repos/tkFerrari_pp/tkFerrariAdas/build/depth_preds/_0_1_d8.jpg"]
# IMAGES = ["/home/elena/repos/Depth-Anything-3/sample_data/cam_surround_f.png"]
# ---------------------------------------------------------------------------
# Carica dati
# ---------------------------------------------------------------------------
c2w_list   = []  # (N, 4, 4) C2W — per reference / debug
w2c_list   = []  # (N, 4, 4) W2C — quello che serve a DA3
intr_list  = []  # (N, 3, 3) intrinsics

# for img_path in IMAGES:
#     C2W  = np.loadtxt(img_path + ".pose")   # (4,4) C2W in OpenCV convention
#     intr = np.loadtxt(img_path + ".intr")   # (3,3)
   
#     # C2W += 0.000001
   
#     W2C  = np.linalg.inv(C2W)               # unica conversione: inv(C2W)



#     c2w_list.append(C2W)
#     w2c_list.append(W2C)
#     intr_list.append(intr)

# Sanity check rapido
centers = np.array([c[:3, 3] for c in c2w_list])
print(f"Camera centers (dovrebbero essere tutti uguali per un fisheye):")
for i, c in enumerate(centers):
    print(f"  cam{i}: {c.round(4)}")
# baseline_max = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max()
# print(f"Baseline max: {baseline_max:.4f} m")
# if baseline_max < 0.01:
#     print("→ zero baseline: Umeyama bypassed (extrinsics iniettate dopo inference)\n")

# ---------------------------------------------------------------------------
# Modello
# ---------------------------------------------------------------------------
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "depth-anything/da3mono-large" if USE_MONO else "depth-anything/da3-large"
print(f"Loading {model_id} on {device}...")
model = DepthAnything3.from_pretrained(model_id).to(device)
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
if USE_MONO:
    prediction = model.inference(
        IMAGES,
        use_ray_pose=True,
        export_dir=None,
        conf_thresh_percentile=0.001,
        num_max_points=10_000_000,
    )
else:
    # Non passiamo extrinsics GT: baseline = 0 → Umeyama scale = 0 → depth = inf.
    # Le intrinsics GT invece sono utili per condizionare le ray directions.
    prediction = model.inference(
        IMAGES,
        # extrinsics=w2c_list,
        # intrinsics=intr_list,
        align_to_input_ext_scale=False,
        use_ray_pose=True,
        export_dir=None,
        conf_thresh_percentile=0.001,
        num_max_points=10_000_000,
        ref_view_strategy="saddle_sim_range"
    )

# Inietta GT extrinsics (W2C) e intrinsics nel prediction object.
# export_to_glb usa prediction.extrinsics per unproiettare depth → world.
N = prediction.depth.shape[0]
if len(w2c_list)>0:
    prediction.extrinsics = np.array([w2c_list[i][:3]  for i in range(N)])  # (N,3,4)
    prediction.intrinsics = np.array([intr_list[i]      for i in range(N)])  # (N,3,3)
    if prediction.conf is None:
        prediction.conf = np.ones_like(prediction.depth)

# ---------------------------------------------------------------------------
# Sky colorization (solo da3mono-large ha lo sky head)
# ---------------------------------------------------------------------------
SKY_COLOR = np.array([100, 149, 237], dtype=np.uint8)
if prediction.sky is not None:
    for i in range(prediction.sky.shape[0]):
        sky_mask = prediction.sky[i]
        img_h, img_w = prediction.processed_images[i].shape[:2]
        if sky_mask.shape != (img_h, img_w):
            sky_pil = Image.fromarray(sky_mask.astype(np.uint8) * 255)
            sky_pil = sky_pil.resize((img_w, img_h), Image.NEAREST)
            sky_mask = np.array(sky_pil) > 127
        prediction.processed_images[i][sky_mask] = SKY_COLOR

# ---------------------------------------------------------------------------
# Export GLB
# ---------------------------------------------------------------------------

if DEPTH:
    import cv2
    # depth = cv2.imread(DEPTH[0])
    # print('depth', depth.shape)
    # print(depth[:,:,0])
    # print('prediction depth', prediction.depth.shape)
    # depth = depth[:,:,0]
    # prediction.depth = depth [None, :, :]
    # print(prediction.depth.shape==depth.shape)
    depth_tk = cv2.imread(DEPTH[0])        # (504, 504), uint8

    # --- predizione ---
    depth_py = prediction.depth                             # (1, 504, 504)
    if hasattr(depth_py, 'cpu'):                            # se è un tensor torch
        pred = depth_py.cpu().numpy()
    depth_py = depth_py.squeeze(0)                              # (504, 504), float

    # --- normalizza entrambe a 0-255 per visualizzarle ---
    def to_vis(d):
        d = d.astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d = (d * 255).astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)

    vis_tk   = depth_tk
    vis_py = to_vis(depth_py)

    side_by_side = np.hstack([vis_tk, vis_py])        # (504, 1008, 3)
    cv2.imshow('tk | python', side_by_side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



export_to_glb(
    prediction=prediction,
    export_dir=OUTPUT_DIR,
    conf_thresh_percentile=0.001,
    num_max_points=10_000_000,
    show_cameras=True,
    export_depth_vis=True,
)
print(f"\nGLB esportato in {OUTPUT_DIR}/scene.glb")
print(f"depth:      {prediction.depth.shape}  range [{prediction.depth.min():.3f}, {prediction.depth.max():.3f}]")
print(f"extrinsics: {prediction.extrinsics.shape}")
print(f"intrinsics: {prediction.intrinsics.shape}")
