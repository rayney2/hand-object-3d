# Hand-Object 3D Reconstruction Pipeline

End-to-end pipeline that takes first-person images of hands holding objects and produces 3D Gaussian splat reconstructions of the held objects.

## How It Works

The pipeline runs in three stages, each backed by a different AI model:

1. **Stage 1 — Hand-Object Detection.** A Faster R-CNN model identifies hand and object bounding boxes, plus contact state (which object the hand is holding).
2. **Stage 2 — Segmentation.** SAM2 takes the highest-confidence object bounding box and produces a precise pixel-level mask of the held object.
3. **Stage 3 — 3D Reconstruction.** sam3d-objects takes the original image plus the mask and generates a 3D Gaussian splat reconstruction (.ply) plus a turntable render (.gif).

Each stage runs in its own conda environment because the three models have incompatible dependencies. A lightweight orchestrator (`run_pipeline.py`) coordinates them via subprocess calls.


## Output Structure

For an input image `photo.jpg`, the pipeline produces:

```
outputs/photo/
├── metadata.json              # Stage 1: hand and object bboxes, scores, contact state
├── detection.png              # Stage 1: visualization with boxes drawn
├── image.png                  # Stage 2: original image
├── 0.png                      # Stage 2: RGBA segmented object cutout
├── photo.ply                  # Stage 3: combined 3D Gaussian splat
├── photo.gif                  # Stage 3: 300-frame turntable render preview
├── photo_data.npz             # Stage 3: all raw Gaussian arrays + transforms as numpy
├── photo_summary.json         # Stage 3: human-readable stats (size, position, rotation)
├── photo_obj0_mesh.obj        # Stage 3: extracted triangle mesh (per object)
└── photo_obj0.glb             # Stage 3: textured 3D model (per object, openable in Blender)
```

---

## Setup

This project orchestrates **three external model repositories** that you must clone and install separately. The pipeline itself does not bundle them.

### 1. Clone the external model repos

Pick a directory to hold them all (e.g. `~/models/`):

```bash
mkdir -p ~/models && cd ~/models

git clone https://github.com/ddshan/hand_object_detector.git
git clone https://github.com/facebookresearch/sam2.git
git clone https://github.com/facebookresearch/sam-3d-objects.git
```

### 2. Set up each model's conda environment

Each repo has its own README with installation instructions. You must follow them in full and confirm each model runs standalone before integrating.

**Hand-Object Detector** — follow [the hand_object_detector README](https://github.com/ddshan/hand_object_detector). Download the pretrained `faster_rcnn_1_8_89999.pth` weights into `models/res101_handobj_100K/pascal_voc/`.

**SAM2** — follow [the sam2 README](https://github.com/facebookresearch/sam2). Download `sam2.1_hiera_large.pt` checkpoint.

**sam3d-objects** — follow [the sam-3d-objects README](https://github.com/facebookresearch/sam-3d-objects). Download all required checkpoints into `checkpoints/hf/`.

After this step you should have **three working conda environments**, each able to run their respective demo scripts standalone. Note the env name of each — you'll need them for `config.yaml`.

### 3. Clone this pipeline repo

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/hand-object-3d.git
cd hand-object-3d
```

### 4. Create the pipeline environment

This is a separate, minimal environment just for the orchestrator:

```bash
conda create -n pipeline python=3.11 -y
conda activate pipeline
pip install -r requirements.txt
```

### 5. Configure paths

Copy the template and fill in your machine-specific paths:

```bash
cp config.template.yaml config.yaml
```

Open `config.yaml` and set:
- The conda env names (must match what you created in step 2)
- Absolute paths to each cloned repo
- Paths to model checkpoints
- Detector model parameters (epoch and checkpoint number from the weights filename)

`config.yaml` is gitignored — it contains your local paths and stays on your machine.

---

## Usage

Activate the pipeline env:

```bash
conda activate pipeline
```

### Run the full pipeline on a folder of images

```bash
python run_pipeline.py --input_dir /path/to/images
```

This processes every `.jpg`, `.jpeg`, and `.png` in the folder through all three stages, batched per stage.

### Run only specific stages

```bash
# Detection only
python run_pipeline.py --input_dir /path/to/images --stages 1

# Detection + segmentation, skip 3D reconstruction
python run_pipeline.py --input_dir /path/to/images --stages 1 2

# Just regenerate the .ply + .gif from existing masks
python run_pipeline.py --input_dir /path/to/images --stages 3
```

### Resume after a crash

`--skip_existing` causes each stage to skip images whose output for that stage already exists. Safe to re-run as many times as needed.

```bash
python run_pipeline.py --input_dir /path/to/images --skip_existing
```

---

## Configuration Reference

`config.yaml` controls all paths and parameters. Fields:

```yaml
environments:
  detector: <conda env name for hand_object_detector>
  sam2:     <conda env name for sam2>
  sam3d:    <conda env name for sam-3d-objects>

paths:
  detector_repo: <absolute path to cloned hand_object_detector>
  sam2_repo:     <absolute path to cloned sam2>
  sam3d_repo:    <absolute path to cloned sam-3d-objects>
  outputs:       ./outputs

detector:
  checkepoch:    8         # from weights filename: faster_rcnn_<sess>_<EPOCH>_<ckpt>.pth
  checkpoint:    89999     # from weights filename
  checksession:  1
  net:           res101    # res50, res101, res152, or vgg16
  thresh_hand:   0.5
  thresh_obj:    0.5
  cuda_device:   0

sam2:
  checkpoint:    <absolute path to sam2.1_hiera_large.pt>
  model_cfg:     configs/sam2.1/sam2.1_hiera_l.yaml
  cuda_device:   0

sam3d:
  config:            checkpoints/hf/pipeline.yaml   # relative to sam3d_repo
  seed:              42
  render_resolution: 512
  fov:               60
  cuda_device:       0
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'yaml'`** — You're not in the `pipeline` env. Run `conda activate pipeline`.

**`conda: command not found` from inside subprocess** — Your conda installation isn't in the non-interactive shell's PATH. Add this to your `~/.bashrc`:
```bash
source ~/miniconda3/etc/profile.d/conda.sh   # adjust path to your conda
```

**Stage 3 `IndexError: list index out of range` in `make_scene`** — Stage 2 produced an empty/bad mask, or the mask file naming doesn't match what stage 3 expects. Confirm `outputs/{name}/0.png` exists and has matching dimensions to `image.png`.

**Stage 3 dimension mismatch errors** — `image.png` and `0.png` were saved by different libraries with different decoders. Stage 2 should write both via cv2; check `stages/stage2_segment.py` if this regresses.

**Out of GPU memory in Stage 3** — Stage 3 already monkey-patches the decoder to offload generators to CPU during decode. If you still OOM, you may need to lower `render_resolution` in config or use a GPU with more VRAM. Tested on a 24 GiB RTX 4090.

**Different GPUs per stage** — Set different `cuda_device` values per stage in `config.yaml` if you want to spread work across multiple GPUs.

---
## Estimating Real-World Size and Distance

The 3D reconstruction in Stage 3 is **scale-ambiguous** by design — a single image can't tell whether you're looking at a real coffee cup or a giant prop. All coordinates in `_summary.json` are in normalized units that describe the object's *shape and relative depth*, not its real size.

To recover real-world measurements you need at least one **reference of known size** in the image. The most natural reference for hand-held photos is the **hand itself**, since human hands have well-documented average dimensions.

Real-world size estimation;

**Step 1 — pixels-to-cm calibration from the hand:**
```
pixels_per_cm = hand_pixel_width / hand_real_width_cm
```

The hand's pixel width comes from Stage 1's `metadata.json` (the `hands[0].bbox` field). The hand's real width is assumed from average adult anatomy.

**Step 2 — object size estimation:**
```
object_real_width_cm = object_pixel_width / pixels_per_cm
```

The object's pixel width comes from Stage 1's `metadata.json` (the `objects[0].bbox` field).

**Step 3 — distance estimation (requires camera focal length):**

Using the pinhole camera model:
```
distance_cm = (focal_length_pixels × object_real_width_cm) / object_pixel_width
```

The focal length in pixels can come from EXIF metadata or be approximated from the focal length in mm and the sensor size:
```
focal_length_pixels ≈ (focal_length_mm × image_width_pixels) / sensor_width_mm
```

## Project Structure

```
hand-object-3d/
├── README.md              # this file
├── requirements.txt       # pipeline env deps (just pyyaml)
├── config.template.yaml   # copy to config.yaml and edit
├── config.yaml            # gitignored, your local paths
├── .gitignore
├── run_pipeline.py        # orchestrator entry point
├── stages/
│   ├── stage1_detect.py        # Faster R-CNN wrapper
│   ├── stage2_segment.py       # SAM2 wrapper
│   └── stage3_reconstruct.py   # sam3d-objects wrapper
└── outputs/               # gitignored, generated at runtime
```

---

## Credits

This pipeline orchestrates three external models:

- [Hand-Object Detector](https://github.com/ddshan/hand_object_detector) (Shan et al.)
- [SAM 2](https://github.com/facebookresearch/sam2) (Meta AI)
- [sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) (Meta AI)

Refer to each project for its own license and citation requirements.

## License

[Choose your license — MIT, Apache 2.0, etc.]