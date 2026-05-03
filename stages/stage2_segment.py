"""
Stage 2: SAM2 segmentation (batch)
Loads SAM2 once, processes all output dirs that have metadata.json,
writes image.png and 0.png to each.
"""
import argparse
import json
import os
import sys
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Folder of original input images")
    parser.add_argument("--output_root", required=True, help="Root output folder (outputs/)")
    parser.add_argument("--sam2_repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already have 0.png")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    sys.path.insert(0, args.sam2_repo)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Find all output dirs that have metadata.json (i.e. passed stage 1)
    extensions = {".jpg", ".jpeg", ".png"}
    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    if not image_files:
        print(f"[Stage 2] No images found in {args.image_dir}")
        sys.exit(0)

    # Filter to only those with metadata.json
    pending = []
    for f in image_files:
        name = os.path.splitext(f)[0]
        out_dir = os.path.join(args.output_root, name)
        meta_path = os.path.join(out_dir, "metadata.json")

        if not os.path.exists(meta_path):
            print(f"[Stage 2] Skipping {f} — no metadata.json (stage 1 incomplete or failed)")
            continue

        if args.skip_existing and os.path.exists(os.path.join(out_dir, "0.png")):
            print(f"[Stage 2] Skipping {f} — already segmented")
            continue

        # Check it actually has objects worth segmenting
        with open(meta_path) as mf:
            data = json.load(mf)
        if not data.get("objects"):
            print(f"[Stage 2] Skipping {f} — no objects detected in stage 1")
            continue

        pending.append(f)

    if not pending:
        print("[Stage 2] Nothing to process.")
        sys.exit(0)

    print(f"[Stage 2] Found {len(pending)} images to segment.")

    # Load SAM2 once
    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint))
    completed, failed = 0, 0

    for image_filename in pending:
        image_name = os.path.splitext(image_filename)[0]
        image_path = os.path.join(args.image_dir, image_filename)
        output_dir = os.path.join(args.output_root, image_name)

        print(f"[Stage 2] ({completed + failed + 1}/{len(pending)}) {image_filename}")

        try:
            with open(os.path.join(output_dir, "metadata.json")) as f:
                data = json.load(f)

            best_obj = max(data["objects"], key=lambda x: x["score"])
            input_box = np.array(best_obj["bbox"])

            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"[Stage 2] WARNING: Could not read {image_path}, skipping.")
                failed += 1
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            masks, _, _ = predictor.predict(
                box=input_box[None, :],
                multimask_output=False
            )

            mask = masks[0]
            alpha = (mask * 255).astype(np.uint8)
            b, g, r = cv2.split(image_bgr)
            rgba = cv2.merge([b, g, r, alpha])

            # Both saved via cv2 to guarantee matching dimensions
            cv2.imwrite(os.path.join(output_dir, "image.png"), image_bgr)
            cv2.imwrite(os.path.join(output_dir, "0.png"), rgba)

            print(f"[Stage 2]   → Segmented (score: {best_obj['score']:.4f})")
            completed += 1

        except Exception as e:
            print(f"[Stage 2] ERROR on {image_filename}: {e}")
            failed += 1
            continue

    print(f"\n[Stage 2] Done. {completed} completed, {failed} failed.")


if __name__ == "__main__":
    main()