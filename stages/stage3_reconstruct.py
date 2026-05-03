"""
Stage 3: SAM3D reconstruction (batch)
Loads the sam3d model once, processes all output dirs that have image.png + 0.png,
writes {image_name}.ply and {image_name}.gif to each.
"""
import argparse
import gc
import os
import sys
import types
import imageio
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True, help="Root output folder (outputs/)")
    parser.add_argument("--sam3d_repo", required=True)
    parser.add_argument("--config", required=True, help="Path to sam3d pipeline.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fov", type=int, default=60)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already have a .ply file")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    sys.path.insert(0, args.sam3d_repo)
    sys.path.insert(0, os.path.join(args.sam3d_repo, "notebook"))

    from inference import (
        Inference, load_image, load_masks,
        make_scene, ready_gaussian_for_video_rendering, render_video
    )

    # Find all output dirs that have image.png + 0.png (i.e. passed stage 2)
    all_dirs = sorted([
        d for d in os.listdir(args.output_root)
        if os.path.isdir(os.path.join(args.output_root, d))
    ])

    pending = []
    for name in all_dirs:
        out_dir = os.path.join(args.output_root, name)
        has_image = os.path.exists(os.path.join(out_dir, "image.png"))
        has_mask = os.path.exists(os.path.join(out_dir, "0.png"))
        already_done = os.path.exists(os.path.join(out_dir, f"{name}.ply"))

        if not (has_image and has_mask):
            print(f"[Stage 3] Skipping {name} — stage 2 incomplete")
            continue
        if args.skip_existing and already_done:
            print(f"[Stage 3] Skipping {name} — already reconstructed")
            continue

        pending.append((name, out_dir))

    if not pending:
        print("[Stage 3] Nothing to process.")
        sys.exit(0)

    print(f"[Stage 3] Found {len(pending)} items to reconstruct.")

    # Load model once
    inference = Inference(args.config, compile=False)

    # Low-VRAM monkey-patch
    original_decode_slat = inference._pipeline.decode_slat.__func__

    def decode_slat_low_vram(self, slat, formats):
        for name in ["ss_generator", "slat_generator"]:
            if name in self.models:
                self.models[name].cpu()
        torch.cuda.empty_cache()
        gc.collect()
        try:
            result = original_decode_slat(self, slat, formats)
        finally:
            for name in ["ss_generator", "slat_generator"]:
                if name in self.models:
                    self.models[name].cuda()
            torch.cuda.empty_cache()
        return result

    inference._pipeline.decode_slat = types.MethodType(
        decode_slat_low_vram, inference._pipeline
    )

    completed, failed = 0, 0

    for image_name, output_dir in pending:
        print(f"\n[Stage 3] ({completed + failed + 1}/{len(pending)}) {image_name}")

        try:
            image = load_image(os.path.join(output_dir, "image.png"))
            masks = load_masks(output_dir, extension=".png")

            if len(masks) == 0:
                print(f"[Stage 3] ERROR: No masks found in {output_dir}, skipping.")
                failed += 1
                continue

            print(f"[Stage 3]   Loaded {len(masks)} mask(s), running inference...")

            outputs = []
            for i, mask in enumerate(masks):
                print(f"[Stage 3]   Mask {i + 1}/{len(masks)}...")
                with torch.no_grad():
                    out = inference(image, mask, seed=args.seed)
                outputs.append(out)
                torch.cuda.empty_cache()
                gc.collect()

            scene_gs = make_scene(*outputs)
            del outputs
            torch.cuda.empty_cache()
            gc.collect()

            scene_gs = ready_gaussian_for_video_rendering(scene_gs)

            ply_path = os.path.join(output_dir, f"{image_name}.ply")
            gif_path = os.path.join(output_dir, f"{image_name}.gif")

            scene_gs.save_ply(ply_path)
            print(f"[Stage 3]   → PLY saved: {ply_path}")

            video_data = render_video(
                scene_gs, r=1, fov=args.fov, resolution=args.resolution
            )["color"]
            del scene_gs
            torch.cuda.empty_cache()

            imageio.mimsave(gif_path, video_data, format="GIF",
                            duration=1000 / 30, loop=0)
            print(f"[Stage 3]   → GIF saved: {gif_path}")
            completed += 1

        except Exception as e:
            print(f"[Stage 3] ERROR on {image_name}: {e}")
            failed += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    print(f"\n[Stage 3] Done. {completed} completed, {failed} failed.")


if __name__ == "__main__":
    main()