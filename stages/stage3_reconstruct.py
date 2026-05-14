"""
Stage 3: SAM3D reconstruction (batch)
Loads the sam3d model once, processes all output dirs that have image.png + 0.png,
writes {image_name}.ply, .gif, _data.npz, _summary.json, and per-object .obj/.glb files.
"""
import argparse
import gc
import json
import os
import sys
import traceback
import types
import imageio
import numpy as np
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


def to_numpy(x):
    """Best-effort conversion of any output value to a numpy array."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (int, float, bool)):
        return np.array(x)
    if isinstance(x, (list, tuple)):
        try:
            return np.array(x)
        except Exception:
            return None
    return None


def extract_object_data(out, idx):
    """Extract all numpy-friendly arrays from a single inference output dict."""
    arrays = {}

    # Per-Gaussian arrays from the GaussianModel object
    try:
        gs = out["gaussian"][0]
        arrays[f"obj{idx}_xyz_local"]     = to_numpy(gs.get_xyz)
        arrays[f"obj{idx}_rgb_dc"]        = to_numpy(gs.get_features_dc)
        arrays[f"obj{idx}_opacity"]       = to_numpy(gs.get_opacity)
        arrays[f"obj{idx}_scaling"]       = to_numpy(gs.get_scaling)
        arrays[f"obj{idx}_rotation_quat"] = to_numpy(gs.get_rotation)
    except Exception as e:
        print(f"[Stage 3]   ! Could not extract Gaussian arrays: {e}")

    # Object-level transform fields
    transform_keys = [
        "rotation", "translation", "scale", "translation_scale",
        "6drotation_normalized", "shape", "downsample_factor",
        "coords", "coords_original", "pointmap", "pointmap_colors",
    ]
    for key in transform_keys:
        if key in out:
            arr = to_numpy(out[key])
            if arr is not None:
                # Sanitize key name for npz (no special chars at start)
                safe_key = key.replace("6d", "sixd")
                arrays[f"obj{idx}_{safe_key}"] = arr

    return arrays


def save_mesh(out, idx, image_name, output_dir):
    """Save mesh as .obj if possible."""
    if "mesh" not in out or out["mesh"] is None:
        return False
    try:
        mesh = out["mesh"]
        mesh_path = os.path.join(output_dir, f"{image_name}_obj{idx}_mesh.obj")
        if hasattr(mesh, "export"):
            mesh.export(mesh_path)
        elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            verts = to_numpy(mesh.vertices)
            faces = to_numpy(mesh.faces)
            with open(mesh_path, "w") as f:
                for v in verts:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        else:
            return False
        print(f"[Stage 3]   → Mesh saved: {mesh_path}")
        return True
    except Exception as e:
        print(f"[Stage 3]   ! Mesh save failed: {e}")
        return False


def save_glb(out, idx, image_name, output_dir):
    """Save GLB binary if available."""
    if "glb" not in out or out["glb"] is None:
        return False
    try:
        glb_path = os.path.join(output_dir, f"{image_name}_obj{idx}.glb")
        glb_obj = out["glb"]
        if isinstance(glb_obj, (bytes, bytearray)):
            with open(glb_path, "wb") as f:
                f.write(glb_obj)
        elif hasattr(glb_obj, "save"):
            glb_obj.save(glb_path)
        elif hasattr(glb_obj, "export"):
            glb_obj.export(glb_path)
        else:
            return False
        print(f"[Stage 3]   → GLB saved: {glb_path}")
        return True
    except Exception as e:
        print(f"[Stage 3]   ! GLB save failed: {e}")
        return False


def quat_to_euler_xyz(quat):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in degrees."""
    if quat is None:
        return None
    q = np.asarray(quat).flatten()
    if q.size != 4:
        return None
    w, x, y, z = q[0], q[1], q[2], q[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))]


def build_summary(image_name, outputs, arrays):
    """Build a human-readable summary JSON using the original inference dict field names."""
    summary = {"image_name": image_name, "n_objects": len(outputs), "objects": []}

    for i in range(len(outputs)):
        xyz_key = f"obj{i}_xyz_local"
        translation_key = f"obj{i}_translation"
        scale_key = f"obj{i}_scale"
        rotation_key = f"obj{i}_rotation"

        obj_summary = {"object_index": i}

        # Gaussian count
        if xyz_key in arrays and arrays[xyz_key] is not None:
            xyz = arrays[xyz_key]
            obj_summary["n_gaussians"] = int(xyz.shape[0])

        # translation (X, Y, Z) — original field
        if translation_key in arrays and arrays[translation_key] is not None:
            t = arrays[translation_key].squeeze().tolist()
            if isinstance(t, list) and len(t) == 3:
                obj_summary["translation"] = t

        # scale (X, Y, Z) — original field (all axes usually equal)
        if scale_key in arrays and arrays[scale_key] is not None:
            s = arrays[scale_key].squeeze().tolist()
            if isinstance(s, list):
                obj_summary["scale"] = s

        # rotation — original field as quaternion, plus Euler for readability
        if rotation_key in arrays and arrays[rotation_key] is not None:
            quat = arrays[rotation_key].squeeze().tolist()
            obj_summary["rotation"] = quat
            obj_summary["rotation_euler_degrees"] = quat_to_euler_xyz(quat)

        # Derived: how far the camera is from the object, in units of the object's own size
        try:
            xyz = arrays[xyz_key]
            longest = float((xyz.max(axis=0) - xyz.min(axis=0)).max())
            depth = obj_summary["translation"][2]
            scale_val = obj_summary["scale"][0]
            obj_summary["distance_in_object_lengths"] = depth / (longest * scale_val)
        except (KeyError, ZeroDivisionError, TypeError):
            pass

        summary["objects"].append(obj_summary)

    return summary


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

    inference = Inference(args.config, compile=False)

    # Low-VRAM monkey-patch: offload generators during decode
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

            # ─── Extract data from each per-mask output BEFORE building scene ──
            all_arrays = {}
            for i, out in enumerate(outputs):
                obj_arrays = extract_object_data(out, i)
                all_arrays.update(obj_arrays)
                save_mesh(out, i, image_name, output_dir)
                save_glb(out, i, image_name, output_dir)

            # Save consolidated NPZ with all objects' raw arrays
            npz_path = os.path.join(output_dir, f"{image_name}_data.npz")
            valid_arrays = {k: v for k, v in all_arrays.items() if v is not None}
            np.savez_compressed(npz_path, **valid_arrays)
            print(f"[Stage 3]   → Data saved: {npz_path} ({len(valid_arrays)} arrays)")

            # Save human-readable summary
            summary = build_summary(image_name, outputs, all_arrays)
            summary_path = os.path.join(output_dir, f"{image_name}_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[Stage 3]   → Summary saved: {summary_path}")

            # ─── Build scene, save PLY, render GIF ──
            scene_gs = make_scene(*outputs)
            del outputs
            torch.cuda.empty_cache()
            gc.collect()

            scene_gs = ready_gaussian_for_video_rendering(scene_gs)

            ply_path = os.path.join(output_dir, f"{image_name}.ply")
            scene_gs.save_ply(ply_path)
            print(f"[Stage 3]   → PLY saved: {ply_path}")

            video_data = render_video(
                scene_gs, r=1, fov=args.fov, resolution=args.resolution
            )["color"]
            del scene_gs
            torch.cuda.empty_cache()

            gif_path = os.path.join(output_dir, f"{image_name}.gif")
            imageio.mimsave(gif_path, video_data, format="GIF",
                            duration=1000 / 30, loop=0)
            print(f"[Stage 3]   → GIF saved: {gif_path}")
            completed += 1

        except Exception as e:
            print(f"[Stage 3] ERROR on {image_name}: {e}")
            traceback.print_exc()
            failed += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    print(f"\n[Stage 3] Done. {completed} completed, {failed} failed.")


if __name__ == "__main__":
    main()