"""
Hand-Object 3D Reconstruction Pipeline
Runs all 3 stages in batch mode — full batch per stage.

Usage:
    python run_pipeline.py --input_dir /path/to/images
    python run_pipeline.py --input_dir /path/to/images --stages 1 2   # run only stages 1 and 2
    python run_pipeline.py --input_dir /path/to/images --skip_existing # skip already done
"""
import argparse
import os
import subprocess
import sys
import yaml


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print("ERROR: config.yaml not found. "
              "Copy config.template.yaml to config.yaml and fill in your paths.")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_stage(label, cmd, env_name):
    full_cmd = f"conda run --no-capture-output -n {env_name} python {' '.join(str(c) for c in cmd)}"
    print(f"\n{'='*60}")
    print(f"  {label}  (env: {env_name})")
    print(f"  {full_cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(full_cmd, shell=True)
    if result.returncode != 0:
        print(f"\nERROR: {label} failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Hand-Object 3D Reconstruction Pipeline")
    parser.add_argument("--input_dir", required=True,
                        help="Folder of input images to process")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3],
                        help="Which stages to run, e.g. --stages 1 2")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip outputs that have already been completed")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = config["paths"]["outputs"]
    os.makedirs(output_root, exist_ok=True)

    skip_flag = ["--skip_existing"] if args.skip_existing else []

    if 1 in args.stages:
        run_stage(
            "Stage 1: Hand-Object Detection",
            [
                "stages/stage1_detect.py",
                "--image_dir", args.input_dir,
                "--output_root", output_root,
                "--detector_repo", config["paths"]["detector_repo"],
                "--checkepoch", config["detector"]["checkepoch"],
                "--checkpoint", config["detector"]["checkpoint"],
                "--checksession", config["detector"]["checksession"],
                "--net", config["detector"]["net"],
                "--thresh_hand", config["detector"]["thresh_hand"],
                "--thresh_obj", config["detector"]["thresh_obj"],
                "--cuda_device", config["detector"].get("cuda_device", 0),
            ] + skip_flag,
            config["environments"]["detector"],
        )

    if 2 in args.stages:
        run_stage(
            "Stage 2: SAM2 Segmentation",
            [
                "stages/stage2_segment.py",
                "--image_dir", args.input_dir,
                "--output_root", output_root,
                "--sam2_repo", config["paths"]["sam2_repo"],
                "--checkpoint", config["sam2"]["checkpoint"],
                "--model_cfg", config["sam2"]["model_cfg"],
                "--cuda_device", config["sam2"].get("cuda_device", 0),
            ] + skip_flag,
            config["environments"]["sam2"],
        )

    if 3 in args.stages:
        run_stage(
            "Stage 3: SAM3D Reconstruction",
            [
                "stages/stage3_reconstruct.py",
                "--output_root", output_root,
                "--sam3d_repo", config["paths"]["sam3d_repo"],
                "--config", os.path.join(
                    config["paths"]["sam3d_repo"], config["sam3d"]["config"]
                ),
                "--seed", config["sam3d"]["seed"],
                "--resolution", config["sam3d"]["render_resolution"],
                "--fov", config["sam3d"]["fov"],
                "--cuda_device", config["sam3d"].get("cuda_device", 0),
            ] + skip_flag,
            config["environments"]["sam3d"],
        )

    print("\n✓ Pipeline complete.")
    print(f"  Outputs in: {output_root}")


if __name__ == "__main__":
    main()