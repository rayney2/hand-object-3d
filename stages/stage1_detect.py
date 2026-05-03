"""
Stage 1: Hand-object detection (batch)
Loads the Faster R-CNN model once, processes all images in a folder,
writes metadata.json and detection visualization to outputs/{image_name}/
"""
import argparse
import json
import os
import sys
import traceback
import numpy as np
import cv2
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Folder of input images")
    parser.add_argument("--output_root", required=True, help="Root output folder (outputs/)")
    parser.add_argument("--detector_repo", required=True)
    parser.add_argument("--load_dir", default="models")
    parser.add_argument("--net", default="res101")
    parser.add_argument("--checksession", type=int, default=1)
    parser.add_argument("--checkepoch", type=int, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--thresh_hand", type=float, default=0.5)
    parser.add_argument("--thresh_obj", type=float, default=0.5)
    parser.add_argument("--dataset", default="pascal_voc")
    parser.add_argument("--cfg_file", default="cfgs/res101.yml")
    parser.add_argument("--class_agnostic", action="store_true")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already have metadata.json")
    return parser.parse_args()


def get_image_blob(im, cfg, im_list_to_blob):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im_resized = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im_resized)
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def load_model(args, cfg, pascal_classes):
    model_dir = os.path.join(
        args.detector_repo,
        args.load_dir,
        f"{args.net}_handobj_100K",
        args.dataset,
    )
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    load_name = os.path.join(
        model_dir,
        f"faster_rcnn_{args.checksession}_{args.checkepoch}_{args.checkpoint}.pth"
    )
    if not os.path.exists(load_name):
        raise FileNotFoundError(f"Checkpoint not found: {load_name}")

    if args.net == "vgg16":
        from model.faster_rcnn.vgg16 import vgg16
        fasterRCNN = vgg16(pascal_classes, pretrained=False,
                           class_agnostic=args.class_agnostic)
    elif args.net in ("res101", "res50", "res152"):
        from model.faster_rcnn.resnet import resnet
        depth = int(args.net.replace("res", ""))
        fasterRCNN = resnet(pascal_classes, depth, pretrained=False,
                            class_agnostic=args.class_agnostic)
    else:
        raise ValueError(f"Unknown network: {args.net}")

    fasterRCNN.create_architecture()
    print(f"[Stage 1] Loading checkpoint: {load_name}")
    checkpoint = torch.load(load_name, map_location="cpu")
    fasterRCNN.load_state_dict(checkpoint["model"])
    if "pooling_mode" in checkpoint:
        cfg.POOLING_MODE = checkpoint["pooling_mode"]
    print("[Stage 1] Model loaded.")
    return fasterRCNN


def run_detection(im, fasterRCNN, args, cfg, pascal_classes, im_list_to_blob):
    from model.roi_layers import nms
    from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

    blobs, im_scales = get_image_blob(im, cfg, im_list_to_blob)
    assert len(im_scales) == 1

    im_data_pt = torch.from_numpy(blobs).permute(0, 3, 1, 2)
    im_info_np = np.array(
        [[blobs.shape[1], blobs.shape[2], im_scales[0]]],
        dtype=np.float32
    )
    im_info_pt = torch.from_numpy(im_info_np)

    im_data = torch.FloatTensor(1).cuda()
    im_info = torch.FloatTensor(1).cuda()
    num_boxes = torch.LongTensor(1).cuda()
    gt_boxes = torch.FloatTensor(1).cuda()
    box_info = torch.FloatTensor(1).cuda()

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        box_info.resize_(1, 1, 5).zero_()

        rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list = fasterRCNN(
            im_data, im_info, gt_boxes, num_boxes, box_info
        )

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    contact_vector = loss_list[0][0]
    offset_vector = loss_list[1][0].detach()
    lr_vector = loss_list[2][0].detach()

    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
    lr = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

    if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            if args.class_agnostic:
                box_deltas = (
                    box_deltas.reshape(-1, 4)
                    * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                )
                box_deltas = box_deltas.reshape(1, -1, 4)
            else:
                box_deltas = (
                    box_deltas.reshape(-1, 4)
                    * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                )
                box_deltas = box_deltas.reshape(1, -1, 4 * len(pascal_classes))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    # Squeeze AFTER box transformation (matches original demo.py order)
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    obj_dets, hand_dets = None, None
    for j in range(1, len(pascal_classes)):
        thresh = args.thresh_hand if pascal_classes[j] == "hand" else args.thresh_obj
        inds = torch.nonzero(scores[:, j] > thresh).reshape(-1)
        if inds.numel() == 0:
            continue

        cls_scores = scores[:, j][inds]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes = (pred_boxes[inds, :] if args.class_agnostic
                     else pred_boxes[inds][:, j * 4:(j + 1) * 4])

        cls_dets = torch.cat(
            (cls_boxes, cls_scores.unsqueeze(1),
             contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]),
            dim=1
        )
        cls_dets = cls_dets[order]
        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.reshape(-1).long()].cpu().numpy()

        if pascal_classes[j] == "targetobject":
            obj_dets = cls_dets
        elif pascal_classes[j] == "hand":
            hand_dets = cls_dets

    return obj_dets, hand_dets


def save_outputs(im, obj_dets, hand_dets, args, output_dir,
                 image_filename, vis_detections_filtered_objects_PIL):
    """Save metadata.json and detection visualization."""
    image_data = {
        "image_name": image_filename,
        "hands": [],
        "objects": [],
    }

    if hand_dets is not None:
        for hand in hand_dets:
            image_data["hands"].append({
                "bbox": hand[0:4].tolist(),
                "score": float(hand[4]),
                "contact_state": int(hand[5]),
                "side": "left" if hand[9] == 0 else "right",
            })

    if obj_dets is not None:
        for obj in obj_dets:
            if obj[4] > args.thresh_obj:
                image_data["objects"].append({
                    "bbox": obj[0:4].tolist(),
                    "score": float(obj[4]),
                    "contact_state": int(obj[5]),
                })

    json_path = os.path.join(output_dir, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(image_data, f, indent=2)

    im_vis = vis_detections_filtered_objects_PIL(
        np.copy(im), obj_dets, hand_dets,
        args.thresh_hand, args.thresh_obj
    )
    vis_path = os.path.join(output_dir, "detection.png")
    im_vis.save(vis_path)

    return image_data


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    # Resolve image_dir + output_root to absolute paths BEFORE chdir
    args.image_dir = os.path.abspath(args.image_dir)
    args.output_root = os.path.abspath(args.output_root)
    args.detector_repo = os.path.abspath(args.detector_repo)

    # Change into detector repo so its imports + config load the same way as demo.py
    os.chdir(args.detector_repo)

    sys.path.insert(0, args.detector_repo)
    import _init_paths
    from model.utils.config import cfg, cfg_from_file, cfg_from_list
    from model.utils.blob import im_list_to_blob
    from model.utils.net_utils import vis_detections_filtered_objects_PIL
    cfg_path = os.path.join(args.detector_repo, args.cfg_file)
    if os.path.exists(cfg_path):
        cfg_from_file(cfg_path)
    cfg_from_list(["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"])
    cfg.USE_GPU_NMS = True
    cfg.CUDA = True
    np.random.seed(cfg.RNG_SEED)

    pascal_classes = np.asarray(["__background__", "targetobject", "hand"])

    extensions = {".jpg", ".jpeg", ".png"}
    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    if not image_files:
        print(f"[Stage 1] No images found in {args.image_dir}")
        sys.exit(0)

    print(f"[Stage 1] Found {len(image_files)} images to process.")

    if args.skip_existing:
        pending = []
        for f in image_files:
            name = os.path.splitext(f)[0]
            out_dir = os.path.join(args.output_root, name)
            if os.path.exists(os.path.join(out_dir, "metadata.json")):
                print(f"[Stage 1] Skipping {f} (already complete)")
            else:
                pending.append(f)
        image_files = pending

    if not image_files:
        print("[Stage 1] All images already processed.")
        sys.exit(0)

    fasterRCNN = load_model(args, cfg, pascal_classes)
    fasterRCNN.cuda()
    fasterRCNN.eval()

    completed, failed = 0, 0

    for image_filename in image_files:
        image_name = os.path.splitext(image_filename)[0]
        image_path = os.path.join(args.image_dir, image_filename)
        output_dir = os.path.join(args.output_root, image_name)

        print(f"[Stage 1] ({completed + failed + 1}/{len(image_files)}) {image_filename}")

        im = cv2.imread(image_path)
        if im is None:
            print(f"[Stage 1] WARNING: Could not read {image_path}, skipping.")
            failed += 1
            continue

        try:
            obj_dets, hand_dets = run_detection(
                im, fasterRCNN, args, cfg, pascal_classes, im_list_to_blob
            )

            os.makedirs(output_dir, exist_ok=True)

            image_data = save_outputs(
                im, obj_dets, hand_dets, args, output_dir,
                image_filename, vis_detections_filtered_objects_PIL
            )

            n_obj = len(image_data["objects"])
            n_hand = len(image_data["hands"])
            best = max((o["score"] for o in image_data["objects"]), default=0.0)
            print(f"[Stage 1]   → {n_obj} object(s), {n_hand} hand(s), "
                  f"best score: {best:.4f}")
            completed += 1

        except Exception as e:
            print(f"[Stage 1] ERROR on {image_filename}: {e}")
            traceback.print_exc()
            failed += 1
            continue

    print(f"\n[Stage 1] Done. {completed} completed, {failed} failed.")


if __name__ == "__main__":
    main()