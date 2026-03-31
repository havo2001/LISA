import argparse
import os
import sys
from functools import partial
from typing import Any
os.environ["MPLBACKEND"] = "Agg"
import cv2
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.busi import BUSIValDataset
from utils.dataset import collate_fn
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter,
    Summary,
    dict_to_cuda,
    intersectionAndUnionGPU,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA BUSI Evaluation (no deepspeed)")
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
    )
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    # BUSI-specific
    parser.add_argument("--busi_json", default="utils/busi.json", type=str,
                        help="Path to the busi.json file")
    parser.add_argument("--workers", default=4, type=int)
    # Checkpoint: path to a merged fp32 .pth file (converted from deepspeed)
    parser.add_argument("--checkpoint", default="", type=str,
                        help="Path to merged model checkpoint (.pth)")
    # TensorBoard
    parser.add_argument("--log_dir", default="./runs/busi_test", type=str)
    parser.add_argument("--vis_save_path", default="./vis_output/busi", type=str,
                        help="Directory to save predicted mask images")
    return parser.parse_args(args)


def find_linear_layers(model, lora_target_modules):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                x not in name
                for x in ["visual_model", "vision_tower", "mm_projector", "text_hidden_fcs"]
            )
            and any(x in name for x in lora_target_modules)
        ):
            lora_module_names.add(name)
    return sorted(list[Any](lora_module_names))


def validate(val_loader, model, writer, args):
    iou_list = []
    dice_list = []
    total_intersection = 0.0
    total_union = 0.0
    total_pred = 0.0
    total_gt = 0.0

    model.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model(**input_dict)

        pred_masks = output_dict["pred_masks"]      # list
        gt_masks = output_dict["gt_masks"][0]       # [num_masks, H, W]
        assert len(pred_masks) == 1

        # original image for visualization
        image_path = input_dict["image_paths"][0]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        for i, (pred_i, gt_i) in enumerate(zip(pred_masks[0], gt_masks)):
            # pred_i: logits -> binary
            pred_i = pred_i.detach().float()
            gt_i = gt_i.detach().float()

            # Make prediction same size as GT for metrics
            if pred_i.shape != gt_i.shape:
                pred_i = torch.nn.functional.interpolate(
                    pred_i[None, None],
                    size=gt_i.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

            pred_bin = (pred_i > 0).to(torch.uint8)
            gt_bin = (gt_i > 0).to(torch.uint8)

            pred_np = pred_bin.cpu().numpy()
            gt_np = gt_bin.cpu().numpy()

            # ----- metrics -----
            intersection = np.logical_and(pred_np == 1, gt_np == 1).sum()
            pred_area = (pred_np == 1).sum()
            gt_area = (gt_np == 1).sum()
            union = pred_area + gt_area - intersection

            if union == 0:
                iou = 1.0
            else:
                iou = intersection / (union + 1e-8)

            if pred_area + gt_area == 0:
                dice = 1.0
            else:
                dice = 2.0 * intersection / (pred_area + gt_area + 1e-8)

            iou_list.append(iou)
            dice_list.append(dice)

            total_intersection += intersection
            total_union += union
            total_pred += pred_area
            total_gt += gt_area

            # ----- save binary prediction mask -----
            mask_save_path = os.path.join(args.vis_save_path, f"{image_name}_mask_{i}.png")
            cv2.imwrite(mask_save_path, pred_np.astype(np.uint8) * 255)

            # ----- overlay on original image -----
            pred_vis = pred_np
            if pred_vis.shape != image_np.shape[:2]:
                pred_vis = cv2.resize(
                    pred_vis.astype(np.uint8),
                    (image_np.shape[1], image_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                pred_vis = pred_vis.astype(bool)

            save_img = image_np.copy()
            save_img[pred_vis] = (
                0.5 * save_img[pred_vis] + 0.5 * np.array([255, 0, 0])
            ).astype(np.uint8)

            overlay_save_path = os.path.join(args.vis_save_path, f"{image_name}_masked_img_{i}.png")
            save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(overlay_save_path, save_img_bgr)

        # optional: also save GT mask for debugging
        # gt_save_path = os.path.join(args.vis_save_path, f"{image_name}_gt_{i}.png")
        # cv2.imwrite(gt_save_path, gt_np.astype(np.uint8) * 255)

    giou = float(np.mean(iou_list)) if len(iou_list) > 0 else 0.0
    gdice = float(np.mean(dice_list)) if len(dice_list) > 0 else 0.0

    ciou = float(total_intersection / (total_union + 1e-8)) if total_union > 0 else 1.0
    cdice = float(2.0 * total_intersection / (total_pred + total_gt + 1e-8)) if (total_pred + total_gt) > 0 else 1.0

    if writer is not None:
        writer.add_scalar("val/giou", giou, 0)
        writer.add_scalar("val/ciou", ciou, 0)
        writer.add_scalar("val/gdice", gdice, 0)
        writer.add_scalar("val/cdice", cdice, 0)

    print("=" * 50)
    print("BUSI Evaluation Results")
    print(f"  gIoU  (mean per-sample IoU):  {giou:.4f}")
    print(f"  cIoU  (cumulative IoU):       {ciou:.4f}")
    print(f"  gDice (mean per-sample Dice): {gdice:.4f}")
    print(f"  cDice (cumulative Dice):      {cdice:.4f}")
    print("=" * 50)

    return giou, ciou, gdice, cdice


def main(args):
    args = parse_args(args)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.vis_save_path, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # Model
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }

    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device="cuda")
    model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # LoRA
    if args.lora_r > 0:
        lora_target_modules = find_linear_layers(model, args.lora_target_modules.split(","))
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))

    # Load checkpoint (must be a merged fp32 .pth, NOT a raw deepspeed folder)
    # To convert a deepspeed checkpoint first run:
    #   python -m deepspeed.utils.zero_to_fp32 <ckpt_dir> <output.pth>
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded.")

    model = model.cuda()
    model.eval()

    # Dataset
    val_dataset = BUSIValDataset(
        json_path=args.busi_json,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        image_size=args.image_size,
    )
    print(f"Evaluating on {len(val_dataset)} BUSI samples.")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=0,
        ),
    )

    validate(val_loader, model, writer, args)


if __name__ == "__main__":
    main(sys.argv[1:])