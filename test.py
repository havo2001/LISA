import argparse
import os
import sys
from functools import partial
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
    return sorted(list(lora_module_names))


def validate(val_loader, model, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    acc_dice_meter = AverageMeter("gDice", ":6.3f", Summary.SUM)

    model.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].float()          # SAM stays fp32
            input_dict["images_clip"] = input_dict["images_clip"].half() # CLIP can stay fp16
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].float()               # SAM stays fp32
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        # Save predicted mask images
        image_path = input_dict["image_paths"][0]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        for i, output_i in enumerate(output_list):
            pred_mask = output_i.cpu().numpy().astype(bool)

            # Binary mask (mask * 100 for visibility)
            mask_save_path = os.path.join(args.vis_save_path, f"{image_name}_mask_{i}.jpg")
            cv2.imwrite(mask_save_path, pred_mask.astype(np.uint8) * 100)

            # Red overlay on original image
            overlay_save_path = os.path.join(args.vis_save_path, f"{image_name}_masked_img_{i}.jpg")
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(overlay_save_path, save_img)

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        acc_dice = 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target

            dice_i = 2 * intersection_i / (union_i + intersection_i + 1e-5)
            dice_i[union_i == 0] = 1.0    # no-object target
            acc_dice += dice_i

        acc_dice = acc_dice.cpu().numpy() / masks_list.shape[0]
        acc_dice_meter.update(acc_dice, n=masks_list.shape[0])

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    # No all_reduce — single GPU, no distributed
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    dice_class = 2 * intersection_meter.sum / (union_meter.sum + intersection_meter.sum + 1e-10)
    cdice = dice_class[1]
    gdice = acc_dice_meter.avg[1]

    if writer is not None:
        writer.add_scalar("val/giou", giou, 0)
        writer.add_scalar("val/ciou", ciou, 0)
        writer.add_scalar("val/gdice", gdice, 0)
        writer.add_scalar("val/cdice", cdice, 0)

    print("=" * 50)
    print("BUSI Evaluation Results")
    print(f"  gIoU  (mean per-sample IoU):  {giou:.4f}")
    print(f"  cIoU  (cumulative IoU):        {ciou:.4f}")
    print(f"  gDice (mean per-sample Dice):  {gdice:.4f}")
    print(f"  cDice (cumulative Dice):       {cdice:.4f}")
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