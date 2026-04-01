import json
import os
import pandas as pd
import PIL.Image as Image

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IMAGE_TOKEN
from model.llava import conversation as conversation_lib


def create_busi_json(data_root, prompt_file, json_path):
    images = []
    df = pd.read_excel(prompt_file)
    for _, row in df.iterrows():
        image_name = row['Image']
        # the form is 102_malignant.png
        image_number = image_name.split('_')[0]
        image_class = image_name.split('_')[1].split('.')[0]
        # image_path 
        new_image_name = f"{image_class} ({image_number}).png"
        new_mask_name = f"{image_class} ({image_number})_mask.png"
        image_path = os.path.join(data_root, image_class, new_image_name)
        mask_path = os.path.join(data_root, image_class, new_mask_name)
        images.append({
            "image": image_path,
            "mask": mask_path,
            "prompt": row['Description']
        })
    with open(json_path, "w") as f:
        json.dump(images, f)


class BUSIValDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        json_path,
        tokenizer,
        vision_tower,
        image_size=1024,
        is_sentence=False,
    ):
        """
        Expected JSON format: a list of dicts like
        [
          {
            "image": "dataset/busi/test/images/xxx.png",
            "mask": "dataset/busi/test/masks/yyy.png",
            "prompt": "breast tumor"
          },
          ...
        ]
        """
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.is_sentence = is_sentence

        with open(json_path, "r") as f:
            self.samples = json.load(f)

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.samples)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad to square
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def _build_conversation(self, prompt_text: str):
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []

        prompt_text = prompt_text.strip()
        if self.is_sentence:
            user_prompt = (
                DEFAULT_IMAGE_TOKEN
                + f"\n {prompt_text} Please output segmentation mask."
            )
        else:
            user_prompt = (
                DEFAULT_IMAGE_TOKEN
                + f"\n What is {prompt_text} in this image? Please output segmentation mask."
            )

        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], "[SEG].")
        return [conv.get_prompt()]

    def _load_image(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_mask(self, mask_path: str):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        # Convert to binary {0,1}
        mask = (mask > 0).astype(np.uint8)

        # Shape: [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)
        return mask

    def __getitem__(self, idx):
        item = self.samples[idx]

        image_path = item["image"]
        mask_path = item["mask"]
        prompt = item["prompt"]

        # Load raw image/mask
        image_np = self._load_image(image_path)
        masks = self._load_mask(mask_path)  # [1, H, W]

        # Build prompt in LISA format
        conversations = self._build_conversation(prompt)

        # CLIP image
        image_clip = self.clip_image_processor.preprocess(
            image_np, return_tensors="pt"
        )["pixel_values"][0]

        # SAM image
        image_sam = self.transform.apply_image(image_np)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(
            torch.from_numpy(image_sam).permute(2, 0, 1).contiguous()
        )

        # Dummy labels: ValDataset uses ignore_label canvas
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,      # str
            image_sam,       # tensor for SAM
            image_clip,      # tensor for CLIP
            conversations,   # list[str]
            masks,           # tensor [1, H, W], binary
            labels,          # tensor [H, W], mostly placeholder
            resize,          # tuple(h, w) after SAM resize
            None,            # questions
            None,            # sampled_classes
            True,            # inference
        )



if __name__ == "__main__":
    if not os.path.exists("utils/train_busi.json"):
        create_busi_json("dataset/Dataset_BUSI_with_GT", "dataset/Train_text.xlsx", "utils/train_busi.json")
    else:
        print("utils/train_busi.json already exists")


    if not os.path.exists("utils/val_busi.json"):
        create_busi_json("dataset/Dataset_BUSI_with_GT", "dataset/Val_text.xlsx", "utils/val_busi.json")
    else:
        print("utils/val_busi.json already exists")

    if not os.path.exists("utils/test_busi.json"):
        create_busi_json("dataset/Dataset_BUSI_with_GT", "dataset/Test_text_original.xlsx", "utils/test_busi.json")
    else:
        print("utils/test_busi.json already exists")

    # test the json file
    with open("utils/train_busi.json", "r") as f:
        data = json.load(f)
    # try to load one sample image and mask together with prompt
    image = data[0]['image']
    mask = data[0]['mask']
    prompt = data[0]['prompt']
    # show the image
    image = Image.open(image)
    image.show()
    mask = Image.open(mask)
    mask.show()
    print(prompt)
   