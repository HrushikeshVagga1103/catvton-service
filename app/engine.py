import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download

# Importing from the CatVTON repo
from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from diffusers.image_processor import VaeImageProcessor

class CatVTONEngine:
    def __init__(self):
        self.device = "cuda"
        # 1. Ensure weights are downloaded
        # This matches the app.py logic to get the local path for checkpoints
        self.repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
        
        # 2. Initialize Pipeline (Matching app.py exactly)
        self.pipeline = CatVTONPipeline(
            base_ckpt="booksforcharlie/stable-diffusion-inpainting",
            attn_ckpt=self.repo_path,
            attn_ckpt_version="mix",
            weight_dtype=init_weight_dtype("bf16"), # Use "bf16" for RunPod GPUs
            use_tf32=True,
            device=self.device
        )

        # 3. Initialize Masking Tools
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8, 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True
        )
        self.automasker = AutoMasker(
            densepose_ckpt=os.path.join(self.repo_path, "DensePose"),
            schp_ckpt=os.path.join(self.repo_path, "SCHP"),
            device=self.device,
        )

    def infer(self, person_img: Image.Image, garment_img: Image.Image, garment_type: str):
        # 1. Preprocessing (Standard CatVTON sizes)
        #    Keep track of original resolution so we can restore it later.
        orig_width, orig_height = person_img.size
        width, height = 768, 1024

        # Use padding instead of cropping for the person image so we don't
        # lose parts of the body, hands, or accessories (e.g. headphones).
        person_img = resize_and_padding(person_img, (width, height))
        garment_img = resize_and_padding(garment_img, (width, height))

        # 2. Generate Mask automatically
        mask = self.automasker(person_img, garment_type)['mask']
        # A slightly smaller blur keeps the edited region closer to the clothing
        # area and helps preserve nearby details like hands and accessories.
        mask = self.mask_processor.blur(mask, blur_factor=3)

        # 3. Run Inference
        # We use a fixed seed of 42 for consistency, or you can randomize it
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        result_image = self.pipeline(
            image=person_img,
            condition_image=garment_img,
            mask=mask,
            num_inference_steps=50,
            guidance_scale=2.5,
            generator=generator
        )[0] # pipeline returns a list, we take the first image
        
        # 4. Project back to the original canvas size. We scale the CatVTON
        #    output to fully cover the original canvas, then center-crop so
        #    there is no padding or background bands left.
        if result_image.size != (orig_width, orig_height):
            # Scale to cover the original resolution.
            scale = max(orig_width / width, orig_height / height)
            new_w = int(width * scale)
            new_h = int(height * scale)

            fitted = result_image.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - orig_width) // 2
            top = (new_h - orig_height) // 2
            right = left + orig_width
            bottom = top + orig_height
            result_image = fitted.crop((left, top, right, bottom))

        return result_image