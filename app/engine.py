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
        width, height = 768, 1024
        person_img = resize_and_crop(person_img, (width, height))
        garment_img = resize_and_padding(garment_img, (width, height))

        # 2. Generate Mask automatically
        mask = self.automasker(person_img, garment_type)['mask']
        mask = self.mask_processor.blur(mask, blur_factor=9)

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

        return result_image