import torch
from CatVTON.model.pipeline import CatVTONPipeline
from CatVTON.model.cloth_masker import AutoMasker

class CatVTONEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the pre-trained pipeline
        self.pipeline = CatVTONPipeline.from_pretrained(
            "zhengchong/CatVTON", 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        # Load the AutoMasker for automatic garment detection
        self.mask_generator = AutoMasker(device=self.device)

    def infer(self, person_img, garment_img, garment_type):
        # 1. Generate the agnostic mask automatically
        mask_result = self.mask_generator(person_img, garment_type)
        mask = mask_result['mask']

        # 2. Run Diffusion
        with torch.inference_mode():
            result = self.pipeline(
                image=person_img,
                condition_image=garment_img,
                mask=mask,
                num_inference_steps=40,
                guidance_scale=2.5
            ).images[0]
        return result