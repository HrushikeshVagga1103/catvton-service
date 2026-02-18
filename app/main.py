from fastapi import FastAPI, HTTPException
from .schemas import TryOnRequest
from .storage import GCSManager
from .engine import CatVTONEngine

app = FastAPI()
gcs = GCSManager(bucket_name="your-garment-bucket")
model_engine = None

@app.on_event("startup")
async def startup_event():
    global model_engine
    model_engine = CatVTONEngine()

@app.post("/generate-tryon")
async def generate(payload: TryOnRequest):
    try:
        # Step 1: Download
        p_img = gcs.load_image_from_url(payload.person_url)
        g_img = gcs.load_image_from_url(payload.garment_url)

        # Step 2: Process
        res_img = model_engine.infer(p_img, g_img, payload.garment_type)

        # Step 3: Upload
        result_url = gcs.upload_pil_image(res_img)
        
        return {"status": "success", "output_url": result_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))