import io
import uuid
from google.cloud import storage
from PIL import Image

class GCSManager:
    def __init__(self, bucket_name: str):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

    def load_image_from_url(self, url: str) -> Image.Image:
        # Assuming URL is a GCS blob name or a direct link
        blob_name = url.split("/")[-1]
        blob = self.bucket.blob(blob_name)
        return Image.open(io.BytesIO(blob.download_as_bytes())).convert("RGB")

    def upload_pil_image(self, image: Image.Image, folder: str = "outputs") -> str:
        filename = f"{folder}/{uuid.uuid4()}.png"
        blob = self.bucket.blob(filename)
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        blob.upload_from_string(buffer.getvalue(), content_type="image/png")
        
        return f"gs://{self.bucket.name}/{filename}"