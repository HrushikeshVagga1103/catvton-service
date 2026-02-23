import io
import uuid
from google.cloud import storage
from PIL import Image

class GCSManager:
    def __init__(self, bucket_name: str):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

    def load_image_from_url(self, url: str) -> Image.Image:
        # 1. Handle full URLs (strip the prefix if present)
        # From: https://storage.googleapis.com/vton_gen_ai/customers/path/img.jpg
        # To: customers/path/img.jpg
        prefix = f"https://storage.googleapis.com/{self.bucket.name}/"
        if url.startswith(prefix):
            blob_name = url.replace(prefix, "")
        elif url.startswith("gs://"):
            blob_name = url.replace(f"gs://{self.bucket.name}/", "")
        else:
            # 2. Assume it's already a clean path like "customers/..."
            blob_name = url

        # REMOVED: blob_name = url.split("/")[-1] <--- This was the culprit!

        blob = self.bucket.blob(blob_name)
        return Image.open(io.BytesIO(blob.download_as_bytes())).convert("RGB")

    def upload_pil_image(self, image: Image.Image, folder: str = "outputs") -> str:
        filename = f"{folder}/{uuid.uuid4()}.png"
        blob = self.bucket.blob(filename)
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        blob.upload_from_string(buffer.getvalue(), content_type="image/png")
        
        # Returning a public-friendly URL is often more useful than gs://
        return f"https://storage.googleapis.com/{self.bucket.name}/{filename}"