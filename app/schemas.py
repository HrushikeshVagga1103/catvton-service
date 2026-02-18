from pydantic import BaseModel

class TryOnRequest(BaseModel):
    person_url: str
    garment_url: str
    garment_type: str = "upper"  # options: "upper", "lower", "overall"