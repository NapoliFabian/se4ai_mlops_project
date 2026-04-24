from pydantic import BaseModel

class News(BaseModel):
    text: str
    label: int