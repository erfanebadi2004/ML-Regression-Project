from typing import List
from pydantic import BaseModel

class FeaturesInput(BaseModel):
    store_nbr: int
    family: object
    onpromotion: int
    city: object
    state: object
    type: object
    cluster: int
    holiday: int
    year: int
    month: int
    day: int
    day_of_week: int
    is_weekend: int
    is_month_end: int

class BatchPredictionRequest(BaseModel):
    samples: List[FeaturesInput]