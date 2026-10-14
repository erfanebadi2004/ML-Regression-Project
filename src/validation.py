from typing import List
from pydantic import BaseModel

class FeaturesInput(BaseModel):

    """
    اعتبار سنجی ویژگی های یک نمونه برای جلوگیری از ورود داده ها با ساختار نادرست به مدل
    """

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

    """
    تعیین کردن ساختار داده های ورودی به مدل و مقدار متناظر با فیلد sample لیستی از شی FeaturesInput است
    """

    samples: List[FeaturesInput]