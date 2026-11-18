from typing import List, Literal
from pydantic import BaseModel, Field

class FeaturesInput(BaseModel):

    """
    اعتبار سنجی ویژگی های یک نمونه برای جلوگیری از ورود داده ها با ساختار نادرست به مدل
    """

    store_nbr: int
    family: object = Literal['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS',
                            'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
                            'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
                            'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES',
                            'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
                            'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
                            'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY',
                            'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
                            'SEAFOOD']
    onpromotion: int
    city: object = Literal['Quito', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra',
                           'Santo Domingo', 'Guaranda', 'Puyo', 'Ambato', 'Guayaquil',
                           'Salinas', 'Daule', 'Babahoyo', 'Quevedo', 'Playas', 'Libertad',
                           'Cuenca', 'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen']

    state: object = Literal['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
                            'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
                            'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
                            'El Oro', 'Esmeraldas', 'Manabi']
    type: object = Literal['D', 'C', 'B', 'E', 'A']
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