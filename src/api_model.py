import skops.io as sio
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from validation import BatchPredictionRequest
from pathlib import Path
from monitoring_data import check_data_drift

# تعیین مسیر فایل های مدل و داده های مرجع و نام ستون ها
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_PATH = BASE_DIR / "Files" / "features_name.pkl"
REFERENCE_PATH = BASE_DIR / "Files" / "reference.csv"
LOG_FILE_PATH = BASE_DIR / "Files" / "app_logging.log"
MODEL_PATH = BASE_DIR / "LgbmRegressor_folder" / "model.skop"

# تنظیم سیستم logging برای ذخیره و نمایش گزارش ها در زمان اجرا برنامه در محیط سرور
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()])

logger = logging.getLogger("LightGBM Regressor ApiModel")

reference_df = None
LGBM_model = None

# بارگزاری داده های مرجع reference و نام ستون ها قبل از شروع کار API برای جلوگیری از خواندن مجدد فایل ها در محیط سرور
try:
    reference_df = pd.read_csv(REFERENCE_PATH)

    with open(FEATURE_PATH, "rb") as features:
        features_name = joblib.load(features)
except Exception as e:
        logger.error(f"Critical error: Reference data failed to load. {REFERENCE_PATH}: {str(e)}")

#بارگزاری مدل lightgbmRegressor با استفاده از skops برای جلوگیری از نفود فایل های مخرب به سیستم
try:
    LGBM_model = sio.load(MODEL_PATH, trusted=['imblearn.over_sampling._smote.base.SMOTE',
    'imblearn.pipeline.Pipeline',
    'sklearn.metrics._dist_metrics.EuclideanDistance64',
    'sklearn.neighbors._kd_tree.KDTree',
    'collections.OrderedDict',
    'lightgbm.basic.Booster',
    'lightgbm.sklearn.LGBMRegressor',
    'numpy.dtype'])

    logger.info(f"Model loaded successfully: {MODEL_PATH}")

except Exception as error:
    logger.exception(f"Model loading failed{str(error)}")

# راه اندازی اپلیکیشن FastAPI
app = FastAPI(title="LGBMRegressorModel_API")

# تعریف مسیر endpoint برای دریافت درخواست های پیش بینی
@app.post("/predict")
async def predict_lgbm_regressor(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    global reference_df
    global LGBM_model

    # بررسی مدل بارگزاری شده برای زمانی که مدل به درستی بارگزاری نشده خطا 500 برمیگرداند
    if LGBM_model is None:
        logger.critical("The model failed to load!")
        raise HTTPException(status_code=500, detail="Server configuration error: Model not found.")

    # زمانی که داده ای در درخواست ها به سمت سرور وجود نداشته باشد خطا 400 برمیگرداند
    if not request.samples:
        logger.warning("Empty samples received.")
        raise HTTPException(status_code=400, detail="No samples were found.")

    try:
        # داده های دریافتی که توسط pydantic اعتبار سنجی شده به DataFrame تبدیل میشوند
        df_samples = pd.DataFrame([s.model_dump() for s in request.samples])

        # انتخاب ستون های مورد نیاز و مرتب سازی بر اساس ترتیب نام ستون های feature_name
        df = df_samples[features_name]
        min_sample = 50

        if len(df) > min_sample:
            # ثبت وظایف در پس زمینه برای بررسی انحراف داده drift
            background_tasks.add_task(check_data_drift, reference_df, df)
        else:
            logger.info("The number of samples is lower than the threshold expected for data monitoring .")

        # انجام فرایند پیش بینی توسط مدل
        predict_model = LGBM_model.predict(df)

        result = predict_model.tolist()

        return {"prediction result" : result}

    except Exception as E:
        logger.error(f"Error during the prediction process: {str(E)}")
        raise HTTPException(status_code=500,detail=f"Prediction error: {str(E)}" )