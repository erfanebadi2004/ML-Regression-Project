import logging
import pandas as pd
import mlflow.sklearn
import joblib
from pipeline_LGBMRegressor import pipeline_lgbm_regressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# تنظیمات مربوط به logging برای ذخیره گزارش ها و نمایش در کنسول
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("app_info.log"), logging.StreamHandler()])

# ایجاد شی logger برای ثبت وقایع و فرایند آموزش مدل
logger = logging.getLogger("Training Model")

# بارگزاری داده ها
df_store = pd.read_csv("Files/store_sales.csv")

# در پیش بینی های سری زمانی نمیتوان تکفیک داده های را به صورت تصادفی انجام داد
# باید از یک نقطه زمانی مشخصی داده های آموزش و تست رو از هم تفکیک کرد
split_date = "2017-07-15"

train = df_store[df_store["date"] < split_date]
test = df_store[df_store["date"] >= split_date]

# جدا سازی متغیر هدف y از ویژگی ها X
y_train = train["sales"].values.reshape(-1)
X_train = train.drop(columns=train[["date", "sales"]])

y_test = test["sales"].values.reshape(-1)
X_test = test.drop(columns=test[["date", "sales"]])

# ذخیره داده های تست برای استفاده در مراحل تست های واحد در فایل test_model.py
X_test.to_csv('prediction_test.csv', index=False)

# تعریف لیست ویژگی های عددی و دسته ای برای استفاده در pipeline
numeric_list = ["onpromotion", "cluster", "month", "day", "day_of_week", "year"]
category_list = ["family", "city", "state", "type", "holiday", "is_weekend", "is_month_end", "store_nbr"]

#ذخیره داده های آموزشی به عنوان فایل مرجع برای انجام Monitoring
# ذخیره نام ویژگی ها برای استفاده در API
X_train.to_csv("reference.csv", index=False)
joblib.dump(X_train.columns.to_list(), filename="Files/features_name.pkl")

# ایجاد pipeline پیش پردازش و مدل سازی
pipeline = pipeline_lgbm_regressor(numeric_features=numeric_list, category_features=category_list)

# آموزش مدل روی داده های train
logger.info("Starting the model training process")
pipeline.fit(X_train, y_train)

# پیش بینی روی داده های test برای ارزیابی مدل
logger.info("Starting the evaluation phase using test data.")
predict_test = pipeline.predict(X_test)

# معیار های ارزیابی مدل
r2 = r2_score(y_test, predict_test)
mae = mean_absolute_error(y_test, predict_test)
mse = root_mean_squared_error(y_test, predict_test)
logger.info(f"r2 Score : {r2}, MAE Score : {mae}, MSE Score : {mse}")

skops_trusted_types = [
    'imblearn.over_sampling._smote.base.SMOTE',
    'imblearn.pipeline.Pipeline',
    'sklearn.metrics._dist_metrics.EuclideanDistance64',
    'sklearn.neighbors._kd_tree.KDTree',
    'collections.OrderedDict',
    'lightgbm.basic.Booster',
    'lightgbm.sklearn.LGBMRegressor',
    'numpy.dtype'
]

# ثبت پارامتر های lightgbm به صورت خودکار
mlflow.lightgbm.autolog()

# اتصال به سرور Mlflow
tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("StoreSalesTimeSeries")

# شروع یک start run جدید برای ثبت نتایج مدل در سرور Mlflow
with mlflow.start_run():
    # ثبت پارامتر های مدل به صورت دستی
    params = {"n_estimators": 200, "objective": "tweedie", "tweedie_variance_power": 1.2, "learning_rate": 0.3}
    mlflow.log_params(params)
    logger.info("Registering model parameters")

    # ثبت معیار های ارزیابی
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    logger.info("Registering model evaluation metrics")

    # ذخیره و ثبت مدل در mlflow registry
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        skops_trusted_types=skops_trusted_types,
        name="LightgbmRegressor_model",
        registered_model_name="LightgbmRegressor",
    )
    logger.info("Registering and saving the trained model")

    print(f"✅ Success! Model logged via mlflow.")

# دانلود فایل های artifact مدل جهت استقرار مدل
model_uri = "models:/LightgbmRegressor/latest"
download_path = "./LgbmRegressor_folder"

mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=download_path)
logger.info("Download model artifact files from MLflow.")