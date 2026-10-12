from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor

# ساخت pipeline برای انجام عملیات های پیش پردازش و مدل سازی برای پیش بینی هایی از نوع regression
def pipeline_lgbm_regressor(numeric_features=None, category_features=None):
    # انجام پیش پردازش های مختص به ویژگی های عددی
    # در ابتدا مقادیر گمشده با مقدار میانه جایگزاری میشوند سپس تنظیم مقیاس ویژگی ها انجام میشود
    numeric_transformer = Pipeline(steps=[
        ("MissingValues", SimpleImputer(strategy="median")),
        ("StandardScaler", StandardScaler())
    ])

    # انجام پیش پردازش های مختص به ویژگی های دسته ای
    # در ابتدا مقادیر گمشده با مقدار most frequent جایگزاری میشوند و سپس هر مقدار منحصر به فرد را تبدیل به بردار های 0 و 1 میکنیم
    category_transformer = Pipeline(steps=[
        ("MissingValues", SimpleImputer(strategy="most_frequent")),
        ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # با استفاده از ColumnTransformer پیش پردازش های عددی و دسته ای به صورت همزمان انجام میشوند و در نهایت با یک دیگر ادغام خواهند شد
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("category", category_transformer, category_features)
    ])

    # پیاده سازی Pipeline نهایی
    # انجام فرایند پیش پردازش های عددی و دسته ای به صورت یکپارچه و مدل سازی
    full_pipeline = Pipeline(steps=[
        ("Preprocess", preprocessor),
        ("Regressor",LGBMRegressor(n_estimators=200, metric=["tweedie"], objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.3, random_state=1234)),
    ])

    return full_pipeline