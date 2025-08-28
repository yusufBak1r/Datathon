import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Verileri oku
train = pd.read_csv("C:/Users/levent/OneDrive/Masaüstü/datathon-2025/train.csv")
test = pd.read_csv("C:/Users/levent/OneDrive/Masaüstü/datathon-2025/test.csv")
sample_submission = pd.read_csv("C:/Users/levent/OneDrive/Masaüstü/datathon-2025/sample_submission.csv")

# Zamanı datetime yap
train["event_time"] = pd.to_datetime(train["event_time"])
test["event_time"] = pd.to_datetime(test["event_time"])

# Yeni zaman tabanlı özellikler ekleyen fonksiyon
def add_time_features(df):
    df['event_hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

# Özellikleri train ve test veri setlerine ekle
train = add_time_features(train)
test = add_time_features(test)

# Güncellenmiş Train için session bazlı özet özellikler
train_features = train.groupby("user_session").agg(
    session_length=("event_time", lambda x: (x.max() - x.min()).total_seconds()),
    n_events=("event_type", "count"),
    n_unique_products=("product_id", "nunique"),
    n_unique_categories=("category_id", "nunique"),
    n_unique_event_types=("event_type", "nunique"),
    user_id=("user_id", "first"),
    # Yeni eklenen özelliklerin ortalama değerlerini alalım
    avg_event_hour=("event_hour", "mean"),
    most_common_day=("day_of_week", lambda x: x.mode()[0] if not x.mode().empty else -1),
    is_weekend_session=("is_weekend", "max") # O seansta bir hafta sonu etkinliği varsa
).reset_index()

# Hedef ekle
session_values = train.groupby("user_session")["session_value"].mean().reset_index()
train_features = train_features.merge(session_values, on="user_session")

# Güncellenmiş Test için özet
test_features = test.groupby("user_session").agg(
    session_length=("event_time", lambda x: (x.max() - x.min()).total_seconds()),
    n_events=("event_type", "count"),
    n_unique_products=("product_id", "nunique"),
    n_unique_categories=("category_id", "nunique"),
    n_unique_event_types=("event_type", "nunique"),
    user_id=("user_id", "first"),
    # Yeni eklenen özelliklerin ortalama değerlerini alalım
    avg_event_hour=("event_hour", "mean"),
    most_common_day=("day_of_week", lambda x: x.mode()[0] if not x.mode().empty else -1),
    is_weekend_session=("is_weekend", "max")
).reset_index()

# Label Encoding (hem train hem test birlikte)
le = LabelEncoder()
all_user_ids = pd.concat([train_features["user_id"], test_features["user_id"]])
le.fit(all_user_ids)

train_features["user_id_enc"] = le.transform(train_features["user_id"])
test_features["user_id_enc"] = le.transform(test_features["user_id"])

# Özellikler listesini güncelle
feature_cols = ["session_length", "n_events", "n_unique_products",
                "n_unique_categories", "n_unique_event_types", "user_id_enc",
                "avg_event_hour", "most_common_day", "is_weekend_session"] # Yeni özellikler eklendi

X = train_features[feature_cols]
y = train_features["session_value"]
X_test_final = test_features[feature_cols]

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Validation skoru
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("Validation RMSE:", rmse)

# Test tahminleri
test_preds = model.predict(X_test_final)

# Submission dosyası
#submission = sample_submission.copy()
#submission["session_value"] = test_preds
#submission.to_csv("submission.csv", index=False)
print("submission.csv oluşturuldu!")
print("Mean session_value:", y.mean())
print("Median session_value:", y.median())