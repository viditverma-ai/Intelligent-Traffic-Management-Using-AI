import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("traffic_data.csv")
df["congestion_level"] = df["congestion_level"].astype(str).str.strip()

# Encode categorical columns
le_time = LabelEncoder()
le_day = LabelEncoder()
le_label = LabelEncoder()

df["time_of_day"] = le_time.fit_transform(df["time_of_day"])
df["day_of_week"] = le_day.fit_transform(df["day_of_week"])
df["congestion_level"] = le_label.fit_transform(df["congestion_level"])
X = df[
    [
        "vehicle_count",
        "avg_speed",
        "time_of_day",
        "day_of_week",
        "signal_time",
    ]
].values

# Target
y = df["congestion_level"].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# CNN input shape
X = X.reshape((X.shape[0], X.shape[1], 1))

# One-hot encoding
print(df["congestion_level"].unique())
print(df["congestion_level"].value_counts())
y = to_categorical(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN Model
print("Output Classes =", y.shape[1])

model = Sequential([
    Conv1D(32, 2, activation="relu", input_shape=(5,1)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(y.shape[1], activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training CNN...")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

loss, acc = model.evaluate(X_test, y_test)

print(f"\nCNN Accuracy: {acc*100:.2f}%")

model.save("cnn_model.keras")

print("\nCNN model saved successfully!")