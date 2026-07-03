import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("===== INTELLIGENT TRAFFIC MANAGEMENT USING AI =====\n")

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("traffic_data.csv")

print("===== TRAFFIC DATA PREVIEW =====")
print(df.head())
print("\n===== BASIC DATA INFO =====")
print(df.info())

# =========================
# 2. CLEAN CONGESTION LABELS
# =========================
df["congestion_level"] = df["congestion_level"].astype(str).str.strip().str.title()

print("\n===== CONGESTION LEVEL COUNT =====")
print(df["congestion_level"].value_counts())

# =========================
# 3. BASIC ANALYSIS
# =========================
print("\n===== AVERAGE VEHICLE COUNT BY DAY =====")
avg_by_day = df.groupby("day_of_week")["vehicle_count"].mean()
print(avg_by_day)

# =========================
# 4. SMART SIGNAL SUGGESTION LOGIC
# =========================
def suggest_signal_time(vehicle_count):
    if vehicle_count >= 150:
        return 80
    elif vehicle_count >= 120:
        return 70
    elif vehicle_count >= 90:
        return 60
    else:
        return 45

df["suggested_signal_time"] = df["vehicle_count"].apply(suggest_signal_time)

print("\n===== SMART SIGNAL SUGGESTIONS =====")
print(df[["time_of_day", "vehicle_count", "signal_time", "suggested_signal_time"]].head(15))

# =========================
# 5. MACHINE LEARNING MODEL
# =========================
print("\n===== TRAINING ML MODEL =====")

# Features and target
X = df[["vehicle_count", "avg_speed"]]
y = df["congestion_level"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model training completed successfully!")
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

# =========================
# 6. SHOW PREDICTION COMPARISON
# =========================
results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred

print("\n===== PREDICTION RESULTS SAMPLE =====")
print(results.head(10))

# =========================
# 7. TEST WITH NEW SAMPLE TRAFFIC DATA
# =========================
print("\n===== SAMPLE TRAFFIC TEST =====")

sample_vehicle_count = 135
sample_avg_speed = 22

sample_data = pd.DataFrame({
    "vehicle_count": [sample_vehicle_count],
    "avg_speed": [sample_avg_speed]
})

predicted_congestion = model.predict(sample_data)[0]
suggested_time = suggest_signal_time(sample_vehicle_count)
print(f"Average Speed: {sample_avg_speed}")
print(f"Predicted Congestion: {predicted_congestion}")
print(f"Suggested Signal Time: {suggested_time} seconds")

print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")