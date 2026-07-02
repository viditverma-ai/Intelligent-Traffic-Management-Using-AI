import pandas as pd

# Load traffic data
df = pd.read_csv("traffic_data.csv")

print("===== TRAFFIC DATA PREVIEW =====")
print(df.head())

print("\n===== BASIC DATA INFO =====")
print(df.info())

print("\n===== CONGESTION LEVEL COUNT =====")
print(df["congestion_level"].value_counts())

print("\n===== AVERAGE VEHICLE COUNT BY DAY =====")
avg_by_day = df.groupby("day_of_week")["vehicle_count"].mean()
print(avg_by_day)

# -------------------------------
# SMART SIGNAL SUGGESTION
# -------------------------------
def suggest_signal_time(vehicle_count):
    if vehicle_count >= 150:
        return 80
    elif vehicle_count >= 120:
        return 70
    elif vehicle_count >= 90:
        return 60
    elif vehicle_count >= 60:
        return 45
    else:
        return 30

df["suggested_signal_time"] = df["vehicle_count"].apply(suggest_signal_time)

print("\n===== SMART SIGNAL SUGGESTIONS =====")
print(df[["time_of_day", "vehicle_count", "signal_time", "suggested_signal_time"]].head(15))

# -------------------------------
# MANUAL CONGESTION PREDICTION LOGIC
# -------------------------------
def predict_congestion(vehicle_count, avg_speed):
    if vehicle_count >= 140 or avg_speed <= 20:
        return "High"
    elif vehicle_count >= 90 or avg_speed <= 30:
        return "Medium"
    else:
        return "Low"

df["predicted_congestion"] = df.apply(
    lambda row: predict_congestion(row["vehicle_count"], row["avg_speed"]),
    axis=1
)

print("\n===== PREDICTED CONGESTION LEVEL =====")
print(df[["vehicle_count", "avg_speed", "congestion_level", "predicted_congestion"]].head(15))

# -------------------------------
# CHECK PREDICTION MATCH %
# -------------------------------
correct = (df["congestion_level"] == df["predicted_congestion"]).sum()
total = len(df)
accuracy = (correct / total) * 100

print("\n===== SIMPLE MODEL ACCURACY =====")
print(f"Matched Predictions: {correct}/{total}")
print(f"Accuracy: {accuracy:.2f}%")

# -------------------------------
# SAMPLE USER INPUT TEST
# -------------------------------
print("\n===== SAMPLE TRAFFIC TEST =====")
sample_vehicle = 135
sample_speed = 22

sample_prediction = predict_congestion(sample_vehicle, sample_speed)
sample_signal = suggest_signal_time(sample_vehicle)

print(f"Vehicle Count: {sample_vehicle}")
print(f"Average Speed: {sample_speed}")
print(f"Predicted Congestion: {sample_prediction}")
print(f"Suggested Signal Time: {sample_signal} seconds")