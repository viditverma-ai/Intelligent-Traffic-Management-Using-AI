import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from llm_helper import traffic_ai_assistant
from tensorflow.keras.models import load_model
import joblib
cnn_model = load_model("cnn_model.keras")
print(cnn_model.output_shape)
st.write("CNN Output Shape:", cnn_model.output_shape)
scaler = joblib.load("scaler.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intelligent Traffic Management Using AI",
    page_icon="🚦",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.metric-card {
    background: #111827;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
}
.big-title {
    font-size: 40px;
    font-weight: 800;
}
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 20px;
}
.small-note {
    color: #94a3b8;
}
.recommend-box {
    background: #111827;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #38bdf8;
    margin-top: 15px;
}
.route-box {
    background: #111827;
    padding: 16px;
    border-radius: 12px;
    margin-top: 10px;
    border-left: 5px solid #22c55e;
}
.emergency-box {
    background: #3b0d0d;
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #ef4444;
    margin-top: 12px;
}
.landing-box {
    background: linear-gradient(135deg, #111827, #1e293b);
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.3);
}
.feature-card {
    background: #111827;
    padding: 18px;
    border-radius: 16px;
    min-height: 160px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
}
.admin-card {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    margin-top: 10px;
    border-left: 5px solid #f59e0b;
}
.info-card {
    background: #111827;
    padding: 16px;
    border-radius: 12px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "entered_app" not in st.session_state:
    st.session_state.entered_app = False

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("traffic_data.csv")
    return df

df = load_data()

# ---------------- DATA CLEANING ----------------
df["congestion_level"] = df["congestion_level"].astype(str).str.strip().str.title()
valid_labels = ["Low", "Medium", "High"]
df = df[df["congestion_level"].isin(valid_labels)].copy()
df = df.dropna(subset=["vehicle_count", "avg_speed", "congestion_level", "time_of_day", "day_of_week"])

# ---------------- LABEL ENCODING ----------------
label_map = {"Low": 0, "Medium": 1, "High": 2}
reverse_label_map = {0: "Low", 1: "Medium", 2: "High"}

df["congestion_encoded"] = df["congestion_level"].map(label_map)

# ---------------- FEATURES ----------------
features = ["vehicle_count", "avg_speed", "time_of_day", "day_of_week"]
X = df[features]
y = df["congestion_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL TRAINING ----------------
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

# ---------------- BEST MODEL ----------------
if rf_acc >= dt_acc:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_acc
else:
    best_model = dt_model
    best_model_name = "Decision Tree"
    best_accuracy = dt_acc

# ---------------- DAY MAP ----------------
day_map = {
    1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
    5: "Friday", 6: "Saturday", 7: "Sunday"
}

# ---------------- SIGNAL SUGGESTION ----------------
def suggest_signal_time(vehicle_count, congestion):
    if congestion == "High":
        if vehicle_count >= 150:
            return 90
        return 80
    elif congestion == "Medium":
        if vehicle_count >= 100:
            return 65
        return 55
    else:
        if vehicle_count >= 70:
            return 45
        return 35

# ---------------- EMERGENCY SIGNAL BOOST ----------------
def apply_emergency_priority(signal_time, emergency_vehicle):
    if emergency_vehicle == "Ambulance":
        return signal_time + 25
    elif emergency_vehicle == "Fire Brigade":
        return signal_time + 20
    elif emergency_vehicle == "Police":
        return signal_time + 15
    return signal_time

# ---------------- RISK SCORE ----------------
def calculate_risk_score(vehicle_count, avg_speed):
    score = (vehicle_count * 0.6) + ((60 - avg_speed) * 1.5)
    score = max(0, min(100, int(score / 2)))
    return score

# ---------------- RISK CATEGORY ----------------
def get_risk_category(risk_score):
    if risk_score >= 75:
        return "Critical Risk"
    elif risk_score >= 45:
        return "Medium Risk"
    return "Low Risk"

# ---------------- AI RECOMMENDATION ----------------
def get_ai_recommendation(congestion, risk_score, vehicle_count, avg_speed, emergency_vehicle):
    if emergency_vehicle != "None":
        return f"🚑 Emergency priority detected for {emergency_vehicle}. Immediately extend green signal time, clear junction path, and prioritize fast lane movement."

    if congestion == "High":
        if risk_score >= 75:
            return "🚨 Heavy congestion detected. Increase green signal time immediately, deploy traffic police support, and consider rerouting vehicles."
        return "⚠ High congestion detected. Increase signal time and monitor the traffic flow closely."
    elif congestion == "Medium":
        if avg_speed < 25:
            return "⚠ Moderate congestion with low speed. Slightly increase green signal time and monitor the junction."
        return "🟡 Moderate congestion. Keep normal monitoring and optimize signal timing slightly."
    else:
        return "✅ Traffic flow is smooth. No urgent action required. Keep current signal timing."

# ---------------- SEVERITY ----------------
def get_severity(congestion):
    if congestion == "High":
        return "🔴 Severe"
    elif congestion == "Medium":
        return "🟡 Moderate"
    return "🟢 Normal"

# ---------------- PEAK HOUR ANALYSIS ----------------
peak_hour = df.groupby("time_of_day")["vehicle_count"].mean().idxmax()
busiest_day = df.groupby("day_of_week")["vehicle_count"].mean().idxmax()
slowest_day = df.groupby("day_of_week")["avg_speed"].mean().idxmin()

# ---------------- ROUTE SIMULATION ----------------
def simulate_routes(vehicle_count, avg_speed, time_of_day):
    routes = []

    route_a_count = max(20, vehicle_count - np.random.randint(5, 25))
    route_a_speed = min(80, avg_speed + np.random.randint(0, 8))

    route_b_count = max(20, vehicle_count + np.random.randint(-10, 15))
    route_b_speed = max(5, avg_speed + np.random.randint(-5, 6))

    route_c_count = max(20, vehicle_count + np.random.randint(10, 35))
    route_c_speed = max(5, avg_speed - np.random.randint(0, 10))

    sample_routes = [
        ("Route A", route_a_count, route_a_speed),
        ("Route B", route_b_count, route_b_speed),
        ("Route C", route_c_count, route_c_speed),
    ]

    for route_name, vc, sp in sample_routes:
        route_risk = calculate_risk_score(vc, sp)

        if vc >= 150 or sp <= 18:
            route_congestion = "High"
        elif vc >= 90 or sp <= 30:
            route_congestion = "Medium"
        else:
            route_congestion = "Low"

        routes.append({
            "Route": route_name,
            "Vehicle Count": vc,
            "Avg Speed": sp,
            "Predicted Congestion": route_congestion,
            "Risk Score": route_risk
        })

    route_df = pd.DataFrame(routes)
    best_route = route_df.sort_values(by=["Risk Score", "Vehicle Count"], ascending=[True, True]).iloc[0]["Route"]
    return route_df, best_route

# ---------------- ADMIN STATS ----------------
def get_admin_summary(history):
    if len(history) == 0:
        return {
            "total_predictions": 0,
            "high_risk_count": 0,
            "emergency_cases": 0,
            "avg_risk": 0,
            "common_congestion": "N/A",
            "common_emergency": "N/A",
            "common_route": "N/A"
        }

    history_df = pd.DataFrame(history)

    high_risk_count = len(history_df[history_df["Risk Score"] >= 75])
    emergency_cases = len(history_df[history_df["Emergency"] != "None"])
    avg_risk = round(history_df["Risk Score"].mean(), 2)
    common_congestion = history_df["Predicted Congestion"].mode()[0]
    common_emergency = history_df["Emergency"].mode()[0]
    common_route = history_df["Best Route"].mode()[0]

    return {
        "total_predictions": len(history_df),
        "high_risk_count": high_risk_count,
        "emergency_cases": emergency_cases,
        "avg_risk": avg_risk,
        "common_congestion": common_congestion,
        "common_emergency": common_emergency,
        "common_route": common_route
    }

# =========================================================
# LANDING PAGE
# =========================================================
if not st.session_state.entered_app:
    st.markdown("""
    <div class="landing-box">
        <h1>🚦 Intelligent Traffic Management Using AI</h1>
        <h3>Smart Traffic Congestion Prediction, Risk Analysis, Emergency Priority and Route Recommendation System</h3>
        <p style="font-size:18px; color:#cbd5e1; margin-top:20px;">
            This AI-powered project predicts traffic congestion, calculates traffic risk,
            recommends optimized signal timing, supports emergency vehicle priority,
            and suggests smarter alternate routes for urban traffic management.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("✨ Key Features")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 AI Congestion Prediction</h4>
            <p>Predicts Low / Medium / High traffic congestion using machine learning models based on live traffic inputs.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <h4>🚑 Emergency Priority</h4>
            <p>Supports Ambulance, Fire Brigade and Police priority by increasing signal time and giving emergency handling recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <h4>🛣 Route Recommendation</h4>
            <p>Simulates alternate routes and recommends the best route based on traffic congestion and risk score.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Dashboard Analytics</h4>
            <p>Shows peak hour, busiest day, traffic trends, model comparison, feature importance and AI insights.</p>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown("""
        <div class="feature-card">
            <h4>📜 Prediction History</h4>
            <p>Stores live predictions and allows report download in CSV format for traffic analysis and review.</p>
        </div>
        """, unsafe_allow_html=True)

    with c6:
        st.markdown("""
        <div class="feature-card">
            <h4>🧠 Smart Signal Suggestion</h4>
            <p>Provides optimized traffic signal timing and AI-based recommendation for better traffic flow management.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    c7, c8, c9 = st.columns([1, 1, 1])
    with c8:
        if st.button("🚀 Enter Dashboard", use_container_width=True):
            st.session_state.entered_app = True
            st.rerun()

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚦 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "🤖 Live Prediction",
        "📊 AI Insights",
        "📁 Dataset Preview",
        "📜 Prediction History",
        "🛠 Admin Dashboard",
        "ℹ️ Project Info"
    ]
)

st.sidebar.write("---")
if st.sidebar.button("🔄 Reset Session"):
    st.session_state.prediction_history = []
    st.session_state.entered_app = False
    st.rerun()

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🚦 Intelligent Traffic Management Using AI</div>', unsafe_allow_html=True)
st.markdown("### Smart Traffic Congestion Prediction, Risk Analysis, Emergency Priority and Signal Time Recommendation System")
st.write("---")

# =========================================================
# PAGE 1 - DASHBOARD
# =========================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="section-title">📌 Dashboard Overview</div>', unsafe_allow_html=True)

    avg_vehicle_count = round(df["vehicle_count"].mean(), 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Best Model", best_model_name)
    col3.metric("Model Accuracy", f"{best_accuracy*100:.2f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("Peak Traffic Hour", f"{peak_hour}:00")
    col5.metric("Busiest Day", day_map[busiest_day])
    col6.metric("Avg Vehicle Count", avg_vehicle_count)

    st.write("")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("🚥 Congestion Level Distribution")
        congestion_counts = df["congestion_level"].value_counts()
        st.bar_chart(congestion_counts)

    with colB:
        st.subheader("🚗 Average Vehicle Count by Day")
        avg_day = df.groupby("day_of_week")["vehicle_count"].mean()
        avg_day.index = avg_day.index.map(day_map)
        st.line_chart(avg_day)

    st.write("")

    colC, colD = st.columns(2)

    with colC:
        st.subheader("⏰ Traffic by Time of Day")
        traffic_by_hour = df.groupby("time_of_day")["vehicle_count"].mean()
        st.area_chart(traffic_by_hour)

    with colD:
        st.subheader("🏎 Average Speed by Day")
        speed_by_day = df.groupby("day_of_week")["avg_speed"].mean()
        speed_by_day.index = speed_by_day.index.map(day_map)
        st.line_chart(speed_by_day)

    st.write("")
    st.subheader("🧠 AI Traffic Summary")
    st.info(
        f"Peak traffic occurs around **{peak_hour}:00**. "
        f"The busiest day appears to be **{day_map[busiest_day]}**, "
        f"while traffic speed is slowest on **{day_map[slowest_day]}**. "
        f"The model suggests that **vehicle count and speed** strongly affect congestion levels."
    )

    st.subheader("📖 About This System")
    st.write("""
    This AI-based traffic management system predicts congestion levels using **vehicle count, average speed, time of day, and day of week**.
    It recommends **optimized traffic signal timings**, calculates **traffic risk scores**, supports **emergency vehicle priority**,
    and provides **route suggestions** for smarter urban traffic control.
    """)

# =========================================================
# PAGE 2 - LIVE PREDICTION
# =========================================================
elif page == "🤖 Live Prediction":
    st.markdown('<div class="section-title">🤖 Live Traffic Prediction</div>', unsafe_allow_html=True)
    st.write("Enter the live traffic details below:")

    col1, col2 = st.columns(2)

    with col1:
        vehicle_count = st.number_input("Vehicle Count", min_value=0, max_value=500, value=120)

    with col2:
        avg_speed = st.number_input("Average Speed", min_value=0, max_value=120, value=25)

    col3, col4 = st.columns(2)

    with col3:
        time_of_day = st.slider("Time of Day", 0, 23, 9)

    with col4:
        day_of_week = st.selectbox(
            "Day of Week",
            [1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: day_map[x]
        )

    emergency_vehicle = st.selectbox(
        "Emergency Vehicle Priority",
        ["None", "Ambulance", "Fire Brigade", "Police"]
    )

    if st.button("Predict Traffic Status"):
        input_df = pd.DataFrame({
            "vehicle_count": [vehicle_count],
            "avg_speed": [avg_speed],
            "time_of_day": [time_of_day],
            "day_of_week": [day_of_week],
        })

        prediction = best_model.predict(input_df)[0]
        cnn_signal_time = suggest_signal_time(vehicle_count, reverse_label_map[prediction])

        cnn_df = pd.DataFrame({
         "vehicle_count": [vehicle_count],
         "avg_speed": [avg_speed],
         "time_of_day": [time_of_day],
         "day_of_week": [day_of_week],
         "signal_time": [cnn_signal_time]
})
        cnn_input = scaler.transform(cnn_df)
        cnn_input = cnn_input.reshape((1, 5, 1))
   

        

        cnn_prediction = cnn_model.predict(cnn_input)
        cnn_prediction = np.argmax(cnn_prediction, axis=1)[0]
        
       
        cnn_label = reverse_label_map[int(cnn_prediction)]
        predicted_label = reverse_label_map[prediction]

        signal_time = suggest_signal_time(vehicle_count, predicted_label)
        signal_time = apply_emergency_priority(signal_time, emergency_vehicle)

        risk_score = calculate_risk_score(vehicle_count, avg_speed)
        risk_category = get_risk_category(risk_score)
        severity = get_severity(predicted_label)
        recommendation = get_ai_recommendation(
            predicted_label, risk_score, vehicle_count, avg_speed, emergency_vehicle
        )

        route_df, best_route = simulate_routes(vehicle_count, avg_speed, time_of_day)

        st.write("---")
        st.subheader("📢 Prediction Result")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RF Prediction", predicted_label)
        c2.metric("CNN Prediction", cnn_label)
        c3.metric("Signal Time", f"{signal_time} sec")
        c4.metric("Risk Score", f"{risk_score}/100")
        c5.metric("Severity", severity)
        # Congestion alerts
        if predicted_label == "High":
            st.error("🚨 Heavy traffic detected. Immediate signal optimization recommended.")
        elif predicted_label == "Medium":
            st.warning("⚠ Moderate congestion detected. Traffic flow should be monitored.")
        else:
            st.success("✅ Traffic is smooth. Congestion level is low.")

        # Risk alerts
        if risk_score >= 75:
            st.error("🚨 Critical traffic risk detected.")
        elif risk_score >= 45:
            st.warning("⚠ Medium traffic risk detected.")
        else:
            st.success("✅ Low traffic risk.")

        # Emergency alert
        if emergency_vehicle != "None":
            st.markdown(f"""
            <div class="emergency-box">
                <h4>🚑 Emergency Priority Activated</h4>
                <p><b>{emergency_vehicle}</b> detected. Signal time has been increased for faster movement and junction clearance.</p>
            </div>
            """, unsafe_allow_html=True)

        # AI Recommendation box
        st.markdown(f"""
        <div class="recommend-box">
            <h4>🤖 AI Recommendation</h4>
            <p>{recommendation}</p>
            <p><b>Risk Category:</b> {risk_category}</p>
        </div>
        """, unsafe_allow_html=True)

        # Route Suggestion
        st.subheader("🛣 Route Suggestion Simulation")
        st.dataframe(route_df, use_container_width=True)

        best_route_row = route_df[route_df["Route"] == best_route].iloc[0]
        st.markdown(f"""
        <div class="route-box">
            <h4>✅ Best Route Recommendation: {best_route}</h4>
            <p>
                Suggested because it has comparatively lower traffic risk and smoother expected movement.
                <br><b>Congestion:</b> {best_route_row["Predicted Congestion"]}
                <br><b>Risk Score:</b> {best_route_row["Risk Score"]}/100
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Save history
        history_row = {
            "Vehicle Count": vehicle_count,
            "Average Speed": avg_speed,
            "Time of Day": time_of_day,
            "Day": day_map[day_of_week],
            "Emergency": emergency_vehicle,
            "Predicted Congestion": predicted_label,
            "Suggested Signal Time": signal_time,
            "Risk Score": risk_score,
            "Risk Category": risk_category,
            "Severity": severity,
            "Best Route": best_route
        }
        st.session_state.prediction_history.append(history_row)

        st.subheader("🤖 Gemini AI Traffic Assistant")

        with st.spinner("Generating AI explanation..."):
            try:
                ai_response = traffic_ai_assistant(
                    vehicle_count=vehicle_count,
                    avg_speed=avg_speed,
                    time_of_day=time_of_day,
                    day=day_map[day_of_week],
                    congestion=predicted_label,
                    risk_score=risk_score,
                    signal_time=signal_time,
                    emergency=emergency_vehicle,
                    route=best_route
                )

                st.success(ai_response)

            except Exception as e:
                st.error(f"Gemini Error: {e}")

# =========================================================
# PAGE 3 - AI INSIGHTS
# =========================================================
elif page == "📊 AI Insights":
    st.markdown('<div class="section-title">📊 AI Traffic Insights</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Peak Traffic Hour", f"{peak_hour}:00")
    col2.metric("Busiest Day", day_map[busiest_day])
    col3.metric("Slowest Day", day_map[slowest_day])

    st.write("")

    st.subheader("🔍 Model Comparison")
    compare_df = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest"],
        "Accuracy (%)": [round(dt_acc * 100, 2), round(rf_acc * 100, 2)]
    })
    st.dataframe(compare_df, use_container_width=True)

    st.subheader("📈 Congestion Trend by Time")
    congestion_hour = df.groupby("time_of_day")["vehicle_count"].mean()
    st.line_chart(congestion_hour)

    st.subheader("📌 Prediction Sample (Actual vs Predicted)")
    results_df = X_test.copy()
    results_df["Actual"] = y_test.map(reverse_label_map)
    results_df["Predicted"] = pd.Series(best_model.predict(X_test), index=y_test.index).map(reverse_label_map)
    st.dataframe(results_df.head(15), use_container_width=True)

    st.subheader("⭐ Feature Importance")
    importance = best_model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(feature_df, use_container_width=True)

    st.subheader("🧠 AI Conclusion")
    top_feature = feature_df.iloc[0]["Feature"]
    st.info(
        f"The best performing model is **{best_model_name}** with accuracy **{best_accuracy*100:.2f}%**. "
        f"The most influential feature in congestion prediction appears to be **{top_feature}**. "
        f"Peak congestion is observed around **{peak_hour}:00**, with **{day_map[busiest_day]}** showing the heaviest traffic pattern."
    )

# =========================================================
# PAGE 4 - DATASET PREVIEW
# =========================================================
elif page == "📁 Dataset Preview":
    st.markdown('<div class="section-title">📁 Dataset Preview</div>', unsafe_allow_html=True)

    st.subheader("Top 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Congestion Counts")
    congestion_df = df["congestion_level"].value_counts().reset_index()
    congestion_df.columns = ["Congestion Level", "Count"]
    st.dataframe(congestion_df, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# =========================================================
# PAGE 5 - PREDICTION HISTORY
# =========================================================
elif page == "📜 Prediction History":
    st.markdown('<div class="section-title">📜 Prediction History & Report</div>', unsafe_allow_html=True)

    if len(st.session_state.prediction_history) == 0:
        st.warning("No predictions made yet. Go to Live Prediction and test the model.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)

        st.subheader("📋 Stored Prediction Records")
        st.dataframe(history_df, use_container_width=True)

        colx, coly, colz = st.columns(3)
        colx.metric("Total Predictions", len(history_df))
        coly.metric("High Risk Cases", len(history_df[history_df["Risk Score"] >= 75]))
        colz.metric("Emergency Cases", len(history_df[history_df["Emergency"] != "None"]))

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Prediction Report CSV",
            data=csv,
            file_name="traffic_prediction_report.csv",
            mime="text/csv"
        )

        if st.button("🗑 Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()

# =========================================================
# PAGE 6 - ADMIN DASHBOARD
# =========================================================
elif page == "🛠 Admin Dashboard":
    st.markdown('<div class="section-title">🛠 Admin Dashboard</div>', unsafe_allow_html=True)

    summary = get_admin_summary(st.session_state.prediction_history)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", summary["total_predictions"])
    c2.metric("High Risk Predictions", summary["high_risk_count"])
    c3.metric("Emergency Cases", summary["emergency_cases"])

    c4, c5, c6 = st.columns(3)
    c4.metric("Average Risk Score", summary["avg_risk"])
    c5.metric("Most Common Congestion", summary["common_congestion"])
    c6.metric("Most Suggested Route", summary["common_route"])

    st.write("")
    st.markdown(f"""
    <div class="admin-card">
        <h4>📌 Admin Summary</h4>
        <p><b>Most Used Emergency Type:</b> {summary["common_emergency"]}</p>
        <p><b>Average Risk Score:</b> {summary["avg_risk"]}</p>
        <p><b>Most Frequent Congestion Level:</b> {summary["common_congestion"]}</p>
        <p><b>Most Suggested Route:</b> {summary["common_route"]}</p>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.prediction_history) == 0:
        st.info("No prediction history available yet. Make some live predictions to populate admin analytics.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)

        st.subheader("📊 Admin Traffic Distribution")
        congestion_counts = history_df["Predicted Congestion"].value_counts()
        st.bar_chart(congestion_counts)

        st.subheader("🚑 Emergency Usage Summary")
        emergency_counts = history_df["Emergency"].value_counts()
        st.bar_chart(emergency_counts)

        st.subheader("🛣 Best Route Frequency")
        route_counts = history_df["Best Route"].value_counts()
        st.bar_chart(route_counts)

# =========================================================
# PAGE 7 - PROJECT INFO
# =========================================================
elif page == "ℹ️ Project Info":
    st.markdown('<div class="section-title">ℹ️ Project Information</div>', unsafe_allow_html=True)

    st.subheader("🎯 Project Objective")
    st.write("""
    The goal of this project is to build an **AI-powered traffic management system**
    that predicts traffic congestion, calculates traffic risk, recommends smart signal timing,
    supports emergency vehicle priority, and suggests better traffic routing decisions.
    """)

    st.subheader("🧠 Technologies Used")
    st.markdown("""
    - **Python**
    - **Pandas / NumPy**
    - **Scikit-learn**
    - **Streamlit**
    - **Decision Tree & Random Forest**
    - **Machine Learning-based Congestion Prediction**
    - **Risk Analysis + Emergency Priority Logic**
    - **Route Recommendation Simulation**
    """)

    st.subheader("🚀 Future Scope")
    st.markdown("""
    - Real-time CCTV traffic monitoring
    - Google Maps / GPS live traffic integration
    - Emergency vehicle auto-priority routing
    - Smart IoT-based signal automation
    - Deep learning-based vehicle detection
    - Cloud deployment with real-time city traffic dashboard
    """)

    st.subheader("👨‍💻 Developed By")
    st.write("Vidit Verma")