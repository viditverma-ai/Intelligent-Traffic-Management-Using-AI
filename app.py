import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
        return f"🚑 Emergency priority detected for **{emergency_vehicle}**. Immediately extend green signal time, clear junction path, and prioritize fast lane movement."

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
            and suggests smarter alternate routes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
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
        "ℹ️ Project Info"
    ]
)

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
            "day_of_week": [day_of_week]
        })

        prediction = best_model.predict(input_df)[0]
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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Congestion", predicted_label)
        c2.metric("Suggested Signal Time", f"{signal_time} sec")
        c3.metric("Traffic Risk Score", f"{risk_score}/100")
        c4.metric("Severity", severity)

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
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Prediction Report CSV",
            data=csv,
            file_name="traffic_prediction_report.csv",
            mime="text/csv"
        )

# =========================================================
# PAGE 6 - PROJECT INFO
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