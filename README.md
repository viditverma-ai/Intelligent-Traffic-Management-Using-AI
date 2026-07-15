# 🚦 Intelligent Traffic Management Using AI

An AI-powered traffic management system built using **Python, Machine Learning, and Streamlit** that predicts traffic congestion, calculates traffic risk, recommends optimized signal timings, supports emergency vehicle priority, and suggests the best alternate route based on simulated traffic conditions.

---

## 📌 Project Overview

Urban traffic congestion is a major challenge in modern cities. This project presents an **Intelligent Traffic Management System** that uses machine learning and smart rule-based logic to analyze live traffic conditions and assist in better traffic control decisions.

The system can:
- Predict **traffic congestion level** (Low / Medium / High)
- Calculate **traffic risk score**
- Recommend **signal timing**
- Support **emergency vehicle priority**
- Simulate **alternate routes** and suggest the best route
- Store **prediction history**
- Provide **admin dashboard analytics**

This project is designed as an interactive **web dashboard** using **Streamlit**.

---

# ✨ Key Features

## 1) AI Congestion Prediction
The system predicts traffic congestion using machine learning models based on:
- Vehicle count
- Average speed
- Time of day
- Day of week

Predicted output:
- **Low Congestion**
- **Medium Congestion**
- **High Congestion**

---

## 2) Traffic Risk Analysis
The system calculates a **Traffic Risk Score** from **0 to 100** using live traffic parameters.

Risk categories:
- **Low Risk**
- **Medium Risk**
- **Critical Risk**

This helps traffic authorities understand how risky the current traffic situation is.

---

## 3) Smart Signal Time Recommendation
Based on predicted congestion level and traffic conditions, the system recommends **optimized traffic signal timing** to improve vehicle movement.

Example:
- High congestion → larger green signal time
- Medium congestion → moderate signal time
- Low congestion → lower signal time

---

## 4) Emergency Vehicle Priority
The system supports priority handling for:
- **Ambulance**
- **Fire Brigade**
- **Police**

If an emergency vehicle is selected, the system increases signal time and gives priority recommendations to clear the route faster.

---

## 5) Route Recommendation Simulation
The project simulates **3 alternate routes**:
- Route A
- Route B
- Route C

Each route is analyzed using:
- Vehicle count
- Average speed
- Congestion status
- Risk score

Then the system recommends the **best route** based on lower congestion and lower risk.

---

## 6) Prediction History
Every live prediction is stored in session history with details such as:
- Vehicle count
- Average speed
- Time of day
- Day
- Emergency type
- Predicted congestion
- Suggested signal time
- Risk score
- Risk category
- Severity
- Best route

The user can also:
- Download prediction report in **CSV format**
- Clear stored prediction history

---

## 7) Admin Dashboard
The Admin Dashboard summarizes prediction activity and provides quick analytics such as:
- Total predictions
- Average risk score
- Most common congestion level
- Most suggested route
- Most used emergency type

This gives a compact overview of traffic prediction behavior.

---

## 8) Dashboard Analytics & Insights
The dashboard provides data visualization and model insights including:
- Congestion level distribution
- Average vehicle count by day
- Traffic trend by time of day
- Speed trend by day
- Peak traffic hour
- Busiest day
- Slowest day
- Model comparison
- Feature importance

---

# 🧠 Machine Learning Models Used

Two machine learning models are used in this project:

## 1. Decision Tree Classifier
Used for traffic congestion classification based on traffic input features.

## 2. Random Forest Classifier
Used as the primary model when it performs better than Decision Tree.

The project automatically compares both models and selects the **best performing model** based on accuracy.

---

# 📂 Project Modules / Pages

The Streamlit application contains the following pages:

## 1. Landing Page
A project introduction page showing:
- Project title
- Project description
- Key features
- Button to enter dashboard

## 2. Dashboard
Shows:
- Total records
- Best model
- Model accuracy
- Peak traffic hour
- Busiest day
- Average vehicle count
- Traffic charts and summary

## 3. Live Prediction
Allows user to enter live traffic values and get:
- Congestion prediction
- Signal timing recommendation
- Risk score
- Severity
- AI recommendation
- Emergency priority response
- Route recommendation

## 4. AI Insights
Shows:
- Model comparison
- Congestion trend
- Prediction sample
- Feature importance
- AI summary / conclusion

## 5. Dataset Preview
Displays:
- Top rows of dataset
- Dataset shape
- Congestion counts
- Statistical summary

## 6. Prediction History
Stores all prediction records and allows CSV download.

## 7. Admin Dashboard
Shows summary metrics and admin-level traffic analysis from stored predictions.

## 8. Project Info
Displays:
- Objective
- Technologies used
- Future scope
- Developer information

---

# 🛠 Tech Stack

## Programming Language
- Python

## Libraries / Frameworks
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## Machine Learning
- Decision Tree Classifier
- Random Forest Classifier

## Data Handling
- CSV dataset
- Session-based prediction history

---

# 📊 Input Features Used

The machine learning model uses the following features:

- **vehicle_count** → number of vehicles present
- **avg_speed** → average vehicle speed
- **time_of_day** → hour of the day
- **day_of_week** → day number (1–7)

---

# 🎯 Output Generated by the System

For each live prediction, the system generates:

- **Predicted Congestion**
- **Suggested Signal Time**
- **Traffic Risk Score**
- **Risk Category**
- **Severity**
- **AI Recommendation**
- **Emergency Priority Response**
- **Best Route Suggestion**

---

# ⚙️ How the System Works

## Step 1: Load Dataset
The traffic dataset is loaded from `traffic_data.csv`.

## Step 2: Data Cleaning
The system cleans the congestion labels and removes missing values.

## Step 3: Label Encoding
Congestion levels are converted into numeric labels:
- Low → 0
- Medium → 1
- High → 2

## Step 4: Model Training
The dataset is split into training and testing sets.  
Then:
- Decision Tree is trained
- Random Forest is trained

## Step 5: Best Model Selection
The model with better accuracy is selected as the prediction model.

## Step 6: Live Prediction
When the user enters traffic inputs, the system:
- Predicts congestion
- Calculates risk score
- Suggests signal timing
- Applies emergency priority if needed
- Simulates routes and recommends the best route
- Saves the result in prediction history

---

# 📈 Risk Score Logic

The project calculates traffic risk score using a custom formula based on:
- Vehicle count
- Average speed

General idea:
- Higher vehicle count increases risk
- Lower speed increases risk

The score is normalized to a range of **0–100**.

---

# 🚨 Emergency Priority Logic

If an emergency vehicle is selected, the system boosts the recommended signal timing:

- **Ambulance** → highest priority
- **Fire Brigade** → high priority
- **Police** → priority support

This simulates real-world smart signal handling for emergency movement.

---

# 🛣 Route Recommendation Logic

The system simulates multiple route conditions and compares them using:
- Vehicle count
- Average speed
- Congestion level
- Risk score

The route with lower risk and smoother movement is recommended as the **best route**.

---

# 📁 Project Structure

```bash
Intelligent-Traffic-Management-Using-AI/
│
├── app.py
├── traffic_data.csv
├── README.md
└── requirements.txt