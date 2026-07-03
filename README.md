# 🚦 Intelligent Traffic Management Using AI

An AI-powered traffic congestion prediction and smart signal recommendation system built using **Python, Machine Learning, and Streamlit**.

This project predicts **traffic congestion levels**, calculates **traffic risk**, suggests **optimized traffic signal timings**, supports **emergency vehicle priority**, and recommends **alternate routes** for smoother urban traffic flow.

---

## 📌 Project Overview

Traffic congestion is one of the biggest urban problems. This project provides a smart AI-based solution that analyzes traffic conditions using:
- Vehicle count
- Average speed
- Time of day
- Day of week
- Emergency vehicle priority

Based on these inputs, the system predicts the traffic congestion level and suggests better traffic signal timing and route decisions.

---

## ✨ Key Features

- 🚥 **Traffic Congestion Prediction**
  - Predicts whether traffic is **Low / Medium / High**

- ⏱ **Smart Signal Time Recommendation**
  - Suggests optimized signal timing based on congestion level and traffic conditions

- ⚠ **Traffic Risk Analysis**
  - Calculates a traffic risk score
  - Categorizes traffic as Low / Medium / Critical Risk

- 🚑 **Emergency Vehicle Priority**
  - Gives special handling for Ambulance / Fire Brigade / Police / VIP movement

- 🛣 **Alternate Route Recommendation**
  - Suggests best route strategy based on congestion and emergency priority

- 📊 **AI Insights Dashboard**
  - Peak traffic hour
  - Busiest day
  - Slowest day
  - Model comparison
  - Feature importance

- 📜 **Prediction History**
  - Stores live predictions made by the user
  - Downloadable report in CSV format

- 🌐 **Interactive Web App**
  - Built using Streamlit for a modern dashboard-style interface

---

## 🧠 Machine Learning Models Used

This project uses two machine learning models for traffic congestion prediction:

- **Decision Tree Classifier**
- **Random Forest Classifier**

The best performing model is automatically selected based on accuracy.

---

## 🛠 Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Machine Learning**

---

## 📂 Project Structure

```bash
Intelligent-Traffic-Management-Using-AI/
│── app.py
│── traffic_data.csv
│── requirements.txt
│── README.md
│── screenshots/
│── output/
│── model/