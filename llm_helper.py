import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Model
model = genai.GenerativeModel("gemini-flash-latest")


def traffic_ai_assistant(
    vehicle_count,
    avg_speed,
    time_of_day,
    day,
    congestion,
    risk_score,
    signal_time,
    emergency,
    route
):

    prompt = f"""
You are an AI Traffic Management Expert.

Traffic Information

Vehicle Count: {vehicle_count}

Average Speed: {avg_speed} km/h

Time: {time_of_day}:00

Day: {day}

Congestion: {congestion}

Risk Score: {risk_score}/100

Signal Time: {signal_time} seconds

Emergency Vehicle: {emergency}

Recommended Route: {route}

Please provide:

1. Explain why congestion occurred.

2. Explain whether the risk score is good or bad.

3. Explain why this signal timing is recommended.

4. Suggest traffic police actions.

5. Suggest advice for citizens.

6. Explain why the selected route is better.

Keep the answer professional, simple and under 200 words.
"""

    response = model.generate_content(prompt)

    return response.text