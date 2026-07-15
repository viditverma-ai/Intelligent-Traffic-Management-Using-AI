from llm_helper import traffic_ai_assistant

response = traffic_ai_assistant(
    vehicle_count=120,
    avg_speed=25,
    time_of_day=18,
    day="Thursday",
    congestion="High",
    risk_score=82,
    signal_time=90,
    emergency="Ambulance",
    route="Route A"
)

print(response)