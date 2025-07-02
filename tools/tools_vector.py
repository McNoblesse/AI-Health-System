import requests
from langchain.tools import Tool
import json

# === Vector Search Tool ===
def call_mcp_vector_search(query: str) -> str:
    try:
        response = requests.post(
            "http://localhost:8000/agent-query",
            json={"user_id": "Dr Deuce", "query": query},
            timeout=10
        )
        data = response.json()
        return data.get("response", "No relevant information found.")
    except Exception:
        return "⚠️ Could not contact the MCP server. Please try again."

vector_search_tool = Tool(
    name="VectorSearchTool",
    func=call_mcp_vector_search,
    description="Retrieves medical or health-related info from the vector store."
)

# === Health Score Analysis Tool ===
def analyze_health_score(health_json: str) -> str:
    user_data = json.loads(health_json)["data"]
    total_score = user_data.get("Total_Health_Score", 0)
    category = user_data.get("Health_Category", "Unknown")
    weak_vitals = [
        f"- {vital.replace('_Score', '')}: Score {user_data[vital]} ⚠️"
        for vital in user_data if "_Score" in vital and user_data[vital] < 70
    ]
    tips = []
    if user_data.get("Glucose", 0) > 100:
        tips.append("🩸 Reduce sugar intake to control glucose levels.")
    if user_data.get("SpO2", 0) < 95:
        tips.append("💨 Improve oxygen intake by engaging in breathing exercises.")
    return json.dumps({
        "Total_Health_Score": total_score,
        "Health_Category": category,
        "Weak_Vitals": "\n".join(weak_vitals) if weak_vitals else "All vitals are in good condition ✅",
        "Personalized_Health_Tips": "\n".join(tips) if tips else "Keep up the good health habits! 🎉"
    }, indent=4)

health_score_analysis_tool = Tool(
    name="HealthScoreAnalysis",
    func=analyze_health_score,
    description="Analyzes a user's health score, highlights weak vitals, and provides personalized health tips."
)

# === Vital Signs Monitoring Tool ===
def monitor_vital_signs(vitals_json: str) -> str:
    user_data = json.loads(vitals_json)["data"]
    alerts = [
        f"🚨 {vital.replace('_Score', '')}: Score {user_data[vital]} (Critical deviation from normal range!)"
        for vital in user_data if "_Score" in vital and user_data[vital] < 50
    ]
    recommendations = []
    if user_data.get("Blood Pressure (Systolic)", 0) > 140:
        recommendations.append("🫀 Reduce salt intake and exercise regularly to lower blood pressure.")
    if user_data.get("SpO2", 0) < 92:
        recommendations.append("💨 Improve air quality and practice deep breathing exercises.")
    return json.dumps({
        "Vital_Sign_Alerts": "\n".join(alerts) if alerts else "✅ All vital signs are stable.",
        "Health_Recommendations": "\n".join(recommendations) if recommendations else "Keep maintaining a healthy lifestyle!"
    }, indent=4)

vital_sign_monitoring_tool = Tool(
    name="VitalSignsMonitoring",
    func=monitor_vital_signs,
    description="Monitors vital signs, detects abnormal patterns, and provides health risk alerts with recommendations."
)

# === Automated Health Consultation Tool ===
def automated_health_consultation(health_json: str) -> str:
    user_data = json.loads(health_json)["data"]
    medical_advice = []
    need_doctor_visit = False
    if user_data.get("Glucose", 0) > 130:
        medical_advice.append("🩸 High blood sugar detected. You may need to consult an **endocrinologist**.")
        need_doctor_visit = True
    if user_data.get("Blood Pressure (Systolic)", 0) > 140:
        medical_advice.append("🫀 High blood pressure detected. Please consult a **cardiologist**.")
        need_doctor_visit = True
    if user_data.get("Malaria") == "Positive":
        medical_advice.append("🦟 Malaria detected. Consult a **general physician** for treatment.")
        need_doctor_visit = True
    return json.dumps({
        "Medical_Advice": "\n".join(medical_advice) if medical_advice else "✅ No immediate health concerns detected.",
        "Doctor_Visit_Recommended": need_doctor_visit
    }, indent=4)

automated_health_consultation_tool = Tool(
    name="AutomatedHealthConsultation",
    func=automated_health_consultation,
    description="Acts as a virtual doctor, analyzing health data and providing preliminary medical advice."
)