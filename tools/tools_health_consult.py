from langchain.tools import Tool
import json

def automated_health_consultation(health_data_json: str) -> str:
    try:
        user_data = json.loads(health_data_json)["data"]
        medical_advice = []
        need_doctor_visit = False

        if user_data.get("Glucose", 0) > 130:
            medical_advice.append("🩸 High blood sugar detected. You may need to consult an **endocrinologist**.")
            need_doctor_visit = True
        if user_data.get("Blood Pressure (Systolic)", 0) > 140:
            medical_advice.append("🫀 High blood pressure detected. Please consult a **cardiologist**.")
            need_doctor_visit = True
        if user_data.get("ECG (Heart Rate)", 0) < 50 or user_data.get("ECG (Heart Rate)", 0) > 110:
            medical_advice.append("💓 Irregular heart rate detected. Consider visiting a **cardiologist**.")
            need_doctor_visit = True
        if user_data.get("Malaria") == "Positive":
            medical_advice.append("🦟 Malaria detected. Consult a **general physician** for treatment.")
            need_doctor_visit = True
        if user_data.get("Weight (BMI)", 0) > 30:
            medical_advice.append("⚖️ Obesity risk detected. You may need to see a **nutritionist**.")
            need_doctor_visit = True

        return json.dumps({
            "Medical_Advice": "\n".join(medical_advice) if medical_advice else "✅ No immediate health concerns detected.",
            "Doctor_Visit_Recommended": need_doctor_visit
        }, indent=4)

    except Exception as e:
        return json.dumps({"error": f"Invalid input format. Error: {str(e)}"})

automated_health_consultation_tool = Tool(
    name="AutomatedHealthConsultation",
    func=automated_health_consultation,
    description="Acts as a virtual doctor, analyzing health data and providing preliminary medical advice."
)
