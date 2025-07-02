# tools/tools_chronic_risk.py

from typing import Dict, Any

def predict_chronic_risk(data: Dict[str, Any]) -> Dict[str, Any]:
    risk_factors = []
    diabetes_risk = 0
    hypertension_risk = 0

    # Diabetes scoring
    glucose = data.get("glucose", 0)
    bmi = data.get("bmi", 0)
    if glucose > 125:
        diabetes_risk += 2
        risk_factors.append("🔴 Very High Glucose Level")
    elif 100 < glucose <= 125:
        diabetes_risk += 1
        risk_factors.append("🟠 Borderline Glucose Level")
    
    if bmi >= 30:
        diabetes_risk += 2
        risk_factors.append("🔴 Obese BMI")
    elif 25 <= bmi < 30:
        diabetes_risk += 1
        risk_factors.append("🟠 Overweight BMI")

    if data.get("family_history_diabetes", "").lower() == "yes":
        diabetes_risk += 1
        risk_factors.append("🧬 Family History of Diabetes")

    if data.get("physical_activity", "").lower() in ["sedentary", "low"]:
        diabetes_risk += 1
        risk_factors.append("🛋️ Low Physical Activity")

    if data.get("diet", "").lower() in ["processed", "unhealthy"]:
        diabetes_risk += 1
        risk_factors.append("🍔 Unhealthy Diet")

    # Hypertension scoring
    systolic = data.get("systolic_bp", 0)
    diastolic = data.get("diastolic_bp", 0)
    if systolic >= 140 or diastolic >= 90:
        hypertension_risk += 2
        risk_factors.append("🔴 High Blood Pressure")
    elif systolic >= 130 or diastolic >= 80:
        hypertension_risk += 1
        risk_factors.append("🟠 Elevated Blood Pressure")

    if data.get("stress", "").lower() == "high":
        hypertension_risk += 1
        risk_factors.append("😰 High Stress Level")

    if data.get("smoking", "").lower() == "yes":
        hypertension_risk += 1
        risk_factors.append("🚬 Smoking Habit")

    if data.get("alcohol", "").lower() == "yes":
        hypertension_risk += 1
        risk_factors.append("🍷 Frequent Alcohol Consumption")

    if data.get("family_history_hypertension", "").lower() == "yes":
        hypertension_risk += 1
        risk_factors.append("🧬 Family History of Hypertension")

    # Final labels
    diabetes_level = "🔴 High Risk" if diabetes_risk >= 4 else "🟠 Moderate Risk" if diabetes_risk >= 2 else "🟢 Low Risk"
    hypertension_level = "🔴 High Risk" if hypertension_risk >= 4 else "🟠 Moderate Risk" if hypertension_risk >= 2 else "🟢 Low Risk"

    return {
        "Diabetes Risk": diabetes_level,
        "Hypertension Risk": hypertension_level,
        "Risk Factors": list(set(risk_factors)),
        "Recommendations": generate_chronic_recommendations(diabetes_level, hypertension_level)
    }

def generate_chronic_recommendations(diabetes_level, hypertension_level):
    recs = []
    if diabetes_level in ["🟠 Moderate Risk", "🔴 High Risk"]:
        recs.append("🥗 Adopt a low-sugar, high-fiber diet.")
        recs.append("🚶 Increase daily physical activity (30+ mins walk).")
        recs.append("🩺 Schedule a fasting glucose or A1C test.")
    if hypertension_level in ["🟠 Moderate Risk", "🔴 High Risk"]:
        recs.append("🧂 Reduce salt and processed food intake.")
        recs.append("🧘 Practice stress reduction techniques (yoga, meditation).")
        recs.append("🩺 Monitor your blood pressure regularly.")
    if diabetes_level == hypertension_level == "🟢 Low Risk":
        recs.append("✅ Maintain your current healthy lifestyle!")
    return recs
