# === tools/tools_weekly_digest.py ===

from datetime import datetime, timedelta
from dateutil import parser
from typing import Dict
import statistics



def generate_weekly_digest(user_id: str, user_health_data: Dict[str, list]):
    if user_id not in user_health_data or not user_health_data[user_id]:
        return {"error": "No vitals data found for this user."}

    now = datetime.now()
    week_ago = now - timedelta(days=7)

    records = [
        entry for entry in user_health_data[user_id]
        if parser.parse(entry.get("timestamp", "")) >= week_ago
    ]

    if not records:
        return {"error": "No recent vitals in the last 7 days."}

    summary = {}
    for key in records[0]:
        if key == "timestamp":
            continue
        try:
            values = [entry[key] for entry in records if isinstance(entry.get(key), (int, float))]
            if values:
                trend = "increasing: 📈" if values[-1] > values[0] else "decreasing: 📉" if values[-1] < values[0] else "stable: ➖"
                summary[key] = {
                    "average": round(statistics.mean(values), 2),
                    "trend": trend
                }
        except:
            continue

    # Recommendations based on summary
    recommendations = []
    for metric, info in summary.items():
        avg = info["average"]
        trend = info["trend"]

        if metric == "Glucose":
            if avg > 100:
                recommendations.append("⚠️ High average glucose detected. Monitor sugar intake and consult a doctor.")
            elif avg < 70:
                recommendations.append("⚠️ Low average glucose. Ensure adequate nutrition.")
        elif metric == "SpO2" and avg < 95:
            recommendations.append("⚠️ Low oxygen levels. Consider respiratory checkups.")
        elif metric == "Temperature" and avg > 37.5:
            recommendations.append("🌡️ Slight fever trend. Stay hydrated and monitor symptoms.")
        elif metric == "Weight (BMI)" and avg > 25:
            recommendations.append("📉 BMI suggests overweight. Consider dietary and fitness improvements.")
        elif metric == "Waist Circumference" and avg > 90:
            recommendations.append("📏 High waist circumference. Abdominal fat risk – exercise more.")
        elif metric in ["Hepatitis B", "Hepatitis C", "Malaria"]:
            continue  # Do not compute numeric summary, just alert on last record

    # Check last infection screening
    last_record = records[-1]
    for infection in ["Hepatitis B", "Hepatitis C", "Malaria"]:
        if infection in last_record:
            value = last_record[infection]
            if value == "Positive":
                recommendations.append(f"🚨 {infection} test is positive. Please consult a healthcare provider.")
            elif value == "Negative":
                recommendations.append(f"✅ {infection} test is negative. No signs of infection.")

    return {
        "summary_period": f"{week_ago.date()} to {now.date()}",
        "weekly_summary": summary,
        "recommendations": recommendations,
        "data_points": len(records)
    }
