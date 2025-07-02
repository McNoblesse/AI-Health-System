#tools/tools_progress_tracker.py

from datetime import datetime, timedelta
from dateutil import parser
import statistics
import logging

def generate_monthly_summary(user_id: str, user_health_data: dict[str, list]):
    if user_id not in user_health_data or not user_health_data[user_id]:
        return {"summary": "No data available for this user."}

    now = datetime.now()
    month_ago = now - timedelta(days=30)

    records = [
        entry for entry in user_health_data[user_id]
        if parser.parse(entry["timestamp"]) >= month_ago
    ]

    if not records:
        return {"summary": "No recent data available in the last 30 days."}

    summary = {}
    for key in records[0]:
        if key == "timestamp":
            continue
        try:
            values = [entry[key] for entry in records if isinstance(entry.get(key), (int, float))]
            if values:
                trend_symbol = "→"
                trend_expl = "Stable trend"
                if values[-1] > values[0]:
                    trend_symbol = "↑"
                    trend_expl = "Increasing trend"
                elif values[-1] < values[0]:
                    trend_symbol = "↓"
                    trend_expl = "Decreasing trend"
                summary[key] = {
                    "avg": round(statistics.mean(values), 2),
                    "min": min(values),
                    "max": max(values),
                    "trend": f"{trend_symbol} {trend_expl}"
                }
        except:
            continue

    return {
        "summary_period": f"{month_ago.date()} to {now.date()}",
        "trend_analysis": summary,
        "data_points": len(records)
    }

def generate_trend_recommendations(trends: dict[str, dict]) -> list[str]:
    tips = []
    for metric, info in trends.items():
        trend = info.get("trend", "")
        avg = info.get("avg")

        if "↑" in trend:
            if metric == "Glucose":
                tips.append("🍬 Glucose levels are increasing. Reduce sugar intake and monitor regularly.")
            elif metric == "Temperature":
                tips.append("🌡️ Rising temperature detected. Check for fever or infection.")
            elif metric == "Blood Pressure (Systolic)":
                tips.append("🩺 Systolic pressure is increasing. Limit sodium and manage stress.")
            elif metric == "Blood Pressure (Diastolic)":
                tips.append("💓 Diastolic pressure rising. Ensure adequate rest and hydration.")
            elif metric == "Weight (BMI)":
                tips.append("⚖️ BMI is going up. Adopt a balanced diet and exercise more.")
            elif metric == "Waist Circumference":
                tips.append("📏 Waist size growing. Watch abdominal fat and eat lean.")
            elif metric == "ECG (Heart Rate)":
                tips.append("❤️ Heart rate rising. Consider cardiovascular check-up.")
        elif "↓" in trend:
            if metric == "SpO2":
                tips.append("🫁 Oxygen saturation decreasing. Improve ventilation and seek help if persistent.")
            elif metric == "Glucose":
                tips.append("🧁 Falling glucose. Ensure stable meal routines.")
            elif metric == "Temperature":
                tips.append("🥶 Temperature dropping. Stay warm and monitor closely.")
            elif metric == "Blood Pressure (Systolic)":
                tips.append("🩸 Systolic drop observed. Check for dizziness or fatigue.")
            elif metric == "Blood Pressure (Diastolic)":
                tips.append("🫀 Diastolic dropping. Ensure hydration and balanced electrolytes.")
            elif metric == "Weight (BMI)":
                tips.append("🍽️ Weight reducing. Confirm it's intentional and healthy.")
            elif metric == "Waist Circumference":
                tips.append("✅ Waist reduction seen. Keep up healthy habits.")
            elif metric == "ECG (Heart Rate)":
                tips.append("📉 Falling heart rate. If paired with fatigue, consult a physician.")
        else:
            if metric == "Glucose":
                tips.append("📊 Glucose stable. Continue balanced meals.")
            elif metric == "Weight (BMI)":
                tips.append("📉 BMI stable. Maintain current regimen.")
            elif metric == "SpO2":
                tips.append("🫁 Oxygen levels are steady. Great work!")

    return tips if tips else ["👍 No critical trends detected. Keep up the good work!"]