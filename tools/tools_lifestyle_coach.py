from datetime import datetime, timedelta
from typing import List, Dict
import statistics
from dateutil import parser

# Global habit history (ideally should be in DB, but kept in-memory for now)
user_health_data: Dict[str, list] = {}

def record_habits(user_id: str, habits: Dict[str, float]):
    entry = {"timestamp": datetime.now().isoformat()}
    entry.update(habits)
    user_health_data.setdefault(user_id, []).append(entry)

def compute_weekly_habit_summary(user_id: str):
    if user_id not in user_health_data or not user_health_data[user_id]:
        return {"summary": "No data available for this user."}

    now = datetime.now()
    week_ago = now - timedelta(days=7)

    records = [
        entry for entry in user_health_data[user_id]
        if parser.parse(entry["timestamp"]) >= week_ago
    ]

    if not records:
        return {"summary": "No recent data available in the last 7 days."}

    summary = {}
    for key in records[0]:
        if key == "timestamp":
            continue
        try:
            values = [entry[key] for entry in records if isinstance(entry.get(key), (int, float))]
            if values:
                trend = (
                    "increasing: 📈 You're improving this habit!" if values[-1] > values[0]
                    else "decreasing: 📉 Consistency has dropped recently." if values[-1] < values[0]
                    else "stable: ➖ Your habit has been stable."
                )
                summary[key] = {
                    "avg": round(statistics.mean(values), 2),
                    "trend": trend
                }
        except:
            continue

    return {
        "summary_period": f"{week_ago.date()} to {now.date()}",
        "habit_summary": summary,
        "data_points": len(records)
    }

def generate_lifestyle_recommendations(summary_data: dict) -> List[str]:
    tips = []
    habits = summary_data.get("habit_summary", {})

    for habit, values in habits.items():
        avg = values.get("average", 0)
        trend = values.get("trend", "")

        # Use full-text trend explanations
        trend_text = {
            "increasing": "📈 You're improving this habit!",
            "decreasing": "📉 Consistency has dropped recently.",
            "stable": "➖ Your habit has been stable."
        }.get(trend, "")

        # Generate human-friendly recommendations
        if habit == "water":
            if avg < 5:
                tips.append(f"💧 Your average water intake is {avg} cups. Try to reach 8 cups daily. {trend_text}")
            else:
                tips.append(f"✅ You're staying hydrated with {avg} cups/day. Great work! {trend_text}")
        elif habit == "rest":
            if avg < 6:
                tips.append(f"🛌 You're sleeping {avg} hrs/night. Aim for 7–9 hours for full recovery. {trend_text}")
        elif habit == "screen_time":
            if avg > 6:
                tips.append(f"📱 Screen time is high at {avg} hrs/day. Take hourly breaks to reduce eye strain. {trend_text}")
        elif habit == "exercise":
            if avg < 3:
                tips.append(f"🏋️‍♂️ You're exercising {avg}x/week. Target 3–4 sessions to boost fitness. {trend_text}")
        elif habit == "meditation":
            if avg < 1:
                tips.append(f"🧘 Try meditating daily to improve mindfulness and calm. {trend_text}")
        elif habit == "fruit":
            if avg < 1:
                tips.append(f"🍎 You're averaging {avg} fruit servings/day. Try reaching 2 daily. {trend_text}")
        elif habit == "vegetable":
            if avg < 1:
                tips.append(f"🥦 Veggies are important! Add 2–3 servings per day to your meals. {trend_text}")
        elif habit == "smoking":
            if avg > 0:
                tips.append(f"🚭 You're averaging {avg} cigarettes/day. Reducing or quitting will improve your health. {trend_text}")
        elif habit == "alcohol":
            if avg > 2:
                tips.append(f"🍷 Your alcohol intake is a bit high at {avg}/week. Consider cutting down. {trend_text}")

    return tips if tips else ["✅ You're doing great! Keep up the healthy habits."]
