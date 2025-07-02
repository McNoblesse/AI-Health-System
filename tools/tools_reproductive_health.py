import json
import pandas as pd
import numpy as np
import datetime
import random
#from datetime import datetime
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

user_health_data: Dict[str, list] = {}

CYCLE_FILE = "user_data.json"
ACTIVITY_FILE = "activity_data.json"
POSTPARTUM_LOG = "postpartum_logs.json"

SYMPTOMS_LIST = [
    "Light spotting (pink or brown)", "Mild cramping",
    "Moderate cramps or back pain with bleeding", "Heavy bleeding with clots + strong cramps",
    "Sharp, stabbing pain on one side + dizziness", "Painless, bright red bleeding",
    "Severe, constant abdominal pain + bleeding", "Decreased fetal movements",
    "Bloody show (mucus mixed with blood)", "Severe headaches + swelling or vision changes"
]

# Postpartum Anomaly Detection
POSTPARTUM_FLAG_MAP = {
    "low_mood_sleep": "🧠 Emotional Health: Mood swings + low sleep raise concern for postpartum depression.",
    "infection_risk": "🩹 Wound Alert: Redness or discharge around the incision can signal infection.",
    "low_feeding_frequency": "🍼 Feeding Alert: Feeding only a few times/day is below normal.",
    "no_urine": "🚨 Hydration Risk: Baby hasn't urinated. May suggest dehydration."
}

def load_json(file):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

def add_cycle_data(user, payload):
    data = load_json(CYCLE_FILE)
    cycles = data.setdefault(user, {}).setdefault("cycle_data", [])
    new_entry = {
        "start_date": payload["start_date"],
        "period_duration": payload["period_duration"],
        #"luteal_phase": payload["luteal_phase"],
       # "stress": payload["stress"],
       # "exercise": payload["exercise"],
       # "sleep": payload["sleep"],
        #"weight_change": payload["weight_change"]
    }
    cycles.append(new_entry)
    data[user]["cycle_data"] = sort_and_recalculate_cycles(cycles)
    save_json(CYCLE_FILE, data)
    return data[user]["cycle_data"]


def sort_and_recalculate_cycles(cycles):
    df = pd.DataFrame(cycles)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df = df.sort_values("start_date")
    cycle_lengths = [28]
    for i in range(1, len(df)):
        cycle_lengths.append((df.iloc[i]["start_date"] - df.iloc[i - 1]["start_date"]).days)
    df["cycle_length"] = cycle_lengths
    df["end_date"] = df["start_date"] + pd.to_timedelta(df["period_duration"], unit='D')
    df["start_date"] = df["start_date"].dt.strftime('%Y-%m-%d')
    df["end_date"] = df["end_date"].dt.strftime('%Y-%m-%d')
    return df.to_dict(orient="records")

def predict_next_cycle(user):
    data = load_json(CYCLE_FILE).get(user, {}).get("cycle_data", [])
    if len(data) < 3:
        return {"warning": "Not enough data for prediction"}
    df = pd.DataFrame(data)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df = df.sort_values("start_date")
    ts = df["cycle_length"].astype(float)
    train = ts[:-1]
    last = df.iloc[-1]["start_date"]

    try:
        if len(train) < 3 or train.nunique() == 1:
            next_cycle_len = round(train.mean())
            model_type = "mean"
        else:
            model = ARIMA(train, order=(1, 1, 1)).fit()
            forecast = model.forecast()
            next_cycle_len = round(forecast[0] if isinstance(forecast, (np.ndarray, list, pd.Series)) else float(forecast))
            model_type = "arima"
    except Exception:
        next_cycle_len = round(train.mean())
        model_type = "mean (fallback)"

    next_start = last + pd.Timedelta(days=next_cycle_len)
    ovulation = next_start - pd.Timedelta(days=14)
    window = f"{(ovulation - pd.Timedelta(days=2)).strftime('%Y-%m-%d')} to {(ovulation + pd.Timedelta(days=2)).strftime('%Y-%m-%d')}"

    return {
        "Predicted Cycle Length": next_cycle_len,
        "Prediction Method": model_type,
        "Next Period Start": next_start.strftime('%Y-%m-%d'),
        "Ovulation Window": window
    }


def calculate_gestational_age(lmp_date):
    today = datetime.date.today()
    delta = (today - lmp_date).days
    return delta // 7, delta % 7

def predict_diagnosis(symptoms, weeks):
    trimester = "First" if weeks <= 12 else "Second" if weeks <= 27 else "Third"
    diagnosis = []
    if trimester == "First":
        if "Light spotting (pink or brown)" in symptoms:
            diagnosis.append("Implantation Bleeding (🟢 Normal)")
        if "Moderate cramps or back pain with bleeding" in symptoms:
            diagnosis.append("Possible Threatened Miscarriage (🟠 Caution)")
        if "Heavy bleeding with clots + strong cramps" in symptoms:
            diagnosis.append("Miscarriage Risk (🔴 Alert)")
    if trimester == "Second":
        if "Painless, bright red bleeding" in symptoms:
            diagnosis.append("Placenta Previa (🔴 High Risk)")
    if trimester == "Third":
        if "Severe headaches + swelling or vision changes" in symptoms:
            diagnosis.append("Possible Preeclampsia (🔴 Critical)")
    return diagnosis or ["No critical symptoms detected"]

def expected_delivery(lmp_date):
    return {
        "Start": (lmp_date + datetime.timedelta(weeks=37)).strftime('%Y-%m-%d'),
        "End": (lmp_date + datetime.timedelta(weeks=42)).strftime('%Y-%m-%d')
    }

def detect_anomalies(mother_info: Dict[str, Any], baby_info: Dict[str, Any]) -> List[str]:
    anomalies = []

    # Mood and sleep
    if mother_info.get("mood") in ["sad", "anxious"] and mother_info.get("sleep_hours", 0) < 4:
        anomalies.append("🧠 Signs of postpartum depression: Low mood + poor sleep")

    # Pain
    if mother_info.get("pain_level", 0) >= 6:
        anomalies.append("💥 High pain level after delivery")

    # Wound and fever
    wound_notes = mother_info.get("wound_notes", "").lower()
    if "redness" in wound_notes or "discharge" in wound_notes:
        anomalies.append("🩹 Wound shows redness or discharge — possible infection")

    wound_data = mother_info.get("wound_data", {})
    if wound_data.get("fever_present", "").lower() == "yes":
        anomalies.append("🌡️ Fever present — infection risk")
    if wound_data.get("post_op_medication", "").lower() == "no":
        anomalies.append("💊 Missed post-op medication")

    # Mood log consistency
    mood_log = mother_info.get("mood_log", [])
    if mood_log.count("sad") + mood_log.count("anxious") >= 3:
        anomalies.append("🧠 Consistent low mood — monitor for PPD")

    # Baby feeding and hydration
    if baby_info.get("feeding_frequency", 0) < 6:
        anomalies.append("🍼 Baby feeding frequency low")
    if not baby_info.get("urinates", True):
        anomalies.append("🚨 Baby has not urinated — check hydration")

    return anomalies


def render_flags(flag_keys):
    return [POSTPARTUM_FLAG_MAP.get(flag, f"Unknown flag: {flag}") for flag in flag_keys]

def track_postpartum_cycle(breastfeeding_months):
    return f"🕒 Ovulation may delay by approx. {breastfeeding_months * 0.5:.1f} months"

# Personalized Recommendations for Reproductive Health Tracker
def get_cycle_recommendations(latest_cycle: Dict[str, Any], prediction: Dict[str, Any], user_id: str) -> List[str]:
    recs = []

    ovulation_window = prediction.get("Ovulation Window", "")
    next_period = prediction.get("Next Period Start", "")

    if ovulation_window and "to" in ovulation_window:
        fertile_start, fertile_end = ovulation_window.split(" to ")
        fertile_start_dt = datetime.datetime.strptime(fertile_start, "%Y-%m-%d").date()
        fertile_end_dt = datetime.datetime.strptime(fertile_end, "%Y-%m-%d").date()
        today = datetime.date.today()

        recs.append(f"🩸 Your next period is predicted on **{next_period}**. Log PMS symptoms like bloating or irritability 3–7 days before.")
        recs.append(f"🧬 Ovulation is expected between **{ovulation_window}** — this is when your chances of pregnancy are highest.")
        recs.append("💡 Tip: Avoid unprotected sex 5 days before ovulation and 1 day after if not planning pregnancy.")

        if today < fertile_start_dt:
            recs.append(f"📅 Fertile window starts in {(fertile_start_dt - today).days} days.")
        elif fertile_start_dt <= today <= fertile_end_dt:
            recs.append("🔔 You are currently in your fertile window!")
        else:
            recs.append("✅ Fertile window has passed. Consider logging any symptoms and stay aware of next cycle.")

    return recs

def get_lifestyle_feedback(payload: Dict[str, Any]) -> List[str]:
    recs = []

    stress = payload.get("stress", "").lower()
    exercise = payload.get("exercise", "").lower()
    sleep = payload.get("sleep", "").lower()
    sleep_hours = payload.get("sleep_hours", 0)
    weight_change = payload.get("weight_change", "").lower()
    weight_amount = payload.get("weight_amount", 0.0)
    water = payload.get("water_intake_liters", 0)
    symptoms = payload.get("symptoms", [])
    sex_type = payload.get("sex_type", "")
    custom_note = payload.get("custom_note", "")

    # Stress
    if stress == "high":
        recs.append("🧘 Stress is high — practice deep breathing, journaling, or light walks.")
    elif stress == "moderate":
        recs.append("🌿 Moderate stress — maintain healthy boundaries and take short breaks.")
    else:
        recs.append("😌 Low stress — excellent! Keep up whatever you're doing.")

    # Exercise
    if exercise == "none":
        recs.append("🏃‍♀️ No exercise logged — light walking or yoga helps hormonal balance.")
    elif exercise == "light":
        recs.append("💪 Light exercise supports circulation and reduces cramps — keep going.")
    elif exercise == "moderate":
        recs.append("🔥 Moderate activity is great — just ensure you're staying hydrated.")
    elif exercise == "intense":
        recs.append("⚠️ Intense workouts may affect periods — balance with proper rest and meals.")

    # Sleep
    if sleep == "poor" or sleep_hours < 5:
        recs.append("🛌 Poor sleep impacts hormone regulation. Try to sleep at least 6–8 hours.")
    elif sleep == "moderate":
        recs.append("🌙 Sleep is average — aim for a consistent bedtime and screen-free evenings.")
    else:
        recs.append("✅ Excellent sleep habits! Hormones thank you.")

    # Water Intake
    if water < 1.5:
        recs.append("💧 Your water intake is low. Aim for at least 2–3 liters per day.")
    else:
        recs.append("✅ Good hydration! Water helps reduce bloating and improve mood.")

    # Weight Change
    if weight_change == "gained":
        recs.append(f"⚖️ You've gained {weight_amount}kg — avoid processed food and walk daily.")
    elif weight_change == "lost":
        recs.append(f"📉 You've lost {weight_amount}kg — ensure you're eating balanced meals.")
    else:
        recs.append("🍎 Your weight is stable — maintain nutritious choices.")

    # Symptoms
    if symptoms:
        for s in symptoms:
            name = s.get("name", "")
            severity = s.get("severity", 1)
            if name.lower() == "cramps":
                if severity >= 4:
                    recs.append("💥 Severe cramps noted. Use heat pads and track for patterns.")
                else:
                    recs.append("🌼 Mild cramps — gentle stretching and hydration may help.")
            elif name.lower() == "fatigue":
                recs.append("😴 Fatigue — eat iron-rich foods and monitor your energy levels.")
            elif name.lower() == "bloating":
                recs.append("💨 Bloating — reduce salty snacks, drink more water, and walk lightly.")

    # Sexual Activity
    if sex_type.lower() == "unprotected":
        recs.append("🔍 You logged unprotected sex — consider ovulation status or emergency contraception if needed.")
    elif sex_type.lower() == "protected":
        recs.append("🛡️ Protected sex logged — great job staying safe!")

    # Custom Note
    if custom_note:
        recs.append(f"📝 Note logged: “{custom_note}”. This will help personalize your insights.")

    return recs


def get_pregnancy_recommendations(payload: Dict[str, Any], diagnosis: List[str], edd: Dict[str, Any], user_id: str) -> List[str]:
    recs = []
    lmp_date = datetime.datetime.strptime(payload["lmp_date"], "%Y-%m-%d").date()
    today = datetime.date.today()
    gestational_weeks = (today - lmp_date).days // 7

    trimester = (
        "First" if gestational_weeks <= 12 else
        "Second" if gestational_weeks <= 27 else
        "Third"
    )

    recs.append(f"🤰 You are currently {gestational_weeks} weeks pregnant and in your **{trimester} trimester**.")
    recs.append(f"📅 Your estimated delivery window is from **{edd['Start']}** to **{edd['End']}**.")

    # Diagnosis insights
    if diagnosis:
        recs.append("🔎 Based on your symptoms:")
        for d in diagnosis:
            recs.append(f"• {d}")

    # Symptom-specific advice
    symptoms = payload.get("symptoms", [])
    if "Painless, bright red bleeding" in symptoms:
        recs.append("⚠️ Bright red bleeding may suggest placenta previa. Avoid heavy lifting and consult your doctor promptly.")
    if "Severe headaches + swelling or vision changes" in symptoms:
        recs.append("🚨 These symptoms may indicate preeclampsia. Seek immediate medical attention.")

    # Lifestyle support
    recs.append("🥗 Nutrition: Prioritize leafy greens, proteins, iron, and folate. Consider prenatal vitamins.")
    recs.append("💧 Hydration: Aim for at least 2.5L of water daily to support fetal development.")
    recs.append("🧘 Gentle movement: Prenatal yoga, walking, and stretching help circulation and reduce stress.")
    recs.append("📋 Tip: Start preparing for your hospital bag and consider birth plan discussions by week 28–30.")

    return recs

# Postpartum recommendations

def get_postpartum_recommendations(
    days_since: int,
    anomalies: List[str],
    baby_info: Dict[str, Any],
    mother_info: Dict[str, Any],
    feeding_style: str,
    delivery_type: str
) -> List[str]:
    recs = []

    # --- Timeline Awareness ---
    recs.append(f"🗓️ It's been {days_since} days since delivery — you're in the early postpartum phase where healing and adjustment are ongoing.")

    # --- Delivery Type Specific Care ---
    if delivery_type.lower() == "cesarean":
        recs.append("⚠️ Cesarean Recovery: Avoid heavy lifting, monitor your incision for redness or pus, and try to rest with your feet elevated.")

    # --- Anomaly Flags ---
    for flag in anomalies:
        flag_lower = flag.lower()

        if "depression" in flag_lower or ("mood" in flag_lower and "sleep" in flag_lower):
            recs.append("🧠 Emotional Health: Mood swings + low sleep raise concern for postpartum depression. Please consult a mental health provider or OB-GYN.")

        if "infection" in flag_lower or "redness" in flag_lower or "discharge" in flag_lower:
            recs.append("🩹 Wound Alert: Redness or discharge around the incision can signal infection. Please get your wound checked immediately.")

        if "feeding" in flag_lower and ("low" in flag_lower or "below" in flag_lower):
            recs.append("🍼 Feeding Alert: Feeding only a few times/day is below normal. Offer breast or bottle every 2–3 hours.")

        if "not urinated" in flag_lower or "hydration" in flag_lower:
            recs.append("🚨 Hydration Risk: Baby hasn't urinated. This may suggest dehydration — seek pediatric care immediately.")

    # --- Baby Observations ---
    if baby_info.get("sleep_hours", 0) < 10:
        recs.append("😴 Baby Sleep: 9 hours is low. Try calming night routines, swaddling, and soft lullabies.")
    if "fussy" in baby_info.get("expression_notes", "").lower():
        recs.append("👶 Baby Behavior: Fussiness may indicate gas, hunger, overstimulation, or need for comfort.")
    if "latch" in baby_info.get("breastfeeding_notes", "").lower():
        recs.append("🤱 Breastfeeding: Latching issues are common. Try different positions and consider seeing a lactation consultant.")
    if baby_info.get("stool_color", "").lower() not in ["yellow", "brown"]:
        recs.append(f"💩 Stool Alert: Green stools can indicate imbalance in foremilk/hindmilk or mild intolerance. Monitor for changes.")

    # --- Mother's Physical Recovery ---
    if mother_info.get("pain_level", 0) >= 6:
        recs.append("💊 Pain Management: Your pain score is high. Contact your doctor — unmanaged pain hinders healing.")
    if mother_info.get("wound_data", {}).get("post_op_medication", "Yes").lower() == "no":
        recs.append("💊 Missed Meds: You’ve not taken your post-op meds today. This may delay healing or increase discomfort.")
    if mother_info.get("wound_data", {}).get("fever_present", "No").lower() == "yes":
        recs.append("🌡️ Fever is present. This may signal infection — don’t ignore it.")

    # --- Emotional Wellbeing ---
    if mother_info.get("mood") in ["anxious", "sad"] or mother_info.get("emotional_state", "").lower() in ["irritable", "tearful"]:
        recs.append("🧘‍♀️ Emotional Check-In: Your mood shows signs of overwhelm. Rest, delegate, talk to a loved one — and consider professional support.")

    # --- Lifestyle & Hormonal Recovery ---
    if "Hair loss" in mother_info.get("body_changes", []):
        recs.append("🧬 Hair Loss: This is common postpartum due to hormone shifts. It usually resolves within 6–12 months.")
    if "Mood swings" in mother_info.get("body_changes", []):
        recs.append("🌪️ Mood Swings: Hormonal readjustments take time. Stay hydrated, eat protein-rich meals, and seek companionship.")

    # --- Feeding Style Tips ---
    if feeding_style.lower() == "mixed":
        recs.append("🔄 Mixed Feeding: Try to establish a rhythm. Start feeds with the breast to build supply, then supplement with formula.")
    elif feeding_style.lower() == "exclusive breastfeeding":
        recs.append("🍼 Exclusive Breastfeeding: Aim for 8–12 feeds per day to maintain milk supply.")
    elif feeding_style.lower() == "formula only":
        recs.append("🍼 Formula Feeding: Track baby’s feeding volume and look for regular wet diapers and consistent weight gain.")

    # --- Ovulation Recovery ---
    recs.append("🩺 Postpartum Ovulation: Breastfeeding can delay ovulation, but it's not 100% reliable. Use protection if avoiding pregnancy.")
    
    # --- Encouragement ---
    recs.append("💖 You're doing your best. Healing, adjusting, and caring — all at once. Take breaks, ask for help, and celebrate tiny wins.")

    return recs


# Updated diagnosis predictor

def predict_diagnosis(symptoms: List[str], gestational_weeks: int) -> List[str]:
    trimester = (
        "First Trimester" if gestational_weeks <= 12 else
        "Second Trimester" if gestational_weeks <= 27 else
        "Third Trimester"
    )

    diagnosis_map = {
        "First Trimester": {
            "Light spotting (pink or brown)": "Implantation bleeding — often normal.",
            "Mild cramping": "Uterine expansion — normal unless severe.",
            "Heavy bleeding with clots + strong cramps": "Possible miscarriage — seek emergency care.",
            "Sharp, stabbing pain on one side + dizziness": "Possible ectopic pregnancy — critical attention needed.",
        },
        "Second Trimester": {
            "Painless, bright red bleeding": "Placenta previa — avoid strain and monitor.",
            "Moderate cramps or back pain with bleeding": "Possible cervical insufficiency.",
        },
        "Third Trimester": {
            "Bloody show (mucus mixed with blood)": "Sign of early labor — stay alert.",
            "Severe, constant abdominal pain + bleeding": "Placental abruption — urgent attention.",
            "Decreased fetal movements": "Fetal distress — seek care.",
            "Severe headaches + swelling or vision changes": "Preeclampsia — monitor BP & consult OB-GYN."
        }
    }

    diagnoses = []
    rules = diagnosis_map.get(trimester, {})
    for sym in symptoms:
        diagnoses.append(f"{sym} → {rules.get(sym, f'Not typical in {trimester}, monitor and report if worsens.')}")
    return diagnoses

# Main routing agent

def run_reproductive_agent(user_id: str, mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if mode == "cycle":

        general_advice = [
                "🧼 Maintain good menstrual hygiene — change your pad or tampon every 4–6 hours to avoid irritation or infection.",
                "💧 Stay hydrated — drinking more water can help reduce bloating and cramps.",
                "📆 Track your cycle regularly to become more aware of your patterns and symptoms.",
                "🩸 Light exercise like walking or stretching may help ease period cramps.",
                "🛌 Rest when needed. Hormonal changes can make you feel tired during menstruation.",
                "🍫 Craving chocolate? It’s okay in moderation — dark chocolate may even boost your mood.",
                "📦 Always have supplies ready — pads, tampons, or menstrual cups. Keep extras in your bag just in case.",
                "📲 Consider using this app to log moods, cramps, or spotting for better awareness over time."
            ]
        cycles = add_cycle_data(user_id, payload)
        latest_cycle = cycles[-1]

        if len(cycles) < 3:
            disclaimer = (
                f"⚠️ *Disclaimer:* You've entered {len(cycles)} cycle sample(s). "
                "This tool requires at least 3 months of data for accurate period and ovulation predictions. "
                "Your input has been logged, and we’ve given lifestyle feedback below. "
                "Once 3 entries are available, predictive recommendations will be enabled. "
                "Always consult a medical expert for personal reproductive guidance."
            )
        
            # General menstrual hygiene and care tips
            general_advice = [
                "🧼 Maintain good menstrual hygiene — change your pad or tampon every 4–6 hours to avoid irritation or infection.",
                "💧 Stay hydrated — drinking more water can help reduce bloating and cramps.",
                "📆 Track your cycle regularly to become more aware of your patterns and symptoms.",
                "🩸 Light exercise like walking or stretching may help ease period cramps.",
                "🛌 Rest when needed. Hormonal changes can make you feel tired during menstruation.",
                "🍫 Craving chocolate? It’s okay in moderation — dark chocolate may even boost your mood.",
                "📦 Always have supplies ready — pads, tampons, or menstrual cups. Keep extras in your bag just in case.",
                "📲 Consider using this app to log moods, cramps, or spotting for better awareness over time."
            ]
        
            return {
                "mode": "Cycle Tracking",
                "latest_cycle": latest_cycle,
                "recommendations": random.sample(general_advice, k=5),
                "disclaimer": disclaimer,
                "entry_count": len(cycles),
                "offer_chat": True,
                "chat_prompt": "Would you like to chat with Dr. Deuce for deeper insights and support on your cycle health?"
            }


        prediction = predict_next_cycle(user_id)
        disclaimer = (
            "⚠️ *Disclaimer:* Predictions are based on your cycle data. While this tool offers data-driven insights, "
            "cycle lengths and symptoms vary. Always seek professional medical consultation when necessary."
        )

        entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": "cycle",
        "input": payload,
        "prediction": prediction if len(cycles) >= 3 else {},
        "recommendations": get_cycle_recommendations(latest_cycle, prediction if len(cycles) >= 3 else {}, user_id)
        }
        user_health_data.setdefault(user_id, []).append(entry)
        
        return {
            "mode": "Cycle Tracking",
            "latest_cycle": latest_cycle,
            "next_prediction": prediction,
            "recommendations": get_cycle_recommendations(latest_cycle, prediction, user_id),
            "recommendations+": random.sample(general_advice, k=3),
            "disclaimer": disclaimer,
            "entry_count": len(cycles),
            "offer_chat": True,
            "chat_prompt": "Would you like to chat with Dr. Deuce for deeper insights and support on your cycle health?"
        }

    elif mode == "lifestyle":
        return {
        "mode": "Lifestyle Insight",
        "recommendations": get_lifestyle_feedback(payload),
        "disclaimer": "⚠️ These lifestyle insights are based on your self-reported data and do not substitute professional medical advice.",
        "offer_chat": True,
        "chat_prompt": "Would you like to chat with Dr. Deuce about improving your stress, sleep, or weight balance?"
    }


    elif mode == "pregnancy":
        lmp = datetime.datetime.strptime(payload["lmp_date"], "%Y-%m-%d").date()
        weeks, days = calculate_gestational_age(lmp)
        diag = predict_diagnosis(payload.get("symptoms", []), weeks)
        edd = expected_delivery(lmp)
        disclaimer = (
            "⚠️ *Disclaimer:* These pregnancy-related predictions and suggestions are based on general medical data "
            "and your inputs. For any unusual symptoms or concerns, always consult a certified obstetrician or health care provider."
        )

        recommendations = get_pregnancy_recommendations(payload, diag, edd, user_id)

        entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": "pregnancy",
        "input": payload,
        "gestational_age": f"{weeks} weeks {days} days",
        "edd": edd,
        "diagnosis": diag,
        "recommendations": recommendations
        }
        user_health_data.setdefault(user_id, []).append(entry)
        
        return {
            "mode": "Pregnancy Monitoring",
            "Gestational Age": f"{weeks} weeks {days} days",
            "Expected Delivery Window": edd,
            "Diagnosis": diag,
            "recommendations": recommendations,
            "disclaimer": disclaimer,
            "offer_chat": True,
            "chat_prompt": "Would you like to chat with Dr. Deuce for more personalized pregnancy tips and support?"
        }

    elif mode == "postpartum":
        delivery = datetime.datetime.strptime(payload["delivery_date"], "%Y-%m-%d").date()
        days = (datetime.date.today() - delivery).days
    
        # ✅ Define mother_info and baby_info BEFORE referencing them
        mother_info = payload["mother"]
        baby_info = payload["baby"]
        feeding_style = payload["feeding_style"]
        delivery_type = payload["type_of_delivery"]
    
        # ✅ Now pass both values to anomaly checker
        anomalies = detect_anomalies(mother_info, baby_info)
    
        ovulation = track_postpartum_cycle(payload["breastfeeding_duration"])
        disclaimer = (
            "⚠️ *Disclaimer:* Postpartum recovery is unique to every mother. These tips are informative and not a substitute for professional care. "
            "If you feel unwell, always consult your doctor or pediatrician."
        )

    
        recommendations = get_postpartum_recommendations(
            days_since=days,
            anomalies=anomalies,
            baby_info=baby_info,
            mother_info=mother_info,
            feeding_style=feeding_style,
            delivery_type=delivery_type
        )

        entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": "postpartum",
        "input": payload,
        "days_since_delivery": days,
        "flags": anomalies,
        "recommendations": recommendations
        }
        user_health_data.setdefault(user_id, []).append(entry)
        
        return {
            "mode": "Postpartum Recovery",
            "Days Since Delivery": days,
            "Flags": anomalies,
            "Ovulation Info": ovulation,
            "recommendations": recommendations,
            "disclaimer": disclaimer,
            "chat_prompt": "Would you like to chat with Dr. Deuce for postpartum recovery advice or baby care tips?"
        }

    return {
        "error": "Invalid mode selected.",
        "offer_chat": False,
        "chat_prompt": "Unable to determine mode. Please check your input."
    }



