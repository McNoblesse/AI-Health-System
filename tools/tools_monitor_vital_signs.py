import json
from datetime import datetime
from langchain.tools import Tool

# Centralized vital sign interpretation logic

def monitor_vital_signs(health_data_json: str, user_health_data: dict = None) -> str:
    try:
        user_data = json.loads(health_data_json).get("data", {})
        explanations = []

        # Optionally store user health data if a reference dict is passed
        if user_health_data is not None:
            user_id = json.loads(health_data_json).get("user_id", "unknown")
            record = {**user_data, "timestamp": datetime.now().isoformat()}
            user_health_data.setdefault(user_id, []).append(record)

        # Iterate over each measured parameter
        for key, value in user_data.items():
            # Handle specific vitals with explanations
            if key == "Glucose":
                if value < 70:
                    explanations.append("🚨 Glucose is too low (Hypoglycemia). Consider eating something sugary immediately.")
                elif value > 100:
                    explanations.append("⚠️ Glucose is high. This could indicate prediabetes or diabetes. Monitor your diet and consult a doctor.")
                else:
                    explanations.append("✅ Glucose is within the normal range. Keep maintaining a balanced diet.")

            elif key == "SpO2":
                if value < 92:
                    explanations.append("🚨 SpO2 is critically low. This could indicate respiratory issues. Seek medical attention immediately.")
                elif value < 95:
                    explanations.append("⚠️ SpO2 is slightly low. Consider improving air quality and practicing deep breathing exercises.")
                else:
                    explanations.append("✅ SpO2 is normal. Your oxygen saturation levels are healthy.")

            elif key == "ECG (Heart Rate)":
                if value < 60:
                    explanations.append("⚠️ Heart rate is low (Bradycardia). This could indicate an underlying condition. Consult a doctor.")
                elif value > 100:
                    explanations.append("⚠️ Heart rate is high (Tachycardia). This could be due to stress, dehydration, or other factors. Monitor closely.")
                else:
                    explanations.append("✅ Heart rate is normal. Your cardiovascular health looks good.")

            elif key == "Blood Pressure (Systolic)":
                if value > 140:
                    explanations.append("🚨 Systolic blood pressure is too high. This could indicate hypertension. Reduce salt intake and consult a doctor.")
                elif value < 90:
                    explanations.append("⚠️ Systolic blood pressure is too low. This could indicate hypotension. Stay hydrated and consult a doctor.")
                else:
                    explanations.append("✅ Systolic blood pressure is normal. Your cardiovascular health is stable.")

            elif key == "Blood Pressure (Diastolic)":
                if value > 90:
                    explanations.append("🚨 Diastolic blood pressure is too high. This could indicate hypertension. Consult a doctor.")
                elif value < 60:
                    explanations.append("⚠️ Diastolic blood pressure is too low. This could indicate hypotension. Stay hydrated and monitor your health.")
                else:
                    explanations.append("✅ Diastolic blood pressure is normal. Your cardiovascular health is stable.")

            elif key == "Temperature":
                if value < 36.0:
                    explanations.append("⚠️ Body temperature is low. This could indicate hypothermia. Stay warm and monitor your health.")
                elif value > 37.5:
                    explanations.append("⚠️ Body temperature is high. This could indicate a fever. Stay hydrated and rest.")
                else:
                    explanations.append("✅ Body temperature is normal. Your body is functioning well.")

            elif key == "Weight (BMI)":
                if value < 16:
                    explanations.append("🚨 BMI is very severely underweight (<16). This may indicate malnutrition or an eating disorder. Seek immediate medical evaluation and consider nutritional therapy.")
                elif 16 <= value < 17:
                    explanations.append("⚠️ BMI is severely underweight (16–16.9). Increased risk of immune deficiency, fertility issues, and osteoporosis. Nutritional improvement is necessary.")
                elif 17 <= value < 18.5:
                    explanations.append("⚠️ BMI is underweight (17–18.4). May indicate inadequate nutrition or other health concerns. Consider increasing caloric intake and consulting a dietitian.")
                elif 18.5 <= value <= 24.9:
                    explanations.append("✅ BMI is in the normal range (18.5–24.9). Maintain your current diet and physical activity for continued health.")
                elif 25 <= value < 30:
                    explanations.append("⚠️ BMI is overweight (25–29.9). Increased risk of cardiovascular diseases. Recommend weight control via reduced sugar, more fiber, and regular aerobic activity.")
                elif 30 <= value < 35:
                    explanations.append("🚨 BMI indicates Obesity Class I (30–34.9). Increased risk of type 2 diabetes, hypertension, and metabolic syndrome. Consider structured weight loss plans.")
                elif 35 <= value < 40:
                    explanations.append("🚨 BMI indicates Obesity Class II (35–39.9). High health risk. Medical weight management and lifestyle intervention highly advised.")
                else:
                    explanations.append("🚨 BMI indicates Obesity Class III (≥40). This is considered severe obesity. High risk of life-threatening conditions. Consult a bariatric specialist for comprehensive care.")

            elif key == "Waist Circumference":
                if value < 80:
                    explanations.append("✅ Waist circumference is in the optimal range (<80 cm). Low risk of abdominal obesity-related complications. Maintain a healthy diet and active lifestyle.")
                elif 80 <= value < 90:
                    explanations.append("⚠️ Waist circumference is borderline high (80–89 cm). There's a growing risk of developing insulin resistance, high blood pressure, and lipid imbalances. Consider reducing processed foods and increasing physical activity.")
                elif 90 <= value < 102:
                    explanations.append("🚨 Waist circumference is high (90–101 cm). This indicates abdominal obesity, which significantly increases the risk of cardiovascular disease, type 2 diabetes, and metabolic syndrome. Adopt a targeted weight management program and monitor waist size regularly.")
                else:
                    explanations.append("🚨 Waist circumference is very high (≥102 cm). This is a strong indicator of visceral fat accumulation and elevated risk of serious conditions like heart attack, stroke, and fatty liver disease. Seek clinical evaluation and implement an aggressive lifestyle intervention immediately.")

            elif key == "Hepatitis B":
                explanations.append("🧬 **Hepatitis B Serology Guide**")
                if value == "Positive":
                    explanations.append("🚨 **Your Hepatitis B test is POSITIVE.** This means you have been exposed to the Hepatitis B virus (HBV).")
                    explanations.append("🧠 Hepatitis B is a viral infection that affects the liver and can be **chronic or acute**. It's spread through blood, sexual contact, or from mother to child.")
                    explanations.append("🩺 **What to do next:** Please consult a hepatologist. You will likely need further tests (e.g., liver function, HBV DNA) to determine if the infection is active or chronic.")
                    explanations.append("📌 Avoid alcohol, get tested for Hepatitis D (which co-infects), and inform partners or close contacts.")
                    explanations.append("💉 Household contacts should be vaccinated if not already. Hepatitis B is preventable with vaccines.")
                elif value == "Negative":
                    explanations.append("✅ **Your Hepatitis B test is NEGATIVE.** You are not currently infected.")
                    explanations.append("💉 If you haven't been vaccinated, now is a good time. Hepatitis B is vaccine-preventable and protection lasts years.")
                    explanations.append("✅ Maintain safe practices to avoid exposure — avoid sharing razors, toothbrushes, or needles.")
                elif value == "Unknown":
                    explanations.append("❓ The result for Hepatitis B is *unclear or incomplete*. Consider retesting or asking your doctor for further interpretation.")

            elif key == "Hepatitis C":
                if value == "Positive":
                    explanations.insert(0, "🧬 **Hepatitis Serology Guide**")
                    explanations.append("🚨 Hepatitis C result is **positive**. This indicates possible chronic liver infection. Further RNA testing is required to confirm active infection.")
                    explanations.append("➡️ **Next Steps:** Schedule follow-up with a liver specialist. Avoid alcohol and get screened for liver damage.")
                elif value == "Negative":
                    explanations.insert(0, "🧬 **Hepatitis Serology Guide**")
                    explanations.append("✅ Hepatitis C result is **negative**. No current infection detected.")
                    explanations.append("🔁 Retesting may be recommended if you have risk factors like past IV drug use or blood transfusion before 1992.")
                elif value == "Unknown":
                    explanations.append("❓ Hepatitis C status is unknown. Please consult a doctor for clarification or a repeat test.")
                    
            elif key == "HIV":
                explanations.append("🧬 **HIV Serology Guide**")
                if value == "Positive":
                    explanations.append("🚨 **Your HIV test is POSITIVE.** This means HIV antibodies were detected in your blood.")
                    explanations.append("🧠 HIV affects the immune system but **can be managed with modern treatments** that allow you to live a long, healthy life.")
                    explanations.append("🩺 **Next Steps:** Schedule a confirmatory test (e.g., Western blot or PCR), and connect with an HIV care provider for antiretroviral therapy (ART).")
                    explanations.append("🌍 You are not alone. Millions live successfully with HIV. Support, care, and privacy are available.")
                    explanations.append("📌 Use protection, avoid sharing needles, and educate those around you. This helps reduce stigma and protect others.")
                elif value == "Negative":
                    explanations.append("✅ **Your HIV test is NEGATIVE.** No HIV antibodies were detected.")
                    explanations.append("🔁 If you had recent risk exposure, consider retesting in 3–6 weeks. HIV may not be immediately detectable after infection.")
                    explanations.append("🛡️ Keep practicing safe sex, use condoms, and consider PrEP if you're at higher risk.")
                elif value == "Unknown":
                    explanations.append("❓ HIV result is unclear. Please consult a doctor or retest for accuracy.")

            elif key == "Malaria":
                if value == "Positive":
                    explanations.append("🚨 **Malaria Test:** Result is *positive*.")
                    explanations.append("🚨 This may indicate a current infection. Seek immediate medical attention, especially if you have fever, chills, or fatigue.")
                elif value == "Negative":
                    explanations.append("✅ **Malaria Test:** Result is *negative*. No presence of malaria detected.")
                else:
                    explanations.append("❓ **Malaria Test:** Unclear result. Consider retesting or consulting a medical provider.")
            
            
            elif key == "Lung Capacity":
                if value < 2.5:
                    explanations.append("⚠️ **Lung Capacity Test:** Capacity is low (< 2.5L). This could indicate restrictive lung issues. Consult a pulmonologist for spirometry and further evaluation.")
                elif 2.5 <= value <= 5.0:
                    explanations.append("✅ **Lung Capacity Test:** Within normal limits. Maintain regular aerobic exercise to support lung health.")
                else:
                    explanations.append("📈 **Lung Capacity Test:** Above normal (> 5.0L). Could indicate athletic conditioning or measurement error — double-check with a medical provider if unsure.")

            elif key == "Widal Test" and isinstance(value, dict):
                explanations.append("🧬 **Widal Test Serology Guide**")
            
                reactive_flags = []
            
                # Typhi O (TO)
                to = value.get("Typhi O", "").lower()
                if to == "reactive":
                    explanations.append("🚨 **Typhi O (TO):** Reactive — suggests *acute typhoid fever*. Consult your physician immediately.")
                    reactive_flags.append("Typhi O")
                else:
                    explanations.append("✅ **Typhi O (TO):** Non-Reactive — no active typhoid infection detected.")
            
                # Typhi H (TH)
                th = value.get("Typhi H", "").lower()
                if th == "reactive":
                    explanations.append("📌 **Typhi H (TH):** Reactive — indicates *past infection* or typhoid vaccination history.")
                    reactive_flags.append("Typhi H")
                else:
                    explanations.append("✅ **Typhi H (TH):** Non-Reactive — no evidence of past typhoid exposure.")
            
                # Paratyphi AH
                ah = value.get("Paratyphi AH", "").lower()
                if ah == "reactive":
                    explanations.append("⚠️ **Paratyphi A (AH):** Reactive — possible *Paratyphoid A infection*. Medical consultation advised.")
                    reactive_flags.append("Paratyphi AH")
                else:
                    explanations.append("✅ **Paratyphi A (AH):** Non-Reactive — no sign of S. paratyphi A infection.")
            
                # Paratyphi BH
                bh = value.get("Paratyphi BH", "").lower()
                if bh == "reactive":
                    explanations.append("⚠️ **Paratyphi B (BH):** Reactive — may suggest *Paratyphoid B*. Further testing recommended.")
                    reactive_flags.append("Paratyphi BH")
                else:
                    explanations.append("✅ **Paratyphi B (BH):** Non-Reactive — no sign of S. paratyphi B infection.")
            
                # Overall Summary
                if not reactive_flags:
                    explanations.append("✅ Overall Summary: All markers are non-reactive. No signs of typhoid or paratyphoid infections.")
                else:
                    explanations.append(f"🔬 Overall Summary: Reactive results for: {', '.join(reactive_flags)}. Prompt clinical review is recommended.")

        return "\n\n".join(explanations)

    except Exception as e:
        return f"⚠️ Error processing vital signs: {e}"


vital_sign_monitoring_tool = Tool(
    name="VitalSignsMonitoring",
    func=monitor_vital_signs,
    description="Monitors vital signs, detects abnormal patterns, and provides health risk alerts with recommendations."
)
