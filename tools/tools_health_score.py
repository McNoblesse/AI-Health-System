class HealthScoreAnalysisTool:
    def __init__(self):
        self.scoring_criteria = {
            "Glucose": {"range": (70, 100), "unit": "mg/dL"},
            "SpO2": {"range": (95, 100), "unit": "%"},
            "Blood Pressure (Systolic)": {"range": (90, 120), "unit": "mmHg"},
            "Blood Pressure (Diastolic)": {"range": (60, 80), "unit": "mmHg"},
            "Weight (BMI)": {"range": (18.5, 24.9), "unit": "kg/m²"},
            "Temperature": {"range": (36.5, 37.5), "unit": "°C"},
            "ECG (Heart Rate)": {"range": (60, 100), "unit": "BPM"},
            "Malaria": {"range": "Negative", "unit": "Binary"},
            "Widal Test": {"range": "Negative", "unit": "Binary"},
            "Hepatitis B": {"range": "Negative", "unit": "Binary"},
            "Voluntary Serology": {"range": "Negative", "unit": "Binary"},
        }

    def evaluate_health_metric(self, metric, value):
        """Evaluates a health metric against its normal range."""
        if metric not in self.scoring_criteria:
            return "Unknown"

        criteria = self.scoring_criteria[metric]
        expected_range = criteria["range"]

        if isinstance(expected_range, tuple):  # Numerical value check
            if not isinstance(value, (int, float)):
                return "Invalid data"
            low, high = expected_range
            if low <= value <= high:
                return "Normal"
            elif value < low:
                return "Low"
            else:
                return "High"

        elif isinstance(expected_range, str):  # Binary test check
            if value == expected_range:
                return "Negative"
            else:
                return "Positive"

        return "Unknown"

    def generate_report(self, health_data: dict) -> dict:
        total_score = 0
        max_score = 0
        vitals_needing_improvement = []
        improvement_tips = []

        for key, value in health_data.items():
            if value in [None, '', 'null']:
                continue  # Skip missing or null fields

            if key == "Weight (BMI)":
                max_score += 10
                if isinstance(value, (int, float)):
                    if value < 18.5:
                        vitals_needing_improvement.append(f"{key} (Low)")
                        improvement_tips.append("Gain weight to reach a healthy BMI range.")
                    elif 18.5 <= value <= 24.9:
                        total_score += 10
                    elif 25 <= value <= 29.9:
                        total_score += 5
                        vitals_needing_improvement.append(f"{key} (Moderately High)")
                        improvement_tips.append("Reduce BMI slightly through diet and exercise.")
                    else:
                        vitals_needing_improvement.append(f"{key} (High)")
                        improvement_tips.append("Reduce Weight (BMI) through proper lifestyle changes.")

            elif key == "Temperature":
                max_score += 5
                if isinstance(value, (int, float)):
                    if 36.1 <= value <= 37.2:
                        total_score += 5
                    else:
                        vitals_needing_improvement.append(f"{key} (Low)" if value < 36.1 else f"{key} (High)")
                        improvement_tips.append("Adjust temperature through dietary or medical guidance.")

            elif key == "Blood Pressure (Systolic)":
                max_score += 5
                if isinstance(value, (int, float)):
                    if 90 <= value <= 120:
                        total_score += 5
                    else:
                        vitals_needing_improvement.append(f"{key} (Abnormal)")
                        improvement_tips.append("Monitor systolic pressure with a doctor.")

            elif key == "Blood Pressure (Diastolic)":
                max_score += 5
                if isinstance(value, (int, float)):
                    if 60 <= value <= 80:
                        total_score += 5
                    else:
                        vitals_needing_improvement.append(f"{key} (Abnormal)")
                        improvement_tips.append("Monitor diastolic pressure with a doctor.")

            elif key in ["Malaria", "Widal Test", "Hepatitis B", "Voluntary Serology"]:
                max_score += 5
                if isinstance(value, str) and value.lower() == "negative":
                    total_score += 5
                else:
                    vitals_needing_improvement.append(f"{key} (Positive)")
                    improvement_tips.append(f"Seek medical attention for {key}.")

            elif key == "SpO2":
                max_score += 5
                if isinstance(value, (int, float)):
                    if value >= 95:
                        total_score += 5
                    else:
                        vitals_needing_improvement.append(f"{key} (Low)")
                        improvement_tips.append("Improve oxygen level through breathing exercises or see a doctor.")

            elif key == "ECG (Heart Rate)":
                max_score += 5
                if isinstance(value, (int, float)):
                    if 60 <= value <= 100:
                        total_score += 5
                    else:
                        vitals_needing_improvement.append(f"{key} (Abnormal)")
                        improvement_tips.append("Consult a cardiologist about your heart rate.")

            elif key == "Waist Circumference":
                max_score += 5
                if isinstance(value, (int, float)) and value <= 94:
                    total_score += 5
                else:
                    vitals_needing_improvement.append(f"{key} (High)")
                    improvement_tips.append("Reduce waist size through targeted exercise.")

            elif key == "Fev":
                max_score += 5
                if isinstance(value, (int, float)) and value >= 80:
                    total_score += 5
                else:
                    vitals_needing_improvement.append(f"{key} (Low)")
                    improvement_tips.append("Improve respiratory function with breathing therapy.")

            elif key == "Perfusion_index":
                max_score += 5
                if isinstance(value, (int, float)) and 0.02 <= value <= 20:
                    total_score += 5
                else:
                    vitals_needing_improvement.append(f"{key} (Abnormal)")
                    improvement_tips.append("Check perfusion with a professional.")

        # Normalize score
        final_score = round((total_score / max_score) * 100) if max_score > 0 else 0

        # Health status logic
        if final_score >= 85:
            status = "Excellent"
        elif final_score >= 70:
            status = "Good"
        elif final_score >= 50:
            status = "Fair"
        else:
            status = "Poor"

        return {
            "Total Score": final_score,
            "Health Status": status,
            "Vitals Needing Improvement": ", ".join(vitals_needing_improvement) if vitals_needing_improvement else "None",
            "Improvement Tips": " ".join(improvement_tips) if improvement_tips else "Keep maintaining your health!",
        }
