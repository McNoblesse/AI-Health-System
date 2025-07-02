import json
# Default health data template
def get_default_health_data():
    return {
        "data": {
            "Glucose": 0.0,
            "Glucose_Score": 0,
            "Glucose_Unit": "mg/dL",
            "SpO2": 0,
            "SpO2_Score": 0,
            "SpO2_Unit": "%",
            "ECG (Heart Rate)": 0,
            "ECG_Score": 0,
            "ECG_Unit": "BPM",
            "Blood Pressure (Systolic)": 0,
            "Blood Pressure (Diastolic)": 0,
            "Blood_Pressure_Score": 0,
            "Blood_Pressure_Unit": "mmHg",
            "Weight (BMI)": 0.0,
            "Weight_Score": 0,
            "Weight_Unit": "lb",
            "Temperature": 0.0,
            "Temperature_Score": 0,
            "Temperature_Unit": "°C",
            "Malaria": "Unknown",
            "Malaria_Score": 0,
            "Widal Test": "Unknown",
            "Widal_Score": 0,
            "Hepatitis B": "Unknown",
            "Hepatitis_B_Score": 0,
            "Voluntary Serology": "Unknown",
            "Voluntary_Serology_Score": 0,
            "Perfusion_index": 0,
            "Perfusion_index_Unit": "L/min",
            "Waist Circumference": 0,
            "Waist_Circumference_Score": 0,
            "Waist_Circumference_Unit": "cm",
            "Fev": 0.0,
            "Fev_unit": "L",
            "Blood_Pressure_Systolic_Score": 0,
            "Blood_Pressure_Diastolic_Score": 0,
            "Widal_Test_Score": 0,
            "Total_Health_Score": 0,
            "Health_Category": "Unknown"
        }
    }

# Collect user input for health data
def collect_user_health_data():
    data = get_default_health_data()["data"]
    print("Enter your health data (leave blank to skip and use default values):")
    for key in data.keys():
        if isinstance(data[key], (int, float)):
            user_input = input(f"{key} ({data[key]}): ")
            if user_input.strip():
                data[key] = float(user_input) if '.' in user_input else int(user_input)
        elif isinstance(data[key], str):
            user_input = input(f"{key} ({data[key]}): ")
            if user_input.strip():
                data[key] = user_input
    return {"data": data}

# Analyze health data
def analyze_health_data(health_data):
    analysis = {}
    vitals_to_improve = []

    # Check vitals and scores
    for key, value in health_data["data"].items():
        if "Score" in key and value < 100:
            vital_name = key.replace("_Score", "").replace("_", " ")
            vitals_to_improve.append(vital_name)

    # Generate analysis
    analysis["Total_Health_Score"] = health_data["data"]["Total_Health_Score"]
    analysis["Health_Category"] = health_data["data"]["Health_Category"]
    analysis["Vitals_Need_Improvement"] = vitals_to_improve

    # Provide personalized health tips
    tips = []
    if "Glucose" in vitals_to_improve:
        tips.append("Maintain a balanced diet to regulate blood sugar levels.")
    if "SpO2" in vitals_to_improve:
        tips.append("Practice deep breathing exercises to improve oxygen saturation.")
    if "Blood Pressure" in vitals_to_improve:
        tips.append("Reduce salt intake and manage stress to improve blood pressure.")
    if "Weight" in vitals_to_improve:
        tips.append("Incorporate regular exercise and monitor calorie intake.")
    if "Temperature" in vitals_to_improve:
        tips.append("Stay hydrated and monitor for signs of fever or hypothermia.")
    if "Waist Circumference" in vitals_to_improve:
        tips.append("Engage in core-strengthening exercises to reduce waist size.")

    analysis["Health_Tips"] = tips
    return analysis

# Simulate sending automated reports
def send_report(user_email, analysis):
    report = {
        "to": user_email,
        "subject": "Your Health Score Analysis Report",
        "body": json.dumps(analysis, indent=4)
    }
    print(f"Sending report to {user_email}...")
    print(json.dumps(report, indent=4))

# Main function
if __name__ == "__main__":
    # Collect user health data
    user_data = collect_user_health_data()

    # Analyze health data
    analysis = analyze_health_data(user_data)

    # Print analysis
    print("Health Analysis:")
    print(json.dumps(analysis, indent=4))

    # Simulate sending report
    user_email = input("Enter your email to receive the report: ")
    send_report(user_email, analysis)