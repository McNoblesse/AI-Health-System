def enrich_kidney_parameters(data: dict) -> dict:
    if not data.get("BUN") and data.get("Serum Urea"):
        data["BUN"] = data["Serum Urea"] / 2.14
    if not data.get("ACR") and data.get("Urine Albumin") and data.get("Urine Creatinine"):
        data["ACR"] = data["Urine Albumin"] / data["Urine Creatinine"]
    if not data.get("BUN/Creatinine Ratio") and data.get("BUN") and data.get("Serum Creatinine"):
        data["BUN/Creatinine Ratio"] = data["BUN"] / data["Serum Creatinine"]
    if not data.get("Urea/Creatinine Ratio") and data.get("Serum Urea") and data.get("Serum Creatinine"):
        data["Urea/Creatinine Ratio"] = data["Serum Urea"] / data["Serum Creatinine"]
    if not data.get("eGFR") and data.get("Serum Creatinine") and data.get("Age") and data.get("Sex"):
        k = 0.742 if data["Sex"].lower() == "female" else 1.0
        data["eGFR"] = 186 * (data["Serum Creatinine"] ** -1.154) * (data["Age"] ** -0.203) * k
    return data

def analyze_kidney_function(data: dict):
    analysis = []
    missing_parameters = []
    parameter_info = {
    "BUN": {"range": (7, 20), "elevated": "High levels may indicate kidney dysfunction or dehydration.", "decreased": "Low levels may indicate malnutrition or liver disease."},
    "Serum Urea": {"range": (2.5, 7.1), "elevated": "Elevated levels may suggest kidney issues or high protein intake.", "decreased": "Low levels may indicate malnutrition or liver disease."},
    "Serum Creatinine": {"range": (0.6, 1.2), "elevated": "High levels may indicate impaired kidney function.", "decreased": "Low levels may indicate reduced muscle mass."},
    "eGFR": {"range": (90, float("inf")), "elevated": None, "decreased": "Lower levels indicate reduced kidney filtration capacity."},
    "BUN/Creatinine Ratio": {"range": (10, 20), "elevated": "Elevated BUN/Creatinine Ratio may indicate dehydration or reduced kidney perfusion.", "decreased": "Low BUN/Creatinine Ratio may indicate liver disease or malnutrition."},
    "Urea/Creatinine Ratio": {"range": (40, 100), "elevated": "High Urea/Creatinine Ratio may indicate dehydration or high protein intake.", "decreased": "Low Urea/Creatinine Ratio may indicate liver disease or malnutrition."},
    "Serum Sodium": {"range": (135, 145), "elevated": "High levels may indicate dehydration.", "decreased": "Low levels may indicate overhydration or kidney dysfunction."},
    "Serum Potassium": {"range": (3.5, 5.0), "elevated": "High levels may indicate kidney dysfunction or acidosis.", "decreased": "Low levels may indicate alkalosis or diuretic use."},
    "Serum Calcium": {"range": (8.8, 10.2), "elevated": "High levels may indicate hyperparathyroidism or cancer.", "decreased": "Low levels may indicate kidney disease or vitamin D deficiency."},
    "Serum Uric Acid": {"range": (3.5, 7.2), "elevated": "High levels may indicate gout or kidney dysfunction.", "decreased": "Low levels may indicate liver disease."},
    "Chloride": {"range": (96, 106), "elevated": "High levels may indicate dehydration.", "decreased": "Low levels may indicate alkalosis."},
    "Bicarbonate": {"range": (22, 29), "elevated": "High levels may indicate metabolic alkalosis.", "decreased": "Low levels may indicate metabolic acidosis."},
    "ACR": {"range": (None, 30), "elevated": "High levels indicate increased albumin excretion, a marker of kidney damage.", "decreased": None}
}

    # Analyze each parameter
    for param, value in data.items():
        if value is None:
            missing_parameters.append(param)
        elif param in parameter_info:
            info = parameter_info[param]
            low, high = info["range"]
            if low is not None and high is not None:
                if low <= value <= high:
                    analysis.append(f"{param}: {value} → Normal")
                elif value > high:
                    analysis.append(f"{param}: {value} → High (Above Normal Range). {info['elevated']}")
                elif value < low:
                    analysis.append(f"{param}: {value} → Low (Below Normal Range). {info['decreased']}")
            elif high is not None and value > high:
                analysis.append(f"{param}: {value} → High (Above Normal Range). {info['elevated']}")
            elif low is not None and value < low:
                analysis.append(f"{param}: {value} → Low (Below Normal Range). {info['decreased']}")

    # Add eGFR stages
    if data["eGFR"]:
        egfr = data["eGFR"]
        if egfr >= 90:
            stage = "Stage 1 (Normal or High)"
        elif 60 <= egfr < 90:
            stage = "Stage 2 (Mildly Decreased)"
        elif 30 <= egfr < 60:
            stage = "Stage 3 (Moderate CKD)"
        elif 15 <= egfr < 30:
            stage = "Stage 4 (Severe CKD)"
        else:
            stage = "Stage 5 (Kidney Failure)"
        analysis.append(f"eGFR: {egfr} → {stage}")

    return analysis, missing_parameters

def reorder_extracted_data(data):
    """
    Reorder the extracted data dictionary to ensure calculated values (ACR, eGFR, BUN/Creatinine Ratio, Urea/Creatinine Ratio, BUN)
    are listed last.
    """
    calculated_keys = ["ACR", "eGFR", "BUN/Creatinine Ratio", "Urea/Creatinine Ratio", "BUN"]
    reordered_data = {key: value for key, value in data.items() if key not in calculated_keys}
    for key in calculated_keys:
        if key in data:
            reordered_data[key] = data[key]
    return reordered_data

def kidney_function_analysis_tool(input_data: dict) -> dict:
    enriched_data = enrich_kidney_parameters(input_data)
    enriched_data = reorder_extracted_data(enriched_data)
    analysis, missing = analyze_kidney_function(enriched_data)

    # Initialize recommendation string
    overall_health = "Unable to determine overall kidney health due to missing data."

    # Retrieve key parameters
    egfr = enriched_data.get("eGFR")
    acr = enriched_data.get("ACR")
    bun_creatinine_ratio = enriched_data.get("BUN/Creatinine Ratio")
    sodium = enriched_data.get("Serum Sodium")
    potassium = enriched_data.get("Serum Potassium")
    bicarbonate = enriched_data.get("Bicarbonate")
    chloride = enriched_data.get("Chloride")
    uric_acid = enriched_data.get("Serum Uric Acid")

    # Apply comprehensive logic for interpreting overall kidney status
    if egfr is not None and acr is not None:
        if egfr >= 90 and acr < 30:
            overall_health = "✅ Your kidney health is normal."
        elif egfr < 90 or acr >= 30:
            if egfr < 30:
                overall_health = "❗ You may have severe kidney disease. Immediate medical attention is recommended."
            elif egfr < 60 or acr >= 300:
                overall_health = "⚠️ You may have moderate kidney impairment. Consult a nephrologist."
            else:
                overall_health = "ℹ️ You may have mild kidney impairment. Routine monitoring and lifestyle changes are recommended."
    elif egfr is not None:
        if egfr >= 90:
            overall_health = "✅ Your kidney filtration rate is normal."
        elif egfr < 30:
            overall_health = "❗ You may have severe kidney disease. Immediate medical attention is recommended."
        elif egfr < 60:
            overall_health = "⚠️ You may have moderate kidney impairment. Consult a nephrologist."
        else:
            overall_health = "ℹ️ You may have mild kidney impairment. Routine monitoring and lifestyle adjustments are recommended."
    elif acr is not None:
        if acr < 30:
            overall_health = "✅ Your albumin-to-creatinine ratio is normal."
        else:
            overall_health = "⚠️ You may have kidney damage. Consult a nephrologist for further evaluation."
    else:
        overall_health = "❓ Insufficient data to fully assess kidney health. Please provide eGFR or ACR."

    # Track any still-missing fields for confidence and recommendations
    missing_parameters = [param for param in [
        "eGFR", "ACR", "Serum Creatinine", "Serum Urea", "Serum Potassium",
        "Bicarbonate", "Chloride", "Serum Calcium", "Serum Uric Acid",
        "BUN", "BUN/Creatinine Ratio", "Urea/Creatinine Ratio"
    ] if enriched_data.get(param) is None]

    if missing_parameters:
        overall_health += f"\n🔎 Note: Assessment based on incomplete data. Missing parameters: {', '.join(missing_parameters)}."

    # Determine confidence level
    confidence = "High" if not missing_parameters else "Medium" if len(missing_parameters) <= 3 else "Low"

    return {
        "analysis": analysis,
        "overall_health": overall_health,
        "confidence_level": confidence,
        "missing_parameters": missing_parameters,
        "data": enriched_data
    }


# === Optional CLI Interface for standalone use ===

def get_manual_input():
    extracted_data = {
        "Serum Urea": float(input("Serum Urea (mg/dL): ")),
        "Serum Creatinine": float(input("Serum Creatinine (mg/dL): ")),
        "Serum Sodium": float(input("Serum Sodium (mmol/L): ")),
        "Serum Potassium": float(input("Serum Potassium (mmol/L): ")),
        "Serum Calcium": float(input("Serum Calcium (mg/dL): ")),
        "Serum Uric Acid": float(input("Serum Uric Acid (mg/dL): ")),
        "Urine Albumin": float(input("Urine Albumin (mg/dL): ")),
        "Urine Creatinine": float(input("Urine Creatinine (mg/dL): ")),
        "Chloride": float(input("Chloride (mmol/L): ")),
        "Bicarbonate": float(input("Bicarbonate (mmol/L): ")),
        "Age": int(input("Age (years): ")),
        "Sex": input("Sex (Male/Female): ").strip().capitalize(),
        # Optional parameters
        "BUN": float(input("BUN (optional, leave blank for system to calculate): ") or 0.0),
        "ACR": float(input("ACR (optional, leave blank for system to calculate): ") or 0.0),
        "BUN/Creatinine Ratio": float(input("BUN/Creatinine Ratio (optional, leave blank for system to calculate): ") or 0.0),
        "Urea/Creatinine Ratio": float(input("Urea/Creatinine Ratio (optional, leave blank for system to calculate): ") or 0.0),
        "eGFR": float(input("eGFR (optional, leave blank for system to calculate): ") or 0.0)
    }

    # Calculate BUN if not provided but Serum Urea is available
    if extracted_data["BUN"] == 0.0 and extracted_data["Serum Urea"] != 0.0:
        extracted_data["BUN"] = extracted_data["Serum Urea"] / 2.14

    # Calculate ACR if not provided but Urine Albumin and Urine Creatinine are available
    if extracted_data["ACR"] == 0.0 and extracted_data["Urine Albumin"] != 0.0 and extracted_data["Urine Creatinine"] != 0.0:
        extracted_data["ACR"] = extracted_data["Urine Albumin"] / extracted_data["Urine Creatinine"]

    # Calculate BUN/Creatinine Ratio if not provided but BUN and Serum Creatinine are available
    if extracted_data["BUN/Creatinine Ratio"] == 0.0 and extracted_data["BUN"] != 0.0 and extracted_data["Serum Creatinine"] != 0.0:
        extracted_data["BUN/Creatinine Ratio"] = extracted_data["BUN"] / extracted_data["Serum Creatinine"]

    # Calculate Urea/Creatinine Ratio if not provided but Serum Urea and Serum Creatinine are available
    if extracted_data["Urea/Creatinine Ratio"] == 0.0 and extracted_data["Serum Urea"] != 0.0 and extracted_data["Serum Creatinine"] != 0.0:
        extracted_data["Urea/Creatinine Ratio"] = extracted_data["Serum Urea"] / extracted_data["Serum Creatinine"]

    # Calculate eGFR if not provided but Age, Sex, and Serum Creatinine are available
    if extracted_data["eGFR"] == 0.0 and extracted_data["Age"] != 0 and extracted_data["Serum Creatinine"] != 0.0:
        k = 0.742 if extracted_data["Sex"] == "Female" else 1.0
        extracted_data["eGFR"] = 186 * (extracted_data["Serum Creatinine"] ** -1.154) * (extracted_data["Age"] ** -0.203) * k

    # Reorder calculated values to come last
    extracted_data = reorder_extracted_data(extracted_data)

    return extracted_data

if __name__ == "__main__":
    print("🔍 Kidney Function CLI Tool")
    extracted_data = get_manual_input()

    # Reorder extracted data to ensure calculated values come last
    extracted_data = reorder_extracted_data(extracted_data)

    print("\n🧪 Extracted Data:")
    for key, value in extracted_data.items():
        print(f"{key}: {value if value is not None else 'Not Provided'}")

    # Generate findings and missing parameters
    analysis, missing_parameters = analyze_kidney_function(extracted_data)

    print(f"\n🔬 Kidney Function Analysis:")
    for key in extracted_data.keys():
        for result in analysis:
            if result.startswith(key):
                print(f"   - {result}")

    # Assess kidney health based on available parameters
    print("\n📊 Findings Summary:")
    overall_health = "Unable to determine overall kidney health due to missing data."

    egfr = extracted_data.get("eGFR")
    acr = extracted_data.get("ACR")
    bun_creatinine_ratio = extracted_data.get("BUN/Creatinine Ratio")
    sodium = extracted_data.get("Serum Sodium")
    potassium = extracted_data.get("Serum Potassium")
    bicarbonate = extracted_data.get("Bicarbonate")
    chloride = extracted_data.get("Chloride")
    uric_acid = extracted_data.get("Serum Uric Acid")

    if egfr is not None and acr is not None:
        if egfr >= 90 and acr < 30:
            overall_health = "✅ Your kidney health is normal."
        elif egfr < 90 or acr >= 30:
            if egfr < 30:
                overall_health = "❗ You may have severe kidney disease. Immediate medical attention is recommended."
            elif egfr < 60 or acr >= 300:
                overall_health = "⚠️ You may have moderate kidney impairment. Consult a nephrologist."
            else:
                overall_health = "ℹ️ You may have mild kidney impairment. Routine monitoring and lifestyle changes are recommended."
    elif egfr is not None:
        if egfr >= 90:
            overall_health = "✅ Your kidney filtration rate is normal."
        elif egfr < 30:
            overall_health = "❗ You may have severe kidney disease. Immediate medical attention is recommended."
        elif egfr < 60:
            overall_health = "⚠️ You may have moderate kidney impairment. Consult a nephrologist."
        else:
            overall_health = "ℹ️ You may have mild kidney impairment. Routine monitoring and lifestyle adjustments are recommended."
    elif acr is not None:
        if acr < 30:
            overall_health = "✅ Your albumin-to-creatinine ratio is normal."
        else:
            overall_health = "⚠️ You may have kidney damage. Consult a nephrologist for further evaluation."
    else:
        overall_health = "❓ Insufficient data to fully assess kidney health. Please provide eGFR or ACR."

    # List truly missing parameters for reference
    missing_parameters = [param for param in [
        "eGFR", "ACR", "Serum Creatinine", "Serum Urea", "Serum Potassium",
        "Bicarbonate", "Chloride", "Serum Calcium", "Serum Uric Acid",
        "BUN", "BUN/Creatinine Ratio", "Urea/Creatinine Ratio"
    ] if extracted_data.get(param) is None]

    if missing_parameters:
        overall_health += f"\nNote: This is based on incomplete data. Missing parameters: {', '.join(missing_parameters)}."

    print(overall_health)

    # Add confidence level
    confidence_level = "High" if not missing_parameters else "Medium" if len(missing_parameters) <= 3 else "Low"
    print(f"\n📈 Confidence Level: {confidence_level}")
    if missing_parameters:
        print(f"\n🧪 Additional tests recommended: {', '.join(missing_parameters)}")
