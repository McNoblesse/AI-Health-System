# tools/tools_auto_bio_completer.py

def check_profile_completeness(profile: dict) -> dict:
    """Check for missing fields in a health profile and return prompts and reasons."""
    
    required_fields = {
        "age": "Helps calculate risk scores and recommend age-specific tests.",
        "sex": "Essential for gender-specific conditions and risk factors.",
        "height": "Used to calculate BMI and assess body composition.",
        "weight": "Used to calculate BMI and monitor weight-related risks.",
        "blood_type": "Important during emergencies and some medication decisions.",
        "chronic_conditions": "Used to tailor health recommendations (e.g., asthma, diabetes).",
        "allergies": "Ensures treatment safety and avoids harmful substances.",
        "smoking": "Affects cardiovascular and respiratory risk scoring.",
        "alcohol": "Impacts liver health and blood pressure risk.",
        "physical_activity": "Influences risk for diabetes, obesity, and heart disease.",
        "diet": "Important for nutrition-based recommendations.",
        "family_history": "Used to calculate hereditary risk for conditions like cancer or hypertension.",
        "medications": "Ensures accurate recommendations and avoids interaction risks."
    }

    missing = []
    prompts = []
    explanations = []

    for field, reason in required_fields.items():
        if field not in profile or profile[field] in [None, "", [], {}, "null"]:
            missing.append(field)
            prompts.append(f"Please provide your {field.replace('_', ' ')}.")
            explanations.append(f"🧠 **{field.replace('_', ' ').title()}**: {reason}")

    return {
        "missing_fields": missing,
        "prompts": prompts,
        "explanations": explanations,
        "completeness_score": round((1 - len(missing)/len(required_fields)) * 100, 2)
    }
