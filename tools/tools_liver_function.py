# tools/tools_liver_function.py
import re
import pdfplumber
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

# --- Enums ---
class MedicalConditionEnum(str, Enum):
    none = "None"
    hypertension = "Hypertension"
    high_cholesterol = "High cholesterol/triglycerides"
    cardiovascular = "Cardiovascular disease"
    cirrhosis = "Liver Cirrhosis"
    hepatitis_b = "Hepatitis B"
    hepatitis_c = "Hepatitis C"
    fatty_liver = "Fatty Liver Disease"
    multiple = "Multiple"

class SmokingAlcoholEnum(str, Enum):
    non = "Non-smoker, non-drinker"
    occasional = "Occasional smoker or drinker"
    regular = "Regular smoker or drinker"
    heavy = "Heavy smoker or drinker"

class DietaryHabitsEnum(str, Enum):
    mostly_healthy = "Mostly Healthy (Fruits, Vegetables, Whole Grains)"
    moderately_healthy = "Moderately Healthy (Mix of healthy and processed foods)"
    mostly_unhealthy = "Mostly Unhealthy (Processed Foods and Sugary Drinks 1–2 times a week)"
    very_unhealthy = "Very Unhealthy (Processed Foods and Sugary Drinks 3–5 times a week)"
    unsure = "Unsure"

class MedicationsEnum(str, Enum):
    bp_med = "Blood Pressure Medication"
    steroids = "Steroids"
    antipsychotics = "Antipsychotics"
    recreational_drugs = "Recreational Drugs"
    liver_related = "Medications Related to Liver Disease (e.g., Methotrexate, Amiodarone)"
    multiple = "Multiple Medications"
    none = "None"

class SymptomEnum(str, Enum):
    jaundice = "Jaundice (yellowing of skin/eyes)"
    fatigue = "Fatigue"
    abdominal_pain = "Abdominal Pain"
    nausea = "Nausea"
    vomiting = "Vomiting"
    dark_urine = "Dark Urine"
    pale_stools = "Pale Stools"
    none = "None"

class HepatitisMarkerEnum(str, Enum):
    hbsag = "HBsAg (Hepatitis B Surface Antigen)"
    hcv_rna = "HCV RNA (Hepatitis C Virus RNA)"
    hav_igg = "HAV IgG (Hepatitis A Virus IgG)"
    hav_igm = "HAV IgM (Hepatitis A Virus IgM)"
    hbcab = "HBcAb (Hepatitis B Core Antibody)"
    hbeag = "HBeAg (Hepatitis B e Antigen)"
    hbeab = "HBeAb (Hepatitis B e Antibody)"
    hdv_rna = "HDV RNA (Hepatitis D Virus RNA)"
    hev_igg = "HEV IgG (Hepatitis E Virus IgG)"
    hev_igm = "HEV IgM (Hepatitis E Virus IgM)"
    none = "None"

class ManualEntryRequest(BaseModel):
    ALT_SGPT: Optional[float] = 0.0
    AST_SGOT: Optional[float] = 0.0
    ALP: Optional[float] = 0.0
    GGT: Optional[float] = 0.0
    Total_Bilirubin: Optional[float] = 0.0
    Direct_Bilirubin: Optional[float] = 0.0
    Albumin: Optional[float] = 0.0
    INR: Optional[float] = 0.0
    Ammonia: Optional[float] = 0.0
    LDH: Optional[float] = 0.0
    Globulin: Optional[float] = 0.0
    AG_Ratio: Optional[float] = 0.0
    ALT_AST_Ratio: Optional[float] = 0.0
    Indirect_Bilirubin: Optional[float] = 0.0
    Total_Protein: Optional[float] = 0.0

    medical_conditions: MedicalConditionEnum
    symptoms: List[SymptomEnum]
    smoking_alcohol_use: SmokingAlcoholEnum
    dietary_habits: DietaryHabitsEnum
    medications: MedicationsEnum
    hepatitis_markers: List[HepatitisMarkerEnum]
    
    def to_extracted_values(self):
        return {
            "ALT (SGPT)": self.ALT_SGPT,
            "AST (SGOT)": self.AST_SGOT,
            "ALP": self.ALP,
            "GGT": self.GGT,
            "Total Bilirubin": self.Total_Bilirubin,
            "Direct Bilirubin": self.Direct_Bilirubin,
            "Albumin": self.Albumin,
            "INR": self.INR,
            "Ammonia": self.Ammonia,
            "LDH": self.LDH,
            "Globulin": self.Globulin,
            "A/G Ratio": self.AG_Ratio,
            "ALT:AST Ratio": self.ALT_AST_Ratio,
            "Indirect Bilirubin": self.Indirect_Bilirubin,
            "Total Protein": self.Total_Protein,
        }
    
# --- Tool Functions ---
def extract_lft_values(text):
    patterns = {
        "Total Bilirubin": r"(?i)\b(?:Total Bilirubin|Bilirubin Total|Serum Bilirubin \(Total\))\b.*?[:=]?\s*([\d.]+)\s*(?:mg/dL)?",
        "Direct Bilirubin": r"(?i)\b(?:Direct Bilirubin|Bilirubin Direct|Serum Bilirubin \(Direct\))\b.*?[:=]?\s*([\d.]+)\s*(?:mg/dL)?",
        "Indirect Bilirubin": r"(?i)\b(?:Indirect Bilirubin|Bilirubin Indirect|Serum Bilirubin \(Indirect\))\b.*?[:=]?\s*([\d.]+)\s*(?:mg/dL)?",
        "ALT (SGPT)": r"(?i)\b(?:ALT|Alanine Transaminase|SGPT)\b.*?[:=]?\s*([\d.]+)\s*(?:U/L|IU/L)?",
        "AST (SGOT)": r"(?i)\b(?:AST|SGOT|Aspartate Transaminase)\b.*?[:=]?\s*([\d.]+)\s*(?:U/L|IU/L)?",
        "ALP": r"(?i)\b(?:ALP|Alkaline Phosphatase|Serum Alkaline Phosphatase)\b.*?[:=]?\s*([\d.]+)\s*(?:U/L|IU/L)?",
        "GGT": r"(?i)\b(?:GGT|Gamma-Glutamyl Transferase|Gamma GT)\b.*?[:=]?\s*([\d.]+)\s*(?:U/L|IU/L)?",
        "Total Protein": r"(?i)\b(?:Total Protein|Serum Protein)\b.*?[:=]?\s*([\d.]+)\s*(?:g/dL)?",
        "Albumin": r"(?i)\b(?:Albumin|Serum Albumin)\b.*?[:=]?\s*([\d.]+)\s*(?:g/dL)?",
        "Globulin": r"(?i)\b(?:Globulin|Serum Globulin)\b.*?[:=]?\s*([\d.]+)\s*(?:g/dL)?",
        "A/G Ratio": r"(?i)\b(?:A\s*:\s*G Ratio|A/G Ratio|Albumin/Globulin Ratio)\b.*?[:=]?\s*([\d.]+)",
        "INR": r"(?i)\b(?:INR|Prothrombin Time \(INR\))\b.*?[:=]?\s*([\d.]+)",
        "Ammonia": r"(?i)\b(?:Ammonia)\b.*?[:=]?\s*([\d.]+)\s*(?:µg/dL)?",
        "LDH": r"(?i)\b(?:LDH|Lactate Dehydrogenase)\b.*?[:=]?\s*([\d.]+)\s*(?:U/L|IU/L)?",
        "Age": r"(?i)\bAge\b.*?[:=]?\s*(\d+)(?:\s*years)?",
        "Gender": r"(?i)\b(?:Sex|Gender)\b.*?[:=]?\s*(Male|Female|M|F)\b"
    }
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = [g for g in match.groups() if g and re.match(r"^\d{1,3}(,\d{3})*(\.\d+)?$|^(Male|Female|M|F)$", g.strip(), re.IGNORECASE)]
            if groups:
                results[key] = groups[-1] if key == "Gender" else float(groups[-1])
    # Calculate ALT:AST Ratio if both values are available
    if "ALT (SGPT)" in results and "AST (SGOT)" in results and results["AST (SGOT)"] != 0:
        results["ALT:AST Ratio"] = round(results["ALT (SGPT)"] / results["AST (SGOT)"], 2)
    return results


def analyze_liver_function(
    extracted_values: Dict[str, Any],
    dietary_habits: Optional[str] = None,
    medications: Optional[str] = None,
    symptoms: Optional[List[str]] = None,
    hepatitis_markers: Optional[List[str]] = None,
    smoking_alcohol_use: Optional[str] = None,
    medical_conditions: Optional[str] = None,
    input_method: str = "Manual Entry"
):
    # Confidence Level
    all_params = [
        "ALT (SGPT)", "AST (SGOT)", "ALP", "GGT", "Total Bilirubin", "Direct Bilirubin",
        "Albumin", "INR", "Ammonia", "LDH", "Globulin", "A/G Ratio", "ALT:AST Ratio", "Indirect Bilirubin", "Total Protein"
    ]
    missing_parameters = [
        key for key in all_params
        if key not in extracted_values or extracted_values[key] == 0.0
    ]
    confidence_level = "High" if not missing_parameters else "Medium" if len(missing_parameters) <= 3 else "Low"

    # Parameter Status with grading and clinical interpretations
    parameter_status = []
    key_observations = []
    abnormal_flags = set()
    # Reference ranges and interpretations for each marker
    ref_ranges = {
        "ALT (SGPT)":      {"low": None, "high": 56, "unit": "U/L", "decreased": None, "elevated": "Elevated (suggests liver inflammation or injury)", "normal": "Normal"},
        "AST (SGOT)":      {"low": None, "high": 40, "unit": "U/L", "decreased": None, "elevated": "Elevated (suggests liver damage or muscle injury)", "normal": "Normal"},
        "ALP":             {"low": 44, "high": 120, "unit": "U/L", "decreased": "Decreased (may suggest malnutrition, hypothyroidism, or zinc deficiency)", "elevated": "Elevated (suggests bile duct obstruction, bone/liver disease)", "normal": "Normal"},
        "GGT":             {"low": 0, "high": 60, "unit": "U/L", "decreased": None, "elevated": "Elevated (suggests cholestasis, alcohol use, or drug effect)", "normal": "Normal"},
        "Total Bilirubin": {"low": 0.3, "high": 1.2, "unit": "mg/dL", "decreased": None, "elevated": "Elevated (suggests jaundice or liver dysfunction)", "normal": "Normal"},
        "Direct Bilirubin": {"low": 0.0, "high": 0.3, "unit": "mg/dL", "decreased": None, "elevated": "Elevated (suggests cholestasis or hepatocellular disease)", "normal": "Normal"},
        "Indirect Bilirubin": {"low": 0.2, "high": 1.0, "unit": "mg/dL", "decreased": None, "elevated": "Elevated (suggests hemolysis or Gilbert's syndrome)", "normal": "Normal"},
        "Albumin":         {"low": 3.5, "high": 5.0, "unit": "g/dL", "decreased": "Decreased (suggests chronic liver disease or malnutrition)", "elevated": None, "normal": "Normal"},
        "Globulin":        {"low": 2.0, "high": 3.5, "unit": "g/dL", "decreased": "Decreased (may suggest immune deficiency or nephrotic syndrome)", "elevated": "Elevated (suggests chronic inflammation or liver disease)", "normal": "Normal"},
        "A/G Ratio":       {"low": 1.0, "high": 2.2, "unit": "", "decreased": "Decreased (suggests chronic liver or kidney disease)", "elevated": "Elevated (may suggest genetic conditions or high protein intake)", "normal": "Normal"},
        "INR":             {"low": 0.8, "high": 1.2, "unit": "", "decreased": "Decreased (may suggest high clotting tendency)", "elevated": "Elevated (suggests impaired liver synthetic function or anticoagulation)", "normal": "Normal"},
        "Ammonia":         {"low": 15, "high": 45, "unit": "µg/dL", "decreased": None, "elevated": "Elevated (suggests hepatic encephalopathy or severe liver dysfunction)", "normal": "Normal"},
        "LDH":             {"low": 140, "high": 280, "unit": "U/L", "decreased": "Decreased (rare, may suggest malnutrition)", "elevated": "Elevated (suggests tissue damage, hemolysis, or liver disease)", "normal": "Normal"},
        "ALT:AST Ratio":   {"low": 1, "high": 2, "unit": "", "decreased": "Decreased (<1, suggests possible alcoholic liver disease)", "elevated": "Elevated (>2, suggests alcoholic hepatitis)", "normal": "Normal (1-2)"},
        "Total Protein":   {"low": 6.0, "high": 8.3, "unit": "g/dL", "decreased": "Decreased (suggests malnutrition, nephrotic syndrome, or liver disease)", "elevated": "Elevated (suggests chronic inflammation or infection)", "normal": "Normal"},
    }

    for param in all_params:
        if param in extracted_values and extracted_values[param] != 0.0:
            value = extracted_values[param]
            ref = ref_ranges[param]
            # Grading and interpretation
            if ref["low"] is not None and value < ref["low"]:
                status = "Decreased"
                interpretation = ref["decreased"] if ref["decreased"] else ref["normal"]
                abnormal_flags.add("decreased")
            elif ref["high"] is not None and value > ref["high"]:
                status = "Elevated"
                interpretation = ref["elevated"] if ref["elevated"] else ref["normal"]
                abnormal_flags.add("elevated")
            else:
                status = "Normal"
                interpretation = ref["normal"]
            # Special case for ALT:AST Ratio
            if param == "ALT:AST Ratio":
                if value < ref["low"]:
                    status = "Decreased"
                    interpretation = ref["decreased"]
                    abnormal_flags.add("decreased")
                elif value > ref["high"]:
                    status = "Elevated"
                    interpretation = ref["elevated"]
                    abnormal_flags.add("elevated")
                else:
                    status = "Normal"
                    interpretation = ref["normal"]
            parameter_status.append(
                f"{param}: {value} {ref['unit']} → {status} ({interpretation})"
            )

    # --- Risk Level Ranking ---
    if "elevated" in abnormal_flags and "decreased" in abnormal_flags:
        risk_level = "High"
    elif "elevated" in abnormal_flags or "decreased" in abnormal_flags:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # --- Recommendations Section ---
    recommendations = []
    recommendations_generated = False

    if input_method == "Upload PDF":
        if "elevated" in abnormal_flags and "decreased" in abnormal_flags:
            recommendations.append("Some of your liver test results are elevated and some are decreased. Please consult your doctor for a comprehensive evaluation.")
        elif "elevated" in abnormal_flags:
            recommendations.append("Some of your liver test results are elevated. Please consult your doctor for further evaluation.")
        elif "decreased" in abnormal_flags:
            recommendations.append("Some of your liver test results are decreased. Please consult your doctor for further evaluation.")
        else:
            recommendations.append("All your liver test results are within normal limits. Maintain a healthy lifestyle.")
        recommendations_generated = True

    else:
        if "elevated" in abnormal_flags and "decreased" in abnormal_flags:
            recommendations.append("Some of your liver test results are elevated and some are decreased. Please consult your doctor for a comprehensive evaluation.")
            recommendations_generated = True
        elif "elevated" in abnormal_flags:
            recommendations.append("Some of your liver test results are elevated. Please consult your doctor for further evaluation.")
            recommendations_generated = True
        elif "decreased" in abnormal_flags:
            recommendations.append("Some of your liver test results are decreased. Please consult your doctor for further evaluation.")
            recommendations_generated = True
        else:
            recommendations.append("All your liver test results are within normal limits. Maintain a healthy lifestyle.")
            recommendations_generated = True

        # Clinical/lifestyle-based recommendations
        # Symptoms
        if symptoms and SymptomEnum.jaundice in symptoms:
            recommendations.append("Yellowing of skin or eyes (jaundice) detected. Seek urgent medical attention.")
        if symptoms and (SymptomEnum.abdominal_pain in symptoms or SymptomEnum.nausea in symptoms or SymptomEnum.vomiting in symptoms):
            recommendations.append("Symptoms such as abdominal pain, nausea, or vomiting may indicate liver or digestive issues. Please consult your doctor.")

        # Dietary habits
        if dietary_habits in [
            DietaryHabitsEnum.mostly_unhealthy,
            DietaryHabitsEnum.very_unhealthy
        ]:
            recommendations.append("Consider improving your dietary habits to reduce liver strain. Focus on fruits, vegetables, and whole grains.")

        # Hepatitis markers
        if hepatitis_markers and (
            HepatitisMarkerEnum.hbsag in hepatitis_markers or HepatitisMarkerEnum.hcv_rna in hepatitis_markers
        ):
            recommendations.append("Positive hepatitis markers detected. Follow up with viral hepatitis screening and liver ultrasound.")

        # Smoking/alcohol use
        if smoking_alcohol_use == SmokingAlcoholEnum.heavy:
            recommendations.append("Heavy smoking or alcohol use can worsen liver health. Please consider reducing or quitting and consult your doctor.")
        elif smoking_alcohol_use == SmokingAlcoholEnum.regular:
            recommendations.append("Regular smoking or alcohol use may impact your liver. Moderation and medical advice are recommended.")

        # Medical conditions
        if medical_conditions in [
            MedicalConditionEnum.cirrhosis,
            MedicalConditionEnum.hepatitis_b,
            MedicalConditionEnum.hepatitis_c,
            MedicalConditionEnum.fatty_liver
        ]:
            recommendations.append("Your medical history indicates a liver-related condition. Ensure regular follow-up with your healthcare provider.")

        # Medications
        if medications in [
            MedicationsEnum.steroids,
            MedicationsEnum.antipsychotics,
            MedicationsEnum.recreational_drugs,
            MedicationsEnum.liver_related
        ]:
            recommendations.append("Your medication use may contribute to liver strain. Consult your doctor about possible alternatives.")

    return {
        "confidence_level": confidence_level,
        "risk_level": risk_level,
        "missing_parameters": missing_parameters,
        "parameter_status": parameter_status,
        "key_observations": key_observations,
        "recommendations": recommendations
    }