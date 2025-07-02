# tools/tools_lipid_profile.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Reference ranges
REF_RANGES = {
    'total_chol': {'low': '<90', 'high': '≥240', 'borderline': '≥200-239.9', 'optimal': '90-199.9'},
    'ldl': {'low': '<100', 'high': '≥160', 'borderline': '130-159.9', 'optimal': '100-129.9'},
    'hdl_male': {'low': '<40', 'high': '≥60', 'borderline': '40-45', 'optimal': '46-59.9'},
    'hdl_female': {'low': '<50', 'high': '≥60', 'borderline': '50-55', 'optimal': '56-59.9'},
    'triglycerides': {'low': '<60', 'high': '≥200', 'borderline': '150-199.9', 'optimal': '60-149.9'},
    'non_hdl': {'low': '<70', 'high': '≥160', 'borderline': '130-159.9', 'optimal': '70-129.9'},
    'vldl': {'low': '<10', 'high': '≥40', 'borderline': '31-39.9', 'optimal': '10-30'},
}

def classify_component(component, value, sex=None):
    """Classify each lipid component"""
    classifications = {
        'total_chol': [
            (200, 'optimal'), (240, 'borderline'), (float('inf'), 'high')],
        'ldl': [
            (100, 'optimal'), (130, 'near optimal'), (160, 'borderline'),
            (190, 'high'), (float('inf'), 'very high')],
        'hdl_male': [
            (40, 'low'), (46, 'borderline'), (60, 'optimal'), (float('inf'), 'high')],
        'hdl_female': [
            (50, 'low'), (56, 'borderline'), (60, 'optimal'), (float('inf'), 'high')],
        'triglycerides': [
            (150, 'optimal'), (200, 'borderline'), (500, 'high'),
            (float('inf'), 'very high')],
        'non_hdl': [
            (130, 'optimal'), (160, 'borderline'), (190, 'high'),
            (float('inf'), 'very high')],
        'vldl': [
            (30, 'optimal'), (40, 'borderline'), (float('inf'), 'high')]
    }
    
    # Handle HDL based on gender
    if component == 'hdl':
        if sex == 'Male':
            component = 'hdl_male'
        elif sex == 'Female':
            component = 'hdl_female'
        else:
            # Default to male if gender not specified
            component = 'hdl_male'
    
    for threshold, label in classifications[component]:
        if value < threshold:
            return label
    return 'unknown'

def calculate_ascvd_risk(data):
    """Simplified ASCVD risk calculation (in real app, use proper formula)"""
    score = 0
    if data['age'] > 45: score += 1
    if data['smoker'] in ['Occasional smoker', 'Regular smoker', 'Heavy smoker']: score += 1
    if data['hypertension'] == 'Yes': score += 1
    if data['diabetes'] in ['Yes, diabetic', 'Yes, pre-diabetic/borderline diabetic']: score += 1
    if data['family_history'] in ['Yes, in immediate family (parents or siblings)', 
                             'Yes, in extended family (grandparents, uncles, aunts)']: score += 1
    
    if 'ldl' in data and data['ldl'] > 160: score += 2
    elif 'ldl' in data and data['ldl'] > 130: score += 1
    
    if score < 2: return 'Low'
    elif 2 <= score < 4: return 'Borderline'
    elif 4 <= score < 6: return 'Intermediate'
    else: return 'High'

def generate_recommendations(data, analysis, ascvd_risk):
    """Generate personalized recommendations"""
    recs = []
    
    # General recommendations
    recs.append("Maintain a heart-healthy diet (Mediterranean diet recommended)")
    recs.append("Aim for at least 150 minutes of moderate exercise weekly")

    # Non-HDL specific
    if 'non_hdl' in analysis:
        if analysis['non_hdl'] in ['high', 'very high']:
            recs.append("Focus on reducing LDL and triglyceride levels to lower Non-HDL cholesterol")
    
    # VLDL specific
    if 'vldl' in analysis:
        if analysis['vldl'] in ['high']:
            recs.append("High VLDL suggests need to reduce simple carbohydrates and alcohol intake")
            recs.append("Consider increasing omega-3 fatty acid consumption")
    
    # LDL-specific
    if 'ldl' in analysis:
        if analysis['ldl'] in ['high', 'very high']:
            recs.append("Consider reducing saturated fats and increasing soluble fiber")
            if ascvd_risk in ['Intermediate', 'High']:
                recs.append("Consult your doctor about statin therapy")
    
    # HDL-specific
    if 'hdl' in analysis and analysis['hdl'] == 'poor':
        recs.append("Increase physical activity to raise HDL levels")
        recs.append("Consider healthy fat sources like olive oil and fatty fish")
    
    # Triglycerides-specific
    if 'triglycerides' in analysis and analysis['triglycerides'] in ['high', 'very high']:
        recs.append("Reduce intake of refined carbohydrates and sugars")
        recs.append("Limit alcohol consumption")
    
    # Risk-specific
    if ascvd_risk == 'High':
        recs.append("Urgent consultation with a cardiologist recommended")
    elif ascvd_risk == 'Intermediate':
        recs.append("Consider more frequent lipid monitoring (every 3-6 months)")
    
    return recs

def plot_lipid_profile(data, save_path=None):
    """Create visualization of lipid profile"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    components = ['Total Cholesterol', 'LDL', 'HDL', 'Triglycerides', 'Non-HDL', 'VLDL']
    values = [data.get('total_chol', 0), data.get('ldl', 0), 
              data.get('hdl', 0), data.get('triglycerides', 0),
              data.get('non_hdl', 0), data.get('vldl', 0)]
    
    # Ideal values
    ideals = [200, 100, 60, 150, 130, 30]
    
    # Plot
    x = range(len(components))
    ax.bar(x, values, width=0.4, label='Your Values')
    ax.bar([i + 0.4 for i in x], ideals, width=0.4, label='Ideal Values')
    
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(components, rotation=45, ha='right')
    ax.set_ylabel('mg/dL')
    ax.set_title('Your Lipid Profile vs Ideal Values')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig

def print_results(data, analysis, ascvd_risk, recommendations):
    """Print analysis results to console"""
    print("\n==== 📊 Your Lipid Profile Analysis ====")
    
    # Print metrics
    print("\nLipid Values:")
    metrics = [
        ("Total Cholesterol", data.get('total_chol', 0), 'total_chol'),
        ("LDL Cholesterol", data.get('ldl', 0), 'ldl'),
        ("HDL Cholesterol", data.get('hdl', 0), 'hdl'),
        ("Triglycerides", data.get('triglycerides', 0), 'triglycerides'),
        ("Non-HDL Cholesterol", data.get('non_hdl', 0), 'non_hdl'),
        ("VLDL Cholesterol", data.get('vldl', 0), 'vldl')
    ]
    
    for label, value, key in metrics:
        if key in analysis:
            print(f"{label}: {value} mg/dL - {analysis.get(key, '')}")
        else:
            print(f"{label}: {value} mg/dL")
    
    # Show ASCVD risk
    print(f"\n🫀 Cardiovascular Risk: {ascvd_risk}")
    
    # Show recommendations
    print("\n💡 Personalized Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Show reference ranges
    print("\n📋 Reference Ranges:")
    for component, ranges in REF_RANGES.items():
        print(f"- {component}:")
        for level, value in ranges.items():
            print(f"  {level}: {value}")


def analyze_lipid_profile(data):
    """Analyze lipid profile and generate recommendations"""
    # Calculate Non-HDL if not provided but total and HDL are available
    if 'non_hdl' not in data and 'total_chol' in data and 'hdl' in data:
        data['non_hdl'] = data['total_chol'] - data['hdl']
    
    # Calculate VLDL if not provided but triglycerides are available
    if 'vldl' not in data and 'triglycerides' in data:
        data['vldl'] = data['triglycerides'] // 5  # Common estimation method
    
    # Risk assessment
    risk_factors = {
        'age': data['age'],
        'smoker': 0 if data['smoker'] == 'Non-smoker' else 1,  # Non-smoker maps to 0 (No), all others to 1 (Yes)
        'hypertension': 1 if data['hypertension'] == 'Yes' else 0,
        'diabetes': 1 if data['diabetes'] in ['Yes, diabetic', 'Yes, pre-diabetic/borderline diabetic'] else 0,
        'family_history': 1 if data['family_history'] in ['Yes, in immediate family (parents or siblings)', 
                                                     'Yes, in extended family (grandparents, uncles, aunts)'] else 0
    }
    
    # Classify each component
    analysis = {}
    for key in ['total_chol', 'ldl', 'hdl', 'triglycerides', 'non_hdl', 'vldl']:
        if key in data:
            # Special handling for HDL which needs gender information
            if key == 'hdl':
                # Check if gender is available in data, otherwise provide a default
                sex = data.get('sex', None)  # Assuming gender is stored in data dictionary
                analysis[key] = classify_component(key, data[key], sex=sex)
            else:
                analysis[key] = classify_component(key, data[key])
    
    # Generate ASCVD risk score (simplified for demo)
    ascvd_risk = calculate_ascvd_risk(data)
    
    # Generate recommendations
    recommendations = generate_recommendations(data, analysis, ascvd_risk)
    
    return {
        "classification": analysis,
        "ascvd_risk": ascvd_risk,
        "recommendations": recommendations,
        "ref_ranges": REF_RANGES
    }
    
    

# Optional CLI runner
def main():
    """Main function without UI components"""
    print("🩸 AI Lipid Profile Analyzer")
    print("============================")
    
    # Initialize data dictionary with defaults
    data = {
        "age": int(input("Age: ")),
        "sex": input("Sex (Male/Female): "),
        "smoker": input("Smoker? (Yes/No): "),
        "hypertension": input("Hypertension? (Yes/No): "),
        "diabetes": input("Diabetes? (Yes/No): "),
        "family_history": input("Family history of heart disease? (Yes/No): "),
        "total_chol": int(input("Total Cholesterol: ")),
        "ldl": int(input("LDL: ")),
        "hdl": int(input("HDL: ")),
        "triglycerides": int(input("Triglycerides: "))
    }

    # Manual Input
    print("\nManual Input (press Enter to keep current values)")
    
    # Demographic info
    print("\nDemographic Information:")
    age_input = input(f"Age [{data['age']}]: ")
    if age_input.strip():
        data['age'] = int(age_input)
    
    sex_input = input(f"Sex (Male/Female) [{data['sex']}]: ")
    if sex_input.strip():
        data['sex'] = sex_input
    
    # Risk factors
    print("\nRisk Factors:")
    print("Smoker options: 1) Non-smoker  2) Occasional smoker  3) Regular smoker  4) Heavy smoker")
    smoker_input = input(f"Select smoker status (1-4) [1]: ")
    if smoker_input.strip():
        smoker_options = ["Non-smoker", "Occasional smoker", "Regular smoker", "Heavy smoker"]
        try:
            data['smoker'] = smoker_options[int(smoker_input) - 1]
        except (ValueError, IndexError):
            print("Invalid input, using default: Non-smoker")
            data['smoker'] = "Non-smoker"
    
    hypertension_input = input(f"Hypertension (Yes/No) [{data['hypertension']}]: ")
    if hypertension_input.strip():
        data['hypertension'] = hypertension_input
    
    print("Diabetes options: 1) No  2) Yes, diabetic  3) Yes, pre-diabetic  4) I don't know")
    diabetes_input = input("Select diabetes status (1-4) [1]: ")
    if diabetes_input.strip():
        diabetes_options = ["No", "Yes, diabetic", "Yes, pre-diabetic/borderline diabetic", "I don't know"]
        try:
            data['diabetes'] = diabetes_options[int(diabetes_input) - 1]
        except (ValueError, IndexError):
            print("Invalid input, using default: No")
            data['diabetes'] = "No"
    
    print("Family history options:")
    print("1) No family history")
    print("2) Yes, in immediate family (parents or siblings)")
    print("3) Yes, in extended family (grandparents, uncles, aunts)")
    print("4) Unsure")
    
    fh_input = input("Select family history option (1-4) [1]: ")
    if fh_input.strip():
        fh_options = [
            "No family history",
            "Yes, in immediate family (parents or siblings)",
            "Yes, in extended family (grandparents, uncles, aunts)",
            "Unsure"
        ]
        try:
            data['family_history'] = fh_options[int(fh_input) - 1]
        except (ValueError, IndexError):
            print("Invalid input, using default: No family history")
            data['family_history'] = "No family history"
    
    # Lipid profile values
    print("\nLipid Profile Values:")

    # Lipid inputs with current values displayed
    total_chol_input = input(f"Total Cholesterol (mg/dL) [{data.get('total_chol', '')}]: ")
    if total_chol_input.strip():
        data['total_chol'] = int(total_chol_input)

    ldl_input = input(f"LDL Cholesterol (mg/dL) [{data.get('ldl', '')}]: ")
    if ldl_input.strip():
        data['ldl'] = int(ldl_input)

    hdl_input = input(f"HDL Cholesterol (mg/dL) [{data.get('hdl', '')}]: ")
    if hdl_input.strip():
        data['hdl'] = int(hdl_input)

    triglycerides_input = input(f"Triglycerides (mg/dL) [{data.get('triglycerides', '')}]: ")
    if triglycerides_input.strip():
        data['triglycerides'] = int(triglycerides_input)

    non_hdl_input = input(f"Non-HDL Cholesterol (mg/dL) [{data.get('non_hdl', '')}]: ")
    if non_hdl_input.strip():
        data['non_hdl'] = int(non_hdl_input)

    vldl_input = input(f"VLDL Cholesterol (mg/dL) [{data.get('vldl', '')}]: ")
    if vldl_input.strip():
        data['vldl'] = int(vldl_input)

    # Calculate Non-HDL if not provided but total and HDL are available
    if 'non_hdl' not in data and 'total_chol' in data and 'hdl' in data:
        data['non_hdl'] = data['total_chol'] - data['hdl']

    # Calculate VLDL if not provided but triglycerides are available
    if 'vldl' not in data and 'triglycerides' in data:
        data['vldl'] = data['triglycerides'] // 5  # Common estimation method

    # Wait for user to request analysis
    input("\nPress Enter to analyze your lipid profile...")

    # Analyze the data
    print("\nAnalyzing lipid profile...")
    analysis, ascvd_risk, recommendations = analyze_lipid_profile(data)
    
    # Display results
    print_results(data, analysis, ascvd_risk, recommendations)
    
    # Generate visualization
    viz_input = input("\nView visualization? (yes/no): ")
    if viz_input.lower().startswith('yes'):
        save_viz = input("Save visualization to file? (yes/no): ")
        if save_viz.lower().startswith('yes'):
            save_path = input("Enter file path (default: lipid_profile.png): ") or "lipid_profile.png"
            plot_lipid_profile(data, save_path=save_path)
        else:
            print("Displaying visualization...")
            plot_lipid_profile(data)

if __name__ == "__main__":
    main()