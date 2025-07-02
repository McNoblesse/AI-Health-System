from typing import Dict, List
import joblib
import pandas as pd

likert_scale = {
    1: "Never",
    2: "Rarely",
    3: "Sometimes",
    4: "Often",
    5: "Always"
}

questions_by_category = {
    "work": [
        "I feel overwhelmed by my job responsibilities.",
        "I struggle to complete tasks due to fatigue or mental exhaustion.",
        "I get fewer than 6 hours of sleep on most workdays.",
        "I rarely take breaks or rest during the workday.",
        "I feel emotionally detached from my work.",
        "I feel recognized and valued at my workplace.",
        "I work beyond 9 hours a day on a regular basis.",
        "I experience physical symptoms such as headaches, fatigue, or insomnia due to work.",
        "I feel like I have a healthy work-life balance.",
        "I enjoy going to work or feel a sense of purpose in my job."
    ],
    "school": [
        "I often feel anxious about deadlines and academic performance.",
        "I struggle to get 7–8 hours of sleep on school nights.",
        "I study or attend schoolwork for more than 8 hours daily.",
        "I feel unable to cope with academic pressure.",
        "I rarely take breaks or engage in non-academic hobbies.",
        "I feel emotionally supported by teachers or school counselors.",
        "I compare myself negatively to other students.",
        "I have trouble focusing and retaining what I study.",
        "I feel burnout from continuous academic demands.",
        "I believe I am managing school and personal life well."
    ],
    "relationship": [
        "I often feel emotionally drained by my relationships.",
        "I find myself avoiding conversations with people close to me.",
        "I feel like my needs are not being acknowledged or understood.",
        "I frequently have conflicts or unresolved tension with loved ones.",
        "I feel pressure to constantly give more than I receive.",
        "I receive emotional support from those close to me.",
        "I often feel lonely even when I am with others.",
        "I feel stressed by trying to maintain harmony in my relationships.",
        "I find joy and peace in my close connections.",
        "I have space to express myself honestly and without judgment."
    ],
    "medical": [
        "I frequently feel tired, even after resting.",
        "My medical condition affects my mood or productivity.",
        "I worry about my health status or future frequently.",
        "I find it hard to manage medication or treatment schedules.",
        "I experience sleep difficulties due to my health issues.",
        "I feel emotionally supported by my healthcare providers.",
        "My health limits my ability to participate in daily activities.",
        "I feel frustrated or helpless about my health condition.",
        "I avoid seeking help even when my symptoms worsen.",
        "I feel in control of my health and wellness decisions."
    ]
}

def interpret_score(score: int, max_score: int) -> str:
    percentage = (score / max_score) * 100
    if percentage <= 50:
        return "🟢 Low stress/burnout"
    elif 51 <= percentage <= 70:
        return "🟡 Moderate stress/burnout"
    else:
        return "🔴 High stress/burnout – consider seeking support"

def score_burnout_assessment(selected_categories: List[str], responses: Dict[str, List[int]]):
    category_results = []

    for category in selected_categories:
        if category not in responses:
            continue
        response_list = responses[category]
        total_score = sum(response_list)
        max_score = len(response_list) * 5
        percentage = round((total_score / max_score) * 100, 2)
        interpretation = interpret_score(total_score, max_score)

        category_results.append({
            "category": category,
            "total_score": total_score,
            "max_score": max_score,
            "percentage": percentage,
            "interpretation": interpretation
        })

    return category_results

def run_mental_health_model(model, phq: Dict[str, int], gad: Dict[str, int], age: int, gender: str, stress_event: bool) -> Dict:
    """Generate mental health risk prediction from PHQ-9 and GAD-7."""
    df = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'recent_stress_event': 1 if stress_event else 0,
        **phq,
        **gad
    }])

    expected_columns = model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out().tolist() + \
                       ['age', 'recent_stress_event'] + \
                       [f'phq_q{i+1}' for i in range(9)] + \
                       [f'gad_q{i+1}' for i in range(7)]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "risk_prediction": "⚠️ High Mental Health Risk" if prediction == 1 else "✅ Low Risk",
        "probability": round(proba * 100, 1),
        "phq_score": sum(phq.values()),
        "gad_score": sum(gad.values())
    }

def get_crisis_resource(country: str) -> str:
    fallback = "https://www.who.int/health-topics/mental-health"
    mental_map = {
        "Argentina": "https://www.asistenciaalsuicida.org.ar",
        "Australia": "https://www.lifeline.org.au",
        "Austria": "https://www.telefonseelsorge.at",
        "Bangladesh": "https://www.shuni.org",
        "Belgium": "https://www.zelfmoord1813.be",
        "Brazil": "https://www.cvv.org.br",
        "Canada": "https://www.crisisservicescanada.ca",
        "China": "https://www.lifeline-shanghai.com",
        "Côte d'Ivoire": "https://borgenproject.org/mental-health-in-cote-divoire/",
        "Czech Republic": "https://www.csspraha.cz",
        "Denmark": "https://www.livslinien.dk",
        "Egypt": "https://help.unhcr.org/egypt/en/health-services/mental-health/",
        "Ethopia": "https://en.peseschkian-stiftung.de/mental-health-project-in-ethiopia",
        "Finland": "https://www.mieli.fi",
        "France": "https://www.expatica.com/fr/healthcare/healthcare-services/mental-healthcare-france-317551/",
        "Gambia": "https://www.gm-nhrc.org/download-file/8b99abcf-d649-11ee-a991-02a8a26af761",
        "Germany": "https://www.deutsche-depressionshilfe.de",
        "Ghana": "https://mha-ghana.com",
        "Greece": "https://www.psyhelp.gr",
        "Hungary": "https://www.sos505.hu",
        "India": "https://www.vandrevalafoundation.com",
        "Ireland": "https://www.pieta.ie",
        "Israel": "https://www.eran.org.il",
        "Italy": "https://www.telefonoamico.it",
        "Kenya": "https://www.mtrh.go.ke/?page_id=288",
        "Malawi": "https://mhlec.com/resources/",
        "Malaysia": "https://www.befrienders.org.my",
        "Mauritius": "https://www.mauritiusmentalhealth.org",
        "Mexico": "https://www.saptel.org.mx",
        "Netherlands": "https://www.113.nl",
        "New Zealand": "https://www.lifeline.org.nz",
        "Nigeria": "https://www.nigerianmentalhealth.org",
        "Norway": "https://www.mentalhelse.no",
        "Pakistan": "https://www.umang.com.pk",
        "Poland": "https://www.116123.pl",
        "Portugal": "https://www.dhi.health.nsw.gov.au/transcultural-mental-health-centre-tmhc/resources/in-your-language/portuguese",
        "Romania": "https://mentalhealthforromania.org/en/",
        "Russia": "https://www.psychiatr.ru",
        "Rwanda": "https://www.pih.org/programs/mental-health",
        "Seychelles": "https://progress.guide/atlas/africa/seychelles/",
        "Singapore": "https://www.sos.org.sg",
        "South Africa": "https://www.safmh.org",
        "South Korea": "https://www.mentalhealthkorea.org",
        "Spain": "https://www.telefonodelaesperanza.org",
        "Sri Lanka": "https://www.sumithrayo.org",
        "Sweden": "https://www.mind.se",
        "Switzerland": "https://www.143.ch",
        "Tanzania": "https://ticc.org/social-programs/mental-health",
        "Thailand": "https://www.samaritansthai.com",
        "Turkey": "https://www.ruhsal.org",
        "Uganda": "https://www.globalhand.org/en/browse/partnering/3/all/organisation/50801",
        "Ukraine": "https://mentalhealth.org.ua",
        "United Arab Emirates": "https://www.mohap.gov.ae",
        "United Kingdom": "https://www.samaritans.org",
        "United States": "https://www.mentalhealth.gov/get-help/immediate-help"
    }
    return mental_map.get(country, fallback)
