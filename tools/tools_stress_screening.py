import joblib
import pandas as pd
import numpy as np

# --- Burnout/Stress Assessment Section ---

# Likert scale for responses
likert_scale = {
    1: "Never",
    2: "Rarely",
    3: "Sometimes",
    4: "Often",
    5: "Always"
}

# Question set
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

def get_valid_input(prompt, min_val, max_val):
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    print("🧠 Stress and Burnout Assessment")
    print("\nLikert Scale Reference:")
    for num, label in likert_scale.items():
        print(f"{num}: {label}")
    print("\n")

    # Category selection
    categories = list(questions_by_category.keys())
    print("Available categories:", ", ".join(categories))
    selected_categories = input("Enter categories to assess (comma-separated, e.g., work,school): ").lower().split(",")
    selected_categories = [cat.strip() for cat in selected_categories if cat.strip() in categories]

    if not selected_categories:
        print("No valid categories selected. Exiting.")
        return

    category_results = []

    # Stress/Burnout Assessment
    for category in selected_categories:
        print(f"\nAssessment for: {category.capitalize()}")
        responses = []
        questions = questions_by_category[category]

        for i, question in enumerate(questions, 1):
            prompt = f"{category.capitalize()} Q{i}: {question}\nEnter response (1-5): "
            response = get_valid_input(prompt, 1, 5)
            print(f"Selected: {likert_scale[response]}")
            responses.append(response)

        total_score = sum(responses)
        max_score = len(questions) * 5
        percentage = round((total_score / max_score) * 100, 2)
        result = interpret_score(total_score, max_score)

        print(f"\n✅ {category.capitalize()} Assessment Complete!")
        print(f"Total Score: {total_score}")
        print(f"Maximum Score: {max_score}")
        print(f"Percentage: {percentage}%")
        print(f"Interpretation: {result}")

        category_results.append({
            "category": category,
            "total_score": total_score,
            "max_score": max_score,
            "percentage": percentage,
            "interpretation": result
        })

    # Display average if multiple categories assessed
    if len(category_results) > 1:
        show_avg = input("\nWould you like to see the overall average across categories? (yes/no): ").lower() == "yes"
        if show_avg:
            avg_percentage = round(sum(r["percentage"] for r in category_results) / len(category_results), 2)
            print(f"\nAverage Percentage Across {len(category_results)} Categories: {avg_percentage}%")
            if avg_percentage <= 50:
                avg_result = "🟢 Low stress/burnout"
            elif 51 <= avg_percentage <= 70:
                avg_result = "🟡 Moderate stress/burnout"
            else:
                avg_result = "🔴 High stress/burnout – consider seeking support"
            print(f"Average Interpretation: {avg_result}")

    # Mental Health Screening (PHQ-9 & GAD-7)
    show_mental_health = input("\nWould you like to take the Depression and Anxiety Screening (PHQ-9 & GAD-7)? (yes/no): ").lower() == "yes"
    if not show_mental_health:
        print("Exiting program.")
        return

    print("\n--- Depression and Anxiety Screening (PHQ-9 & GAD-7) ---")
    print("About:")
    print("- PHQ-9 assesses depression severity")
    print("- GAD-7 assesses anxiety severity")
    print("- Scores are combined to assess mental health risk")
    print("- All responses are anonymous")

    # Load trained model
    try:
        model = joblib.load(r".\mental_health_risk_predictor.pkl")
    except Exception as e:
        print(f"Could not load mental health model: {e}")
        return

    # Country selection
    countries = ["Argentina", "Australia", "Austria", "Bangladesh", "Belgium", "Brazil", "Canada", "China", "Côte d'Ivoire", "Czech Republic", 
                 "Denmark", "Egypt", "Ethopia", "Finland", "France", "Gambia", "Germany", "Ghana", "Greece", "Hungary", "India", "Ireland", 
                 "Israel", "Italy", "Kenya", "Malawi", "Malaysia", "Mauritius", "Mexico", "Netherlands", "New Zealand", "Nigeria", "Norway", 
                 "Pakistan", "Poland", "Portugal", "Romania", "Russia", "Rwanda", "Seychelles", "Singapore", "South Africa", "South Korea", 
                 "Spain", "Sri Lanka", "Sweden", "Switzerland", "Tanzania", "Thailand", "Turkey", "Uganda", "Ukraine", "United Arab Emirates", 
                 "United Kingdom", "United States"]
    print("\nAvailable countries:", ", ".join(countries))
    country = input("Select your country: ").strip()
    if country not in countries:
        country = "United States"  # Default to US if invalid
        print("Invalid country selected. Defaulting to United States.")

    # Display crisis resources based on country
    crisis_resources = {
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
    print(f"\nCrisis Resource for {country}: {crisis_resources[country]}")

    # Demographic information
    print("\nBasic Information")
    age = get_valid_input("Age (12-120): ", 12, 120)
    print("Gender options: Male, Female, Other/Prefer not to say")
    gender = input("Select your gender: ").strip()
    if gender not in ["Male", "Female", "Other/Prefer not to say"]:
        gender = "Other/Prefer not to say"
        print("Invalid gender selected. Defaulting to Other/Prefer not to say.")
    stress_event = input("Recent stressful life event? (Yes/No): ").strip().lower()
    stress_event = "Yes" if stress_event == "yes" else "No"

    # PHQ-9 Questions
    print("\nPHQ-9 Depression Screening")
    phq_questions = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling/staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself - or that you're a failure",
        "Trouble concentrating on things",
        "Moving/speaking slowly or being fidgety/restless",
        "Thoughts of self-harm or suicide"
    ]

    phq_responses = {}
    for i, question in enumerate(phq_questions, 1):
        print(f"{i}. {question}")
        print("Options: 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day")
        phq_responses[f'phq_q{i}'] = get_valid_input("Enter response (0-3): ", 0, 3)

    # GAD-7 Questions
    print("\nGAD-7 Anxiety Screening")
    gad_questions = [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it's hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen"
    ]

    gad_responses = {}
    for i, question in enumerate(gad_questions, 1):
        print(f"{i}. {question}")
        print("Options: 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day")
        gad_responses[f'gad_q{i}'] = get_valid_input("Enter response (0-3): ", 0, 3)

    # Create input dataframe
    input_data = {
        'age': age,
        'gender': gender,
        'recent_stress_event': 1 if stress_event == "Yes" else 0,
        **phq_responses,
        **gad_responses
    }

    df = pd.DataFrame([input_data])

    # Add missing columns (if any from original training data)
    expected_columns = model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out().tolist() + \
                       ['age', 'recent_stress_event'] + \
                       [f'phq_q{i+1}' for i in range(9)] + \
                       [f'gad_q{i+1}' for i in range(7)]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Prediction and results
    try:
        # Make prediction
        proba = model.predict_proba(df)[0][1]
        prediction = model.predict(df)[0]

        # Display results
        print("\nAssessment Results")
        if prediction == 1:
            print("Our screening suggests you may benefit from professional support")
            print("Please consider reaching out to a mental health professional")
        else:
            print("Our screening suggests lower risk of mental health concerns")
            print("Remember: Regular check-ins on mental health are important for everyone")

        print(f"Risk Probability: {proba*100:.1f}%")
        print(f"PHQ-9 Total Score: {sum(phq_responses.values())}/27")
        print(f"GAD-7 Total Score: {sum(gad_responses.values())}/21")

        # Detailed resources based on country
        print("\nResources:")
        if country == "Argentina":
            print("""
            - **Suicide Prevention Hotline**: 135 (24/7)
            - [Asistencia al Suicida](https://www.asistenciaalsuicida.org.ar)
            - **Hospital Nacional Mental Health**: 0800-345-1435
            - [Mental Health Argentina](https://www.argentina.gob.ar/salud/mental)
            """)
        elif country == "Australia":
            print("""
            - **Lifeline Australia**: 13 11 14
            - [Beyond Blue](https://www.beyondblue.org.au): 1300 22 4636
            - [Kids Helpline](https://www.kidshelpline.com.au): 1800 55 1800
            """)
        elif country == "Austria":
            print("""
            - **Crisis Hotline**: 144 or 112 
            - [Psychosocial Services Austria](https://eu-promens.eu/exchange-visit-austria-1/pages/programme)
            - **Youth Support**: 147 Rat auf Draht
            - [Mental healthcare in Austria](https://www.expatica.com/at/healthcare/healthcare-services/austria-mental-health-109300/)
            """)
        elif country == "Bangladesh":
            print("""
            - **National Helpline**: 09666777222
            - [Mental Health Bangladesh](https://www.dghs.gov.bd)
            - **Kaan Pete Roi**: 09606900100
            - [Moner Bondhu](https://www.monerbondhu.com): 09612444999
            """)
        elif country == "Belgium":
            print("""
            - **Zelfmoordlijn 1813**: 1813
            - [Te Gek!?](https://www.tegek.be): 9000
            - [Awel Youth Line](https://www.awel.be): 102
            """)
        elif country == "Brazil":
            print("""
            - **CVV Suicide Prevention**: 188 (24/7)
            - [Mental Health Brazil](https://www.cvv.org.br)
            - **Psychiatric Emergency**: 190
            """)
        elif country == "Canada":
            print("""
            - **Crisis Services Canada**: 1-833-456-4566
            - [Kids Help Phone](https://kidshelpphone.ca): 1-800-668-6868
            - [Hope for Wellness Helpline](https://www.hopeforwellness.ca): 1-855-242-3310
            """)
        elif country == "China":
            print("""
            - **Beijing Suicide Research Center**: 800-810-1117
            - [Mental Health China](http://www.crisis.org.cn) 
            - **Psychological Support Hotline**: 010-82951332
            - [Lifeline Shanghai](https://www.lifeline-shanghai.com): 400-821-1215
            """)
        elif country == "Côte d'Ivoire":
            print("""
            - [Mental Health Authority Côte d'Ivoire](https://borgenproject.org/mental-health-in-cote-divoire/): (253) 433-7118
            - [National Mental Health Programme](https://reliefweb.int/report/cote-divoire/optimizing-mental-health-care-prayer-camps-cote-divoire): submit@reliefweb.int
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Czech Republic":
            print("""
            - **Crisis Linka**: 116 123
            - [Czech Psychiatric Society](https://www.psychiatrie.cz): +420 773 786 133
            - **Don't Give Up!**: 778 870 344
            - [Online Therapy CZ](https://www.terap.io)
            """)
        elif country == "Denmark":
            print("""
            - **Livslinien**: 70 201 201
            - [PsykiatriFonden](https://www.psykiatrifonden.dk): 39 25 25 25
            - **Børns Vilkår**: 116 111 (Children's Help)
            """)
        elif country == "Egypt":
            print("""
            - [Mental Health Service](https://egyptiansocietyformh.com): contact@egyptiansocietyformh.com
            - [UNHCR](https://help.unhcr.org/egypt/en/health-services/mental-health/): 0220816831 
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Ethopia":
            print("""
            - [Mental Health Service](https://mhsua.org/contact/): +251 945 565656
            - [Ethiopia Community Support And Advocacy Center](https://www.ecsac.org/mentalhealth): (571) 351-6117
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Finland":
            print("""
            - **MIELI Crisis Center**: 09 2525 0111
            - [Mental Health Finland](https://www.mieli.fi)
            - **Children and Youth**: 116 111
            - [Online Therapy Finland](https://mielipalvelut.fi/therapy-in-english-mielipalvelut/?gad_source=1&gad_campaignid=20578186544&gbraid=0AAAAADPTl64ZwpDOHfNKnLxekhgkDAYU5&gclid=Cj0KCQjw0LDBBhCnARIsAMpYlAoFpQmaqBxD-03MXfOJ8tf9dGiOrMk4gGsSIp9tRzp7L60dECPMnoQaAt9TEALw_wcB)
            """)
        elif country == "France":
            print("""
            - **SOS Amitié**: 09 72 39 40 50
            - [La Croix-Rouge Écoute](https://www.croix-rouge.fr): 0 800 858 858
            - [Fil Santé Jeunes](https://www.filsantejeunes.com): 0 800 235 236
            - [Association France Dépression](https://www.france-depression.org)
            """)
        elif country == "Gambia":
            print("""
            - [Mental Health Awareness in Ghana](https://www.my-gambia.com/mymagazine/supportive-activists-foundation-saf/#:~:text=Supportive%20Activist%27s%20Foundation%20is%20a,ill%2Dhealth%20and%20the%20needy.): +220 214 00 00
            - [Mental Health Services in Gambia](https://www.betterplace.org/en/projects/106360-capacity-building-mental-health-services-in-gambia): +49 30 568 38659
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Germany":
            print("""
            - **Emergency Psychological Help**: 0800 111 0 111
            - [German Depression Aid](https://www.deutsche-depressionshilfe.de)
            - [Telefonseelsorge](https://www.telefonseelsorge.de): 0800 111 0 222
            - [Psychotherapeutic Federal Chamber](https://www.bptk.de)
            """)
        elif country == "Ghana":
            print("""
            - [Mental Health Authority Ghana](https://mha-ghana.com): 0800678678
            - [Mental Health Foundation of Ghana](https://www.mhinnovation.net/organisations/mental-health-foundation-ghana)
            - [Care and Action for Mental Health in Africa Ghana](https://www.camha.org)
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Greece":
            print("""
            - **Suicide Help Greece**: 1018
            - [Klimaka NGO Crisis Line](https://www.klimaka.org.gr): 1056
            - **Child Support**: 115 25 (Hellenic Pediatric Association)
            - [Greek Mental Health Society](https://www.psyhelp.gr)
            """)
        elif country == "Hungary":
            print("""
            - **SOS Mental Health**: 06 80 505 505
            - [Hungarian Psychiatric Society](https://www.europsy.net/npa-members/?id=13): 1 2006533 1 3920063
            - **Blue Line Crisis Center**: 06-80-820-111
            - [Online Therapy Hungary](https://www.therapyroute.com/therapists/hungary/1)
            """)
        elif country == "India":
            print("""
            - **Vandrevala Foundation**: 1860 2662 345
            - [iCall Psychosocial Helpline](https://icallhelpline.org): 9152987821
            - [AASRA Crisis Line](https://www.aasra.info): 91-9820466726
            """)
        elif country == "Ireland":
            print("""
            - **Pieta House**: 1800 247 247
            - [Aware Depression Support](https://www.aware.ie): 1800 80 48 48
            - **Samaritans Ireland**: 116 123
            - [Turn2Me Online Therapy](https://www.turn2me.ie)
            """)
        elif country == "Israel":
            print("""
            - **ERAN Emotional First Aid**: 1201
            - [Ministry of Health](https://www.health.gov.il): *2974 from any phone
            - **SAHAR Emotional Support**: 1-800-363-363
            - [Natal Trauma Support](https://www.natal.org.il): 1-800-363-363
            """)
        elif country == "Italy":
            print("""
            - **Telefono Amico**: 02 2327 2327
            - [Samaritans Onlus](https://findahelpline.com/organizations/samaritans-onlus): 06 77208977
            - [La Voce Amica](https://www.lavoceamica.it): 02 873 873
            - [Emergency Psychological Support]: 800 833 833
            """)
        elif country == "Kenya":
            print("""
            - [Suicide Prevention](https://befrienders.org/find-support-now/befrienders-kenya/?country=ke): +254 722 178 177
            - [Mental Health Foundation Helpline](https://mental360.or.ke): +254710360360
            - [Kamili Organization](https://www.kamilimentalhealth.org): +254 (0)700 327 701
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Malawi":
            print("""
            - [Local mental health support](https://mhlec.com/resources/): +265 1 311 690
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Malaysia":
            print("""
            - **Befrienders KL**: 03-76272929
            - [Mental Health Malaysia](https://www.befrienders.org.my)
            - **Ministry of Health**: 03-29359935
            - [Talian Kasih](https://www.jkm.gov.my): 15999 (Domestic violence/abuse)
            """)
        elif country == "Mauritius":
            print("""
            - [Mauritius Mental Health Association](https://www.actogether.mu/find-an-ngo/mauritius-mental-health-association): +230 404 2113
            - [Special Education Needs Authority](https://sena.govmu.org/sena/?page_id=2892): 460 3015
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Mexico":
            print("""
            - **SAPTEL Crisis Line**: 55 5259-8121 (24/7)
            - [Mental Health Mexico](https://www.saptel.org.mx)
            - **UNAM Psychological Support**: 55 5025-0855
            """)
        elif country == "Netherlands":
            print("""
            - **113 Suicide Prevention**: 0900 0113
            - [MIND Korrelatie](https://www.mindkorrelatie.nl): 0900 1450
            - [iPractice Online Therapy](https://www.ipractice.nl)
            - [De Luisterlijn](https://www.deluisterlijn.nl): 0900 0767
            """)
        elif country == "New Zealand":
            print("""
            - **Lifeline Aotearoa**: 0800 543 354
            - [Youthline](https://www.youthline.co.nz): 0800 376 633
            - [Depression Helpline](https://www.depression.org.nz): 0800 111 757
            """)
        elif country == "Nigeria":
            print("""
            - [Nigerian Mental Health] (https://www.nigerianmentalhealth.org): +234 818 659 4160
            - [Mentally Aware Nigeria Initiative (MANI)](https://mentallyaware.org): 08091116264
            - [Suicide Research and Prevention Initiative](https://www.surpinng.com): +234-908-021-7555
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187
            """)
        elif country == "Norway":
            print("""
            - **Mental Helse**: 116 123
            - [Kirkens SOS](https://www.kirkens-sos.no): 22 40 00 40
            - **Children's Help Line**: 116 111
            - [Online Therapy Norway](https://www.psykologportalen.no)
            """)
        elif country == "Pakistan":
            print("""
            - **Umang Helpline**: 0311-7786264
            - [Ministry of NHS](https://www.nhsrc.gov.pk): 1166
            - **Karachi Suicide Prevention**: 021-111-911-911
            """)
        elif country == "Poland":
            print("""
            - **Kryzysowy Telefon Zaufania**: 116 123
            - [ITAKA Foundation](https://www.stopdepresji.pl): 22 654 40 41
            - [Youth Support Line](https://www.liniadzieciom.pl): 116 111
            - [Mental Health Helpline]: 800 702 222
            """)
        elif country == "Portugal":
            print("""
            - **SOS Voz Amiga**: 213 544 545
            - [Portuguese Mental Health & Addictions Services](https://www.uhn.ca/MentalHealth/Clinics/Portuguese_Addiction_Services): 416 603 5974
            - **Conversa Amiga**: 808 237 327
            - [APSI Suicide Prevention](https://www.apsi.org.pt): : 21 884 41 00
            """)
        elif country == "Romania":
            print("""
            - **Telefonul Alb**: 0800 0700 10
            - [ASUR Romanian Psychologists](https://www.asur.ro)
            - **Child Helpline**: 116 111
            - [Mental Health Initiative Supports](https://www.opensocietyfoundations.org/newsroom/mental-health-initiative-supports-monitoring-project-romania-advance-rights-people): +1 212-548-0378
            """)
        elif country == "Russia":
            print("""
            - **Emergency Psychological Help**: 8-800-333-44-34
            - [Mental Health Russia](https://www.psychiatr.ru)
            - [Krizisnaya Liniya](https://www.telefon-doveria.ru): 8-800-2000-122
            """)
        elif country == "Rwanda":
            print("""
            - [MENTAL HEALTH DEPARTMENT](https://www.chub.rw/clinical-service-division/mental-health): +250 789660010
            - [Mental Health Division](https://rbc.gov.rw/who-we-are/our-divisions-and-units/mental-health-division): 114
            - [Emergency Line](https://rbc.gov.rw/who-we-are/our-divisions-and-units/mental-health-division): 912
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Seychelles":
            print("""
            - [Suicide Prevention](https://progress.guide/atlas/africa/seychelles/): +248 432 3535
            - [Mental Health Helpline](https://progress.guide/atlas/africa/seychelles/): +248 438 8000
            - [Emergency Line]: 151
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Singapore":
            print("""
            - **Institute of Mental Health**: 6389-2222 (24h emergency)
            - [SOS Samaritans](https://www.sos.org.sg): 1-767 (24/7)
            - **Silver Ribbon SG**: 6386-1928
            - [HealthHub Mental Wellness](https://www.healthhub.sg)
            """)
        elif country == "South Africa":
            print("""
            - [Suicide Crisis Helpline](https://mha-ghana.com): 0800 567 567
            - [SA Mental Health Foundation](https://www.scan-network.org.za/ngo-listings/sa-mental-health-foundation/): 0828670390
            - [INALA MENTAL HEALTH FOUNDATION](https://www.inala.org.za): Email: hello@inala.org.za
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "South Korea":
            print("""
            - **Suicide Prevention Hotline**: 1577-0199
            - [Korea Mental Health Foundation]((https://www.mentalhealthkorea.org)
            - **Lifeline Korea**: 1588-9191
            - [Seoul Global Center](https://global.seoul.go.kr): 02-2075-4180 (Foreign language support)
            """)
        elif country == "Spain":
            print("""
            - **Teléfono de la Esperanza**: 717 003 717
            - [Cruz Roja Escucha](https://www.cruzroja.es): 900 107 917
            - [ANAR Foundation](https://www.anar.org): 900 20 20 10
            - [Confidential Suicide Hotline]: 914 590 055
            """)
        elif country == "Sri Lanka":
            print("""
            - **Sumithrayo**: 011-2696666
            - [National Institute of Mental Health](https://www.nimh.health.gov.lk): 1926
            - **CCCline**: 1333
            - [Shanthi Maargam](https://www.shanthimaargam.org): 071-7639898
            """)
        elif country == "Sweden":
            print("""
            - **Mind Sverige**: 901 01 (Chat available)
            - [Bris Youth Support](https://www.bris.se): 116 111
            - [Självmordslinjen](https://www.sjalvmordslinjen.se): 901 01
            - [Kry Mental Health Services](https://www.kry.se)
            """)
        elif country == "Switzerland":
            print("""
            - **Die Dargebotene Hand**: 143
            - [Pro Mente Sana](https://www.promentesana.ch): 0848 800 858
            - [SafeZone Online Counseling](https://www.safezone.ch)
            - [Children's Advice Line](https://www.147.ch): 147
            """)
        elif country == "Tanzania":
            print("""
            - [Mwanamke Initiatives Foundation](https://www.mif.or.tz/our-work/program/health-program): +255 623 057 457
            - [TAHMEF](https://www.tahmef.org): +255 692 773 854
            - [Arise International Mental Health Foundation](https://arisementalhealthfoundation.com)
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Thailand":
            print("""
            - **Samaritans of Thailand**: 02-713-6793 (EN/TH)
            - [Department of Mental Health](https://www.dmh.go.th): 1323
            - **Bangkok Mental Health**: 02-026-5905
            - [Sati App](https://www.sati.app) (Digital support)
            """)
        elif country == "Turkey":
            print("""
            - **Psychological Support Line**: 182
            - [Turkish Mental Health Foundation](https://www.ruhsal.org)
            - [Psikolojik Destek Hattı](https://www.psikolog.org.tr): 0850 280 1475
            """)
        elif country == "Uganda":
            print("""
            - [Mental Health in Uganda](https://mhu.ug): 0800212121
            - [Haven Mental Health Foundation](https://www.havenmentalhealthfoundation.org): +256 751902509
            - [Find a Therapist](https://turbomedics.com) : +234 913 106 0187           
            """)
        elif country == "Ukraine":
            print("""
            - **Emergency Mental Health Hotline**: 0 800 100 102 (24/7)
            - [Ukrainian Mental Health Center](https://mentalhealth.org.ua): +38(044)503-87-33
            - **UNICEF Support Line**: 0 800 500 225
            - [Psychological First Aid Ukraine](https://www.learning.foundation/ukraine)
            - **International Red Cross Support**: +380 44 235 1515
            - [WHO Mental Health Resources](https://www.who.int/ukraine)
            """)
        elif country == "United Arab Emirates":
            print("""
            - **Dubai Health Authority**: 800342
            - [Al Amal Hospital Mental Health](https://www.mohap.gov.ae)
            """)
        elif country == "United Kingdom":
            print("""
            - **Samaritans**: 116 123 (24/7)
            - [NHS Mental Health Services](https://www.nhs.uk)
            - [Mind UK](https://www.mind.org.uk): 0300 123 3393
            - [Shout Crisis Text Line]: Text SHOUT to 85258
            """)
        else:
            print("""
            - [National Suicide Prevention Lifeline]((https://suicidepreventionlifeline.org): 1-800-273-8255
            - [Crisis Text Line](https://www.crisistextline.org): Text HOME to 741741
            - [NAMI Helpline](https://www.nami.org): 1-800-950-6264
            - [Find a Therapist](https://www.psychologytoday.com)
            """)

    except Exception as e:
        print(f"Error in processing: {str(e)}")

    print("\n---")
    print("Disclaimer: This tool is not a substitute for professional medical advice.")
    print("Always seek the advice of qualified health providers with any questions you may have regarding medical conditions.")

if __name__ == "__main__":
    main()

__all__ = [
    "questions_by_category",
    "score_burnout_assessment",
    "run_mental_health_model",
    "interpret_score"
]
