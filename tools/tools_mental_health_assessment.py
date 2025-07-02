import json
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthAssessmentTool:
    """
    Comprehensive Mental Health Assessment Tool that includes:
    - Stress/Burnout Assessment (4 categories: work, school, relationship, medical)
    - PHQ-9 Depression Screening
    - GAD-7 Anxiety Screening
    - ML-based Mental Health Risk Prediction
    - Crisis Resource Recommendations
    """

    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'mental_health_risk_predictor.pkl')
        self.model = None
        self._load_model()

        # Likert scale for stress/burnout assessment
        self.likert_scale = {
            1: "Never",
            2: "Rarely",
            3: "Sometimes",
            4: "Often",
            5: "Always"
        }

        # Question sets for stress/burnout assessment
        self.stress_questions = {
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

        # PHQ-9 Depression Screening Questions
        self.phq9_questions = [
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

        # GAD-7 Anxiety Screening Questions
        self.gad7_questions = [
            "Feeling nervous, anxious, or on edge",
            "Not being able to stop worrying",
            "Worrying too much about different things",
            "Trouble relaxing",
            "Being so restless that it's hard to sit still",
            "Becoming easily annoyed or irritable",
            "Feeling afraid as if something awful might happen"
        ]

        # Crisis resources by country - matching stress_depression2.py format
        self.crisis_resources = {
            "Argentina": {
                "resources": [
                    "**Suicide Prevention Hotline**: 135 (24/7)",
                    "[Asistencia al Suicida](https://www.asistenciaalsuicida.org.ar)",
                    "**Hospital Nacional Mental Health**: 0800-345-1435",
                    "[Mental Health Argentina](https://www.argentina.gob.ar/salud/mental)"
                ]
            },
            "Australia": {
                "resources": [
                    "**Lifeline Australia**: 13 11 14",
                    "[Beyond Blue](https://www.beyondblue.org.au): 1300 22 4636",
                    "[Kids Helpline](https://www.kidshelpline.com.au): 1800 55 1800"
                ]
            },
            "Austria": {
                "resources": [
                    "**Crisis Hotline**: 144 or 112",
                    "[Psychosocial Services Austria](https://eu-promens.eu/exchange-visit-austria-1/pages/programme)",
                    "**Youth Support**: 147 Rat auf Draht",
                    "[Mental healthcare in Austria](https://www.expatica.com/at/healthcare/healthcare-services/austria-mental-health-109300/)"
                ]
            },
            "Bangladesh": {
                "resources": [
                    "**National Helpline**: 09666777222",
                    "[Mental Health Bangladesh](https://www.dghs.gov.bd)",
                    "**Kaan Pete Roi**: 09606900100",
                    "[Moner Bondhu](https://www.monerbondhu.com): 09612444999"
                ]
            },
            "Belgium": {
                "resources": [
                    "**Zelfmoordlijn 1813**: 1813",
                    "[Te Gek!?](https://www.tegek.be): 9000",
                    "[Awel Youth Line](https://www.awel.be): 102"
                ]
            },
            "Brazil": {
                "resources": [
                    "**CVV Suicide Prevention**: 188 (24/7)",
                    "[Mental Health Brazil](https://www.cvv.org.br)",
                    "**Psychiatric Emergency**: 190"
                ]
            },
            "Canada": {
                "resources": [
                    "**Crisis Services Canada**: 1-833-456-4566",
                    "[Kids Help Phone](https://kidshelpphone.ca): 1-800-668-6868",
                    "[Hope for Wellness Helpline](https://www.hopeforwellness.ca): 1-855-242-3310"
                ]
            },
            "China": {
                "resources": [
                    "**Beijing Suicide Research Center**: 800-810-1117",
                    "[Mental Health China](http://www.crisis.org.cn)",
                    "**Psychological Support Hotline**: 010-82951332",
                    "[Lifeline Shanghai](https://www.lifeline-shanghai.com): 400-821-1215"
                ]
            },
            "Côte d'Ivoire": {
                "resources": [
                    "[Mental Health Authority Côte d'Ivoire](https://borgenproject.org/mental-health-in-cote-divoire/): (253) 433-7118",
                    "[National Mental Health Programme](https://reliefweb.int/report/cote-divoire/optimizing-mental-health-care-prayer-camps-cote-divoire): submit@reliefweb.int",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Czech Republic": {
                "resources": [
                    "**Crisis Linka**: 116 123",
                    "[Czech Psychiatric Society](https://www.psychiatrie.cz): +420 773 786 133",
                    "**Don't Give Up!**: 778 870 344",
                    "[Online Therapy CZ](https://www.terap.io)"
                ]
            },
            "Denmark": {
                "resources": [
                    "**Livslinien**: 70 201 201",
                    "[PsykiatriFonden](https://www.psykiatrifonden.dk): 39 25 25 25",
                    "**Børns Vilkår**: 116 111 (Children's Help)"
                ]
            },
            "Egypt": {
                "resources": [
                    "[Mental Health Service](https://egyptiansocietyformh.com): contact@egyptiansocietyformh.com",
                    "[UNHCR](https://help.unhcr.org/egypt/en/health-services/mental-health/): 0220816831",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Ethiopia": {
                "resources": [
                    "[Mental Health Service](https://mhsua.org/contact/): +251 945 565656",
                    "[Ethiopia Community Support And Advocacy Center](https://www.ecsac.org/mentalhealth): (571) 351-6117",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Finland": {
                "resources": [
                    "**MIELI Crisis Center**: 09 2525 0111",
                    "[Mental Health Finland](https://www.mieli.fi)",
                    "**Children and Youth**: 116 111",
                    "[Online Therapy Finland](https://mielipalvelut.fi/therapy-in-english-mielipalvelut/)"
                ]
            },
            "France": {
                "resources": [
                    "**SOS Amitié**: 09 72 39 40 50",
                    "[La Croix-Rouge Écoute](https://www.croix-rouge.fr): 0 800 858 858",
                    "[Fil Santé Jeunes](https://www.filsantejeunes.com): 0 800 235 236",
                    "[Association France Dépression](https://www.france-depression.org)"
                ]
            },
            "Gambia": {
                "resources": [
                    "[Mental Health Awareness in Gambia](https://www.my-gambia.com/mymagazine/supportive-activists-foundation-saf/): +220 214 00 00",
                    "[Mental Health Services in Gambia](https://www.betterplace.org/en/projects/106360-capacity-building-mental-health-services-in-gambia): +49 30 568 38659",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Germany": {
                "resources": [
                    "**Emergency Psychological Help**: 0800 111 0 111",
                    "[German Depression Aid](https://www.deutsche-depressionshilfe.de)",
                    "[Telefonseelsorge](https://www.telefonseelsorge.de): 0800 111 0 222",
                    "[Psychotherapeutic Federal Chamber](https://www.bptk.de)"
                ]
            },
            "Ghana": {
                "resources": [
                    "[Mental Health Authority Ghana](https://mha-ghana.com): 0800678678",
                    "[Mental Health Foundation of Ghana](https://www.mhinnovation.net/organisations/mental-health-foundation-ghana)",
                    "[Care and Action for Mental Health in Africa Ghana](https://www.camha.org)",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Greece": {
                "resources": [
                    "**Suicide Help Greece**: 1018",
                    "[Klimaka NGO Crisis Line](https://www.klimaka.org.gr): 1056",
                    "**Child Support**: 115 25 (Hellenic Pediatric Association)",
                    "[Greek Mental Health Society](https://www.psyhelp.gr)"
                ]
            },
            "Hungary": {
                "resources": [
                    "**SOS Mental Health**: 06 80 505 505",
                    "[Hungarian Psychiatric Society](https://www.europsy.net/npa-members/?id=13): 1 2006533 1 3920063",
                    "**Blue Line Crisis Center**: 06-80-820-111",
                    "[Online Therapy Hungary](https://www.therapyroute.com/therapists/hungary/1)"
                ]
            },
            "India": {
                "resources": [
                    "**Vandrevala Foundation**: 1860 2662 345",
                    "[iCall Psychosocial Helpline](https://icallhelpline.org): 9152987821",
                    "[AASRA Crisis Line](https://www.aasra.info): 91-9820466726"
                ]
            },
            "Ireland": {
                "resources": [
                    "**Pieta House**: 1800 247 247",
                    "[Aware Depression Support](https://www.aware.ie): 1800 80 48 48",
                    "**Samaritans Ireland**: 116 123",
                    "[Turn2Me Online Therapy](https://www.turn2me.ie)"
                ]
            },
            "Israel": {
                "resources": [
                    "**ERAN Emotional First Aid**: 1201",
                    "[Ministry of Health](https://www.health.gov.il): *2974 from any phone",
                    "**SAHAR Emotional Support**: 1-800-363-363",
                    "[Natal Trauma Support](https://www.natal.org.il): 1-800-363-363"
                ]
            },
            "Italy": {
                "resources": [
                    "**Telefono Amico**: 02 2327 2327",
                    "[Samaritans Onlus](https://findahelpline.com/organizations/samaritans-onlus): 06 77208977",
                    "[La Voce Amica](https://www.lavoceamica.it): 02 873 873",
                    "[Emergency Psychological Support]: 800 833 833"
                ]
            },
            "Kenya": {
                "resources": [
                    "[Suicide Prevention](https://befrienders.org/find-support-now/befrienders-kenya/?country=ke): +254 722 178 177",
                    "[Mental Health Foundation Helpline](https://mental360.or.ke): +254710360360",
                    "[Kamili Organization](https://www.kamilimentalhealth.org): +254 (0)700 327 701",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Malawi": {
                "resources": [
                    "[Local mental health support](https://mhlec.com/resources/): +265 1 311 690",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Malaysia": {
                "resources": [
                    "**Befrienders KL**: 03-76272929",
                    "[Mental Health Malaysia](https://www.befrienders.org.my)",
                    "**Ministry of Health**: 03-29359935",
                    "[Talian Kasih](https://www.jkm.gov.my): 15999 (Domestic violence/abuse)"
                ]
            },
            "Mauritius": {
                "resources": [
                    "[Mauritius Mental Health Association](https://www.actogether.mu/find-an-ngo/mauritius-mental-health-association): +230 404 2113",
                    "[Special Education Needs Authority](https://sena.govmu.org/sena/?page_id=2892): 460 3015",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Mexico": {
                "resources": [
                    "**SAPTEL Crisis Line**: 55 5259-8121 (24/7)",
                    "[Mental Health Mexico](https://www.saptel.org.mx)",
                    "**UNAM Psychological Support**: 55 5025-0855"
                ]
            },
            "Netherlands": {
                "resources": [
                    "**113 Suicide Prevention**: 0900 0113",
                    "[MIND Korrelatie](https://www.mindkorrelatie.nl): 0900 1450",
                    "[iPractice Online Therapy](https://www.ipractice.nl)",
                    "[De Luisterlijn](https://www.deluisterlijn.nl): 0900 0767"
                ]
            },
            "New Zealand": {
                "resources": [
                    "**Lifeline Aotearoa**: 0800 543 354",
                    "[Youthline](https://www.youthline.co.nz): 0800 376 633",
                    "[Depression Helpline](https://www.depression.org.nz): 0800 111 757"
                ]
            },
            "Nigeria": {
                "resources": [
                    "[Nigerian Mental Health](https://www.nigerianmentalhealth.org): +234 818 659 4160",
                    "[Mentally Aware Nigeria Initiative (MANI)](https://mentallyaware.org): 08091116264",
                    "[Suicide Research and Prevention Initiative](https://www.surpinng.com): +234-908-021-7555",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Norway": {
                "resources": [
                    "**Mental Helse**: 116 123",
                    "[Kirkens SOS](https://www.kirkens-sos.no): 22 40 00 40",
                    "**Children's Help Line**: 116 111",
                    "[Online Therapy Norway](https://www.psykologportalen.no)"
                ]
            },
            "Pakistan": {
                "resources": [
                    "**Umang Helpline**: 0311-7786264",
                    "[Ministry of NHS](https://www.nhsrc.gov.pk): 1166",
                    "**Karachi Suicide Prevention**: 021-111-911-911"
                ]
            },
            "Poland": {
                "resources": [
                    "**Kryzysowy Telefon Zaufania**: 116 123",
                    "[ITAKA Foundation](https://www.stopdepresji.pl): 22 654 40 41",
                    "[Youth Support Line](https://www.liniadzieciom.pl): 116 111",
                    "[Mental Health Helpline]: 800 702 222"
                ]
            },
            "Portugal": {
                "resources": [
                    "**SOS Voz Amiga**: 213 544 545",
                    "[Portuguese Mental Health & Addictions Services](https://www.uhn.ca/MentalHealth/Clinics/Portuguese_Addiction_Services): 416 603 5974",
                    "**Conversa Amiga**: 808 237 327",
                    "[APSI Suicide Prevention](https://www.apsi.org.pt): 21 884 41 00"
                ]
            },
            "Romania": {
                "resources": [
                    "**Telefonul Alb**: 0800 0700 10",
                    "[ASUR Romanian Psychologists](https://www.asur.ro)",
                    "**Child Helpline**: 116 111",
                    "[Mental Health Initiative Supports](https://www.opensocietyfoundations.org/newsroom/mental-health-initiative-supports-monitoring-project-romania-advance-rights-people): +1 212-548-0378"
                ]
            },
            "Russia": {
                "resources": [
                    "**Emergency Psychological Help**: 8-800-333-44-34",
                    "[Mental Health Russia](https://www.psychiatr.ru)",
                    "[Krizisnaya Liniya](https://www.telefon-doveria.ru): 8-800-2000-122"
                ]
            },
            "Rwanda": {
                "resources": [
                    "[MENTAL HEALTH DEPARTMENT](https://www.chub.rw/clinical-service-division/mental-health): +250 789660010",
                    "[Mental Health Division](https://rbc.gov.rw/who-we-are/our-divisions-and-units/mental-health-division): 114",
                    "[Emergency Line](https://rbc.gov.rw/who-we-are/our-divisions-and-units/mental-health-division): 912",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Seychelles": {
                "resources": [
                    "[Suicide Prevention](https://progress.guide/atlas/africa/seychelles/): +248 432 3535",
                    "[Mental Health Helpline](https://progress.guide/atlas/africa/seychelles/): +248 438 8000",
                    "[Emergency Line]: 151",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Singapore": {
                "resources": [
                    "**Institute of Mental Health**: 6389-2222 (24h emergency)",
                    "[SOS Samaritans](https://www.sos.org.sg): 1-767 (24/7)",
                    "**Silver Ribbon SG**: 6386-1928",
                    "[HealthHub Mental Wellness](https://www.healthhub.sg)"
                ]
            },
            "South Africa": {
                "resources": [
                    "[Suicide Crisis Helpline](https://mha-ghana.com): 0800 567 567",
                    "[SA Mental Health Foundation](https://www.scan-network.org.za/ngo-listings/sa-mental-health-foundation/): 0828670390",
                    "[INALA MENTAL HEALTH FOUNDATION](https://www.inala.org.za): Email: hello@inala.org.za",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "South Korea": {
                "resources": [
                    "**Suicide Prevention Hotline**: 1577-0199",
                    "[Korea Mental Health Foundation](https://www.mentalhealthkorea.org)",
                    "**Lifeline Korea**: 1588-9191",
                    "[Seoul Global Center](https://global.seoul.go.kr): 02-2075-4180 (Foreign language support)"
                ]
            },
            "Spain": {
                "resources": [
                    "**Teléfono de la Esperanza**: 717 003 717",
                    "[Cruz Roja Escucha](https://www.cruzroja.es): 900 107 917",
                    "[ANAR Foundation](https://www.anar.org): 900 20 20 10",
                    "[Confidential Suicide Hotline]: 914 590 055"
                ]
            },
            "Sri Lanka": {
                "resources": [
                    "**Sumithrayo**: 011-2696666",
                    "[National Institute of Mental Health](https://www.nimh.health.gov.lk): 1926",
                    "**CCCline**: 1333",
                    "[Shanthi Maargam](https://www.shanthimaargam.org): 071-7639898"
                ]
            },
            "Sweden": {
                "resources": [
                    "**Mind Sverige**: 901 01 (Chat available)",
                    "[Bris Youth Support](https://www.bris.se): 116 111",
                    "[Självmordslinjen](https://www.sjalvmordslinjen.se): 901 01",
                    "[Kry Mental Health Services](https://www.kry.se)"
                ]
            },
            "Switzerland": {
                "resources": [
                    "**Die Dargebotene Hand**: 143",
                    "[Pro Mente Sana](https://www.promentesana.ch): 0848 800 858",
                    "[SafeZone Online Counseling](https://www.safezone.ch)",
                    "[Children's Advice Line](https://www.147.ch): 147"
                ]
            },
            "Tanzania": {
                "resources": [
                    "[Mwanamke Initiatives Foundation](https://www.mif.or.tz/our-work/program/health-program): +255 623 057 457",
                    "[TAHMEF](https://www.tahmef.org): +255 692 773 854",
                    "[Arise International Mental Health Foundation](https://arisementalhealthfoundation.com)",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Thailand": {
                "resources": [
                    "**Samaritans of Thailand**: 02-713-6793 (EN/TH)",
                    "[Department of Mental Health](https://www.dmh.go.th): 1323",
                    "**Bangkok Mental Health**: 02-026-5905",
                    "[Sati App](https://www.sati.app) (Digital support)"
                ]
            },
            "Turkey": {
                "resources": [
                    "**Psychological Support Line**: 182",
                    "[Turkish Mental Health Foundation](https://www.ruhsal.org)",
                    "[Psikolojik Destek Hattı](https://www.psikolog.org.tr): 0850 280 1475"
                ]
            },
            "Uganda": {
                "resources": [
                    "[Mental Health in Uganda](https://mhu.ug): 0800212121",
                    "[Haven Mental Health Foundation](https://www.havenmentalhealthfoundation.org): +256 751902509",
                    "[Find a Therapist](https://turbomedics.com): +234 913 106 0187"
                ]
            },
            "Ukraine": {
                "resources": [
                    "**Emergency Mental Health Hotline**: 0 800 100 102 (24/7)",
                    "[Ukrainian Mental Health Center](https://mentalhealth.org.ua): +38(044)503-87-33",
                    "**UNICEF Support Line**: 0 800 500 225",
                    "[Psychological First Aid Ukraine](https://www.learning.foundation/ukraine)",
                    "**International Red Cross Support**: +380 44 235 1515",
                    "[WHO Mental Health Resources](https://www.who.int/ukraine)"
                ]
            },
            "United Arab Emirates": {
                "resources": [
                    "**Dubai Health Authority**: 800342",
                    "[Al Amal Hospital Mental Health](https://www.mohap.gov.ae)"
                ]
            },
            "United Kingdom": {
                "resources": [
                    "**Samaritans**: 116 123 (24/7)",
                    "[NHS Mental Health Services](https://www.nhs.uk)",
                    "[Mind UK](https://www.mind.org.uk): 0300 123 3393",
                    "[Shout Crisis Text Line]: Text SHOUT to 85258"
                ]
            },
            "United States": {
                "resources": [
                    "[National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org): 1-800-273-8255",
                    "[Crisis Text Line](https://www.crisistextline.org): Text HOME to 741741",
                    "[NAMI Helpline](https://www.nami.org): 1-800-950-6264",
                    "[Find a Therapist](https://www.psychologytoday.com)"
                ]
            }
        }

    def _load_model(self):
        """Load the trained mental health risk prediction model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Mental health risk prediction model loaded successfully")
            else:
                logger.warning(f"Model file not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None

    def interpret_stress_score(self, score: int, max_score: int) -> str:
        """Interpret stress/burnout score"""
        percentage = (score / max_score) * 100
        if percentage <= 50:
            return "🟢 Low stress/burnout"
        elif 51 <= percentage <= 70:
            return "🟡 Moderate stress/burnout"
        else:
            return "🔴 High stress/burnout – consider seeking support"

    def assess_stress_burnout(self, responses: Dict[str, List[int]]) -> Dict[str, Any]:
        """Assess stress and burnout across multiple categories"""
        results = {}

        for category, category_responses in responses.items():
            if category in self.stress_questions:
                total_score = sum(category_responses)
                max_score = len(category_responses) * 5
                percentage = round((total_score / max_score) * 100, 2)
                interpretation = self.interpret_stress_score(total_score, max_score)

                results[category] = {
                    "total_score": total_score,
                    "max_score": max_score,
                    "percentage": percentage,
                    "interpretation": interpretation
                }

        # Calculate overall average if multiple categories
        if len(results) > 1:
            avg_percentage = round(sum(r["percentage"] for r in results.values()) / len(results), 2)
            if avg_percentage <= 50:
                avg_interpretation = "🟢 Low stress/burnout"
            elif 51 <= avg_percentage <= 70:
                avg_interpretation = "🟡 Moderate stress/burnout"
            else:
                avg_interpretation = "🔴 High stress/burnout – consider seeking support"

            results["overall"] = {
                "average_percentage": avg_percentage,
                "interpretation": avg_interpretation
            }

        return results

    def assess_phq9(self, responses: List[int]) -> Dict[str, Any]:
        """Assess depression using PHQ-9 scale"""
        total_score = sum(responses)

        if total_score <= 4:
            severity = "Minimal depression"
            recommendation = "Monitor symptoms and maintain healthy lifestyle habits"
        elif total_score <= 9:
            severity = "Mild depression"
            recommendation = "Consider lifestyle changes and monitor symptoms closely"
        elif total_score <= 14:
            severity = "Moderate depression"
            recommendation = "Consider professional consultation and treatment options"
        elif total_score <= 19:
            severity = "Moderately severe depression"
            recommendation = "Professional treatment is recommended"
        else:
            severity = "Severe depression"
            recommendation = "Immediate professional intervention is strongly recommended"

        return {
            "total_score": total_score,
            "max_score": 27,
            "severity": severity,
            "recommendation": recommendation
        }

    def assess_gad7(self, responses: List[int]) -> Dict[str, Any]:
        """Assess anxiety using GAD-7 scale"""
        total_score = sum(responses)

        if total_score <= 4:
            severity = "Minimal anxiety"
            recommendation = "Continue current coping strategies"
        elif total_score <= 9:
            severity = "Mild anxiety"
            recommendation = "Consider stress management techniques"
        elif total_score <= 14:
            severity = "Moderate anxiety"
            recommendation = "Consider professional consultation"
        else:
            severity = "Severe anxiety"
            recommendation = "Professional treatment is recommended"

        return {
            "total_score": total_score,
            "max_score": 21,
            "severity": severity,
            "recommendation": recommendation
        }

    def predict_mental_health_risk(self, age: int, gender: str, recent_stress_event: bool,
                                 phq9_responses: List[int], gad7_responses: List[int]) -> Dict[str, Any]:
        """Predict mental health risk using the trained ML model"""
        if self.model is None:
            return {
                "error": "Mental health risk prediction model not available",
                "risk_level": "Unable to assess",
                "confidence": 0.0
            }

        try:
            # Prepare data for prediction
            data = {
                'age': [age],
                'gender': [gender],
                'recent_stress_event': [1 if recent_stress_event else 0]
            }

            # Add PHQ-9 responses
            for i, response in enumerate(phq9_responses, 1):
                data[f'phq_q{i}'] = [response]

            # Add GAD-7 responses
            for i, response in enumerate(gad7_responses, 1):
                data[f'gad_q{i}'] = [response]

            # Create DataFrame
            df = pd.DataFrame(data)

            # Make prediction
            prediction = self.model.predict(df)[0]
            prediction_proba = self.model.predict_proba(df)[0]

            # Interpret results
            risk_level = "High Risk" if prediction == 1 else "Low Risk"
            confidence = max(prediction_proba) * 100

            return {
                "risk_level": risk_level,
                "confidence": round(confidence, 2),
                "prediction": int(prediction),
                "probabilities": {
                    "low_risk": round(prediction_proba[0] * 100, 2),
                    "high_risk": round(prediction_proba[1] * 100, 2)
                }
            }

        except Exception as e:
            logger.error(f"Error in mental health risk prediction: {str(e)}")
            return {
                "error": f"Prediction error: {str(e)}",
                "risk_level": "Unable to assess",
                "confidence": 0.0
            }

    def get_crisis_resources(self, country: str) -> Dict[str, Any]:
        """Get crisis intervention resources for a specific country"""
        if country not in self.crisis_resources:
            # Return a generic message for unsupported countries
            return {
                "resources": [
                    "**International Crisis Support**: Please contact your local emergency services",
                    "**WHO Mental Health Resources**: https://www.who.int/health-topics/mental-health",
                    "**Crisis Text Line Global**: https://www.crisistextline.org/",
                    "**Find Local Support**: Contact your healthcare provider or local mental health services"
                ]
            }
        return self.crisis_resources[country]

    def get_supported_countries(self) -> List[str]:
        """Get list of countries with crisis resource support"""
        return sorted(list(self.crisis_resources.keys()))

    def generate_recommendations(self, stress_results: Dict, phq9_results: Dict,
                               gad7_results: Dict, risk_prediction: Dict) -> List[str]:
        """Generate personalized mental health recommendations"""
        recommendations = []

        # Stress/burnout recommendations
        if "overall" in stress_results:
            avg_percentage = stress_results["overall"]["average_percentage"]
            if avg_percentage > 70:
                recommendations.extend([
                    "🧘‍♀️ Practice daily stress management techniques (meditation, deep breathing)",
                    "⏰ Establish better work-life boundaries and time management",
                    "💤 Prioritize 7-9 hours of quality sleep each night",
                    "🤝 Consider speaking with a mental health professional"
                ])
            elif avg_percentage > 50:
                recommendations.extend([
                    "🌱 Incorporate regular relaxation activities into your routine",
                    "🚶‍♀️ Engage in regular physical exercise or movement",
                    "📱 Consider using stress management apps or tools"
                ])

        # Depression recommendations
        if phq9_results["total_score"] > 9:
            recommendations.extend([
                "☀️ Maintain a regular daily routine and sleep schedule",
                "🏃‍♀️ Engage in regular physical activity, even light exercise",
                "👥 Stay connected with supportive friends and family",
                "🎯 Set small, achievable daily goals"
            ])

        # Anxiety recommendations
        if gad7_results["total_score"] > 9:
            recommendations.extend([
                "🧠 Practice mindfulness and grounding techniques",
                "📝 Try journaling to process worries and thoughts",
                "🎵 Use relaxation techniques like progressive muscle relaxation",
                "⚡ Limit caffeine and alcohol intake"
            ])

        # High-risk recommendations
        if risk_prediction.get("risk_level") == "High Risk":
            recommendations.extend([
                "🚨 Consider scheduling an appointment with a mental health professional",
                "📞 Keep crisis hotline numbers readily available",
                "🛡️ Develop a safety plan with trusted friends or family",
                "💊 Discuss treatment options with a healthcare provider"
            ])

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:8]  # Limit to 8 recommendations

    def generate_follow_up_reminders(self, stress_results: Dict, phq9_results: Dict,
                                   gad7_results: Dict, risk_prediction: Dict) -> List[str]:
        """Generate follow-up reminders based on assessment results"""
        reminders = []

        # High-risk follow-up
        if risk_prediction.get("risk_level") == "High Risk":
            reminders.append("📅 Schedule a mental health professional consultation within 1-2 weeks")
            reminders.append("🔄 Retake this assessment in 2 weeks to monitor progress")

        # Moderate to severe symptoms
        elif (phq9_results["total_score"] > 14 or gad7_results["total_score"] > 14 or
              (stress_results.get("overall", {}).get("average_percentage", 0) > 70)):
            reminders.append("📋 Consider professional consultation within 2-4 weeks")
            reminders.append("📊 Retake this assessment in 3-4 weeks")

        # Mild to moderate symptoms
        elif (phq9_results["total_score"] > 4 or gad7_results["total_score"] > 4 or
              (stress_results.get("overall", {}).get("average_percentage", 0) > 50)):
            reminders.append("🔍 Monitor symptoms and retake assessment in 4-6 weeks")
            reminders.append("📈 Track daily mood and stress levels")

        # General wellness
        else:
            reminders.append("✅ Continue current wellness practices")
            reminders.append("🔄 Consider retaking this assessment in 2-3 months for wellness monitoring")

        return reminders

    def comprehensive_assessment(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive mental health assessment"""
        try:
            # Extract user data
            age = user_data.get("age", 25)
            gender = user_data.get("gender", "Other")
            recent_stress_event = user_data.get("recent_stress_event", False)

            # Assessment responses
            stress_responses = user_data.get("stress_responses", {})
            phq9_responses = user_data.get("phq9_responses", [0] * 9)
            gad7_responses = user_data.get("gad7_responses", [0] * 7)

            # Perform assessments
            stress_results = self.assess_stress_burnout(stress_responses) if stress_responses else {}
            phq9_results = self.assess_phq9(phq9_responses)
            gad7_results = self.assess_gad7(gad7_responses)

            # ML risk prediction
            risk_prediction = self.predict_mental_health_risk(
                age, gender, recent_stress_event, phq9_responses, gad7_responses
            )

            # Generate recommendations and follow-up
            recommendations = self.generate_recommendations(
                stress_results, phq9_results, gad7_results, risk_prediction
            )
            follow_up_reminders = self.generate_follow_up_reminders(
                stress_results, phq9_results, gad7_results, risk_prediction
            )

            # Get crisis resources
            country = user_data.get("country")
            if not country:
                return {
                    "error": "Country is required for crisis resource recommendations",
                    "timestamp": datetime.now().isoformat()
                }
            crisis_resources = self.get_crisis_resources(country)

            # Compile results
            assessment_results = {
                "timestamp": datetime.now().isoformat(),
                "user_info": {
                    "age": age,
                    "gender": gender,
                    "recent_stress_event": recent_stress_event,
                    "country": country
                },
                "assessments": {
                    "stress_burnout": stress_results,
                    "depression_phq9": phq9_results,
                    "anxiety_gad7": gad7_results,
                    "ml_risk_prediction": risk_prediction
                },
                "recommendations": recommendations,
                "follow_up_reminders": follow_up_reminders,
                "crisis_resources": crisis_resources,
                "summary": self._generate_summary(stress_results, phq9_results, gad7_results, risk_prediction)
            }

            return assessment_results

        except Exception as e:
            logger.error(f"Error in comprehensive assessment: {str(e)}")
            return {
                "error": f"Assessment error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _generate_summary(self, stress_results: Dict, phq9_results: Dict,
                         gad7_results: Dict, risk_prediction: Dict) -> str:
        """Generate a summary of the mental health assessment"""
        summary_parts = []

        # Overall risk level
        risk_level = risk_prediction.get("risk_level", "Unable to assess")
        confidence = risk_prediction.get("confidence", 0)
        summary_parts.append(f"🎯 **Overall Mental Health Risk**: {risk_level} (Confidence: {confidence}%)")

        # Depression assessment
        phq9_severity = phq9_results.get("severity", "Unknown")
        phq9_score = phq9_results.get("total_score", 0)
        summary_parts.append(f"😔 **Depression (PHQ-9)**: {phq9_severity} (Score: {phq9_score}/27)")

        # Anxiety assessment
        gad7_severity = gad7_results.get("severity", "Unknown")
        gad7_score = gad7_results.get("total_score", 0)
        summary_parts.append(f"😰 **Anxiety (GAD-7)**: {gad7_severity} (Score: {gad7_score}/21)")

        # Stress/burnout assessment
        if "overall" in stress_results:
            avg_percentage = stress_results["overall"]["average_percentage"]
            interpretation = stress_results["overall"]["interpretation"]
            summary_parts.append(f"🔥 **Stress/Burnout**: {interpretation} ({avg_percentage}%)")

        return "\n".join(summary_parts)
