# AI Health System 🧠💊

> **Revolutionizing Personalized Healthcare with AI Agents & FastAPI**

---

## 🧩 Project Structure

```bash
AI-Health-System/
│
├── agent_server.py              # FastAPI Server & routing for all AI agents
├── requirements.txt             # Python dependencies
├── README.md                    # Full documentation & setup instructions
│
├── data/
│   ├── user_data.json           # Stores menstrual cycle and lifestyle entries
│   ├── agent_server.json     # Tracks logs for agent server entries
│
├── tools/                       # All AI agent tools and modules
│   ├── tools_reproductive_health.py
│   ├── tools_pregnancy_tracker.py
│   ├── tools_postpartum.py
│   ├── tools_health_score.py
│   ├── tools_kidney_analysis.py
│   ├── tools_liver_analysis.py
│   ├── tools_progress_tracker.py
│   ├── tools_chronic_risk.py
│   ├── tools_weekly_digest.py
│   ├── tools_symptom_checker.py
│   ├── tools_vital_monitor.py
│   ├── tools_hiv_serology.py
│   ├── tools_mental_health.py
│   ├── tools_lab_explainer.py
│   └── tools_device_recommender.py
│
├── ai_chatbot/
│   ├── chatbot_router.py         # Chat interaction layer for Dr. Deuce
│   ├── chatbot_config.json       # Context & persona for Dr. Deuce
│   └── vectorstore_index/        # Vector database index
│
└── wellness_center/             # Agents built for medical specialists
    ├── tools_specialist_monitoring.py
    ├── tools_appointment_scheduler.py
    ├── tools_treatment_recommender.py
    ├── tools_risk_group_monitor.py
    └── tools_patient_summary.py
```

---

## 🚀 Getting Started

📱 Test with Streamlit UI
A visual Streamlit interface is available for local testing!

```bash
# Launch the Streamlit test interface
$ streamlit run agent_app.py

```

```bash
# 1. Clone the Repo
$ git clone https://github.com/McNoblesse/AI-Health-System.git
$ cd AI-Health-System

# 2. Install Requirements
$ pip install -r requirements.txt

# 3. Run Locally
$ uvicorn agent_server:app --reload

# 4. Test via FastAPI Docs
Navigate to: http://localhost:8000/docs
```

---

## 🧠 Core AI Tools for Patients

Each tool is modular and REST-compatible. Here's a summary:

| Agent Tool                     | Description                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------- |
| **Menstrual Cycle Tracker**    | Tracks cycle entries, predicts ovulation & period, gives lifestyle insights. |
| **Pregnancy Monitoring Agent** | Tracks gestation age, delivery date, and symptom-based alerts.               |
| **Postpartum Tracker Agent**   | Assesses mother-baby health & flags anomalies during recovery.               |
| **Health Score System**        | Calculates health index using vitals, symptoms, and penalties for deviation. |
| **Kidney Function Analyzer**   | Analyzes BUN, Creatinine, eGFR etc. with interpretation & warning flags.     |
| **Liver Profile Test**         | Flags hepatitis and liver disease based on LFT biomarkers.                   |
| **Lung Capacity Tool**         | Evaluates FVC/FEV for breathing disorders like COPD.                         |
| **Vitals AI Monitor**          | Real-time health monitor for BP, glucose, ECG, SpO2, temp etc.               |
| **Waist & BMI Insight Agent**  | Tracks obesity/metabolic risk & offers lifestyle adjustments.                |
| **Smart Symptom Checker**      | NLP-powered assistant for analyzing symptoms & suggesting likely conditions. |
| **Mental Health Assessment**   | Uses PHQ-9, GAD-7 metrics to detect depression/anxiety symptoms.             |
| **HIV/Hepatitis Guide**        | Returns test interpretation and supportive next steps.                       |
| **Lab Test Explainer**         | Explains medical tests & their meanings to users in plain language.          |
| **Device Recommender Agent**   | Suggests health devices based on user health conditions or history.          |
| **Progress Tracker**           | Summarizes trends over 30 days using vitals & logs.                          |
| **Lifestyle Chat Coach**       | Gives advice on stress, exercise, hydration & wellness daily.                |
| **Weekly Wellness Digest**     | Auto-generates personalized weekly health insights with goals.               |
| **S.I Unit Converter**         | Converts test values between units for easier interpretation.                |
| **Medical Doc Summarizer**     | Summarizes uploaded health documents and PDF research papers.                |

---

## 🏥 Specialist Wellness Center Tools

These agents provide clinical-level monitoring and summaries for doctors/pharmacists:

| Agent Tool                        | Description                                                      |
| --------------------------------- | ---------------------------------------------------------------- |
| **Patient Health Summary Agent**  | Provides concise health status overview from all logs/tests.     |
| **Treatment Plan Recommender**    | Suggests treatment paths based on symptoms & results.            |
| **Remote Monitoring Alert Agent** | Alerts specialist of any unusual patient vitals or risks.        |
| **Chronic Risk Tracker**          | Tracks long-term disease trends & alerts if risks increase.      |
| **Risk Group Monitor**            | Groups patients by severity/risk for batch management.           |
| **Real-time Health Score Agent**  | Summarizes all current vitals into a single health score.        |
| **Appointment Scheduler Agent**   | Helps decide when and who needs immediate follow-ups.            |
| **Symptom Checker AI**            | Checks symptoms across multiple patients and flags serious ones. |
| **Progress Report Agent**         | Tracks goal progress for nutrition, sleep, stress, and vitals.   |
| **ECG & MRI Image Detector**      | Classifies uploaded images as normal/abnormal for referral.      |

---

## 💬 Dr. Deuce: AI Chatbot System

🤖 `Dr. Deuce` is the system’s smart conversational layer built with:

* 🧠 LLM: Qwen 2.5 (1.5B) via Ollama
* 🔍 RAG Vectorstore (FAISS)
* 🔁 Context management via LangGraph
* ⚙️ Integrated tools: `reproductive`, `health_score`, `vitals`, `digest`, `tracker`, etc.

**Chat Capabilities:**

* Health questions, symptom interpretation
* Cycle tracking and ovulation insight
* Lifestyle improvement tips
* Personalized care guidance

---

## 🧪 Sample cURL Test

```bash
curl -X POST http://localhost:8000/reproductive-agent \
 -H 'Content-Type: application/json' \
 -d '{
   "user_id": "jane_doe_001",
   "mode": "cycle",
   "payload": {
     "start_date": "2025-05-01",
     "period_duration": 5,
     "luteal_phase": 14,
     "stress": "Moderate",
     "exercise": "Light",
     "sleep": "Good",
     "weight_change": "None"
   }
 }'
```

---

## 🔧 Deployment Tips

```bash
# Create GitHub repo & push
$ git init
$ git add .
$ git commit -m "Initial commit with full AI health system"
$ git remote add origin https://github.com/McNoblesse/AI-Health-System.git
$ git push -u origin main

# Host on Render, Railway, or Replit (use uvicorn)
```

---

## 📌 Final Notes

* 🛡️ **Security Tip**: Always anonymize user health data if sharing.
* 📚 Add test samples to `/tests/` for validation
* 🎯 Use this repo to demonstrate applied AI in health tech interviews!

---

> Built with ❤️ and LLMs to empower better health management.

📧 Contact: `nobleindepth@gmail.com`
🌐 Live Demo: *Streamlit aplication / FAST API (Swagger UI)*
