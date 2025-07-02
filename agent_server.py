from fastapi import FastAPI, Request
from fastapi import APIRouter, HTTPException
import pdfplumber
from fastapi.responses import JSONResponse
import uvicorn
import ollama
import faiss
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from langchain_ollama import OllamaEmbeddings
import json
import os
import sys
import traceback
import logging
from datetime import datetime, timedelta
import statistics
from dateutil import parser
from fastapi import UploadFile, File, Form
from typing import Optional
from typing import Literal, Dict, Any
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_server.log', encoding='utf-8')
    ]
)

# Add tools path to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))

# Import tools
try:
    from tools.tools_health_score import HealthScoreAnalysisTool
    from tools.tools_monitor_vital_signs import monitor_vital_signs
    from tools.tools_health_data_json import get_default_health_data
    from tools.tools_kidney_function import kidney_function_analysis_tool
    from tools.tools_lipid_profile import analyze_lipid_profile
    from tools.tools_chronic_risk import predict_chronic_risk
    from tools.tools_doc_summarizer import extract_text_from_pdf, summarize_medical_text, extract_text_from_upload, extract_text_from_docx
    from tools.tools_lifestyle_coach import (record_habits, compute_weekly_habit_summary, generate_lifestyle_recommendations)
    from tools.tools_weekly_digest import generate_weekly_digest
    from tools.tools_progress_tracker import (generate_monthly_summary, generate_trend_recommendations)
    from tools.tools_mental_health_assessment import MentalHealthAssessmentTool
    from tools.tools_liver_function import (analyze_liver_function, extract_lft_values, MedicalConditionEnum, SmokingAlcoholEnum, DietaryHabitsEnum, MedicationsEnum, SymptomEnum, HepatitisMarkerEnum, ManualEntryRequest)
    from tools.tools_reproductive_health import run_reproductive_agent


    logging.info("Successfully imported health tools")
except ImportError as e:
    logging.error(f"Failed to import tools: {e}")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(title="Integrated Health Agent API",
              description="API that combines chat, health score analysis, vital signs monitoring, and health consultation")

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unhandled error: {str(exc)}"
    logging.error(f"Exception: {error_msg}")
    logging.error(f"Traceback: {traceback.format_exc()}")
    # Use request parameter to get the path that caused the error
    path = request.url.path if hasattr(request, 'url') else "unknown path"
    logging.error(f"Error occurred at path: {path}")
    return JSONResponse(
        status_code=500,
        content={"error": error_msg}
    )

# === MODEL CONSTANTS ===
QWEN_MODEL = "qwen2.5:1.5b"
DEEPSEEK_MODEL = "deepseek-r1:1.5b"
DEFAULT_MODEL = QWEN_MODEL

# === VECTOR STORE PATHS ===
VECTOR_STORE_PATHS = {
    QWEN_MODEL: {
        "index": r".\Vector_Store\qwen2.5-1.5b\index.faiss",
        "metadata": r".\Vector_Store\qwen2.5-1.5b\index.pkl"
    },
    DEEPSEEK_MODEL: {
        "index": r".\Vector_Store\deepseek-r1-1.5b\index.faiss",
        "metadata": r".\Vector_Store\deepseek-r1-1.5b\index.pkl"
    }
}

# === STORE CHAT TITLES, HISTORIES, AND USER HEALTH DATA ===
chat_titles = {}  # Dictionary to store session titles per user
chat_histories: Dict[str, List[Dict[str, str]]] = {}
user_health_data: Dict[str, Dict[str, Any]] = {}  # Dictionary to store health data by user ID
MAX_HISTORY_LENGTH = 10
TOP_K = 1  # Number of relevant documents to fetch

# === LOAD FAISS INDEXES & METADATA ===
vector_indexes = {}
vector_docs = {}
embedding_models = {}

for model_name, paths in VECTOR_STORE_PATHS.items():
    try:
        vector_indexes[model_name] = faiss.read_index(paths["index"])
        logging.info(f"✅ FAISS index loaded for {model_name}")
    except Exception as e:
        logging.error(f"❌ Error loading FAISS index for {model_name}: {e}")

    try:
        with open(paths["metadata"], "rb") as f:
            vector_docs[model_name] = pickle.load(f)
        logging.info(f"✅ Metadata loaded for {model_name}")
    except Exception as e:
        logging.error(f"❌ Error loading metadata for {model_name}: {e}")

    try:
        embedding_models[model_name] = OllamaEmbeddings(model=model_name)
        logging.info(f"✅ Embedding model loaded for {model_name}")
    except Exception as e:
        logging.error(f"❌ Error loading embedding model {model_name}: {e}")

# === REQUEST MODELS ===
class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    query: str
    model: str = DEFAULT_MODEL

class VitalSignsRequest(BaseModel):
    user_id: str
    vital_signs: Dict[str, Any]

class HealthScoreRequest(BaseModel):
    user_id: str
    health_data: Dict[str, Any]  # Allow any type of value (string, float, etc.)

class KidneyFunctionRequest(BaseModel):
    user_id: str
    kidney_data: Dict[str, Any]  # Accepts both float and string types like "Sex"

class LipidProfileRequest(BaseModel):
    user_id: str
    lipid_data: Dict[str, Any]  # Accepts both string and numeric values

class ProgressTrackRequest(BaseModel):
    user_id: str
    vital_signs: Dict[str, Any]
    
class ChronicRiskRequest(BaseModel):
    user_id: str
    chronic_data: Dict[str, Any]

class UserProfileRequest(BaseModel):
    user_id: str
    profile: Dict[str, Any]

class DocSummaryRequest(BaseModel):
    user_id: str
    model: Optional[str] = DEFAULT_MODEL  # Optional fallback

class LifestyleHabitRequest(BaseModel):
    user_id: str
    habits: Dict[str, float]  # e.g., {"walk": 5000, "water": 6, ...}

class DigestRequest(BaseModel):
    user_id: str
    vital_signs: Dict[str, Any]

class MentalHealthAssessmentRequest(BaseModel):
    user_id: str
    assessment_data: Dict[str, Any]  # Contains age, gender, country, stress responses, PHQ-9, GAD-7, etc.

class LiverFunctionAssessmentRequest(BaseModel):
    user_id: str
    lft_data: ManualEntryRequest

class ReproductiveHealthRequest(BaseModel):
    user_id: str
    mode: Literal["cycle","lifestyle", "pregnancy", "postpartum"]
    payload: Dict[str, Any]



# === HELPER FUNCTIONS ===
def generate_chat_title(first_query: str) -> str:
    """Generate a title by extracting key words from the query."""
    words = first_query.split()[:10]  # Take the first 5 words
    return " ".join(words).title()

def retrieve_context(query: str, model_name: str, top_k: int = TOP_K):
    """Retrieve relevant context from the vector store"""
    if model_name not in vector_indexes or model_name not in vector_docs:
        logging.warning(f"Vector index or docs not found for model {model_name}")
        return ""

    try:
        # Get the embedding model
        embedder = embedding_models[model_name]

        # Generate embedding for the query
        query_embedding = np.array([embedder.embed_query(query)]).astype("float32")

        # Search the vector index
        index = vector_indexes[model_name]
        documents = vector_docs[model_name]

        _, indices = index.search(query_embedding, top_k)

        # Get the relevant documents
        relevant_docs = [
            documents.get(int(idx), {}).get("text", "") if isinstance(documents.get(int(idx), {}), dict)
            else str(documents.get(int(idx), ""))
            for idx in indices[0]
        ]


        return " ".join(relevant_docs)
    except Exception as e:
        logging.error(f"Error retrieving context: {e}")
        logging.error(traceback.format_exc())
        return ""

def add_health_record(user_id: str, vitals: Dict[str, float], user_health_data: Dict):
    entry = {
        "timestamp": datetime.now().isoformat()
    }
    entry.update(vitals)
    user_health_data.setdefault(user_id, []).append(entry)
    

    
def analyze_health_score(health_data: Dict[str, Any]) -> Dict:
    """Analyze health score data"""
    try:
        # Log the input data for debugging
        logging.info(f"Health data received: {json.dumps(health_data)}")

        # Process the health data to ensure it's in the correct format
        processed_data = {}
        for key, value in health_data.items():
            # Handle test results with string values
            if key in ["Malaria", "Hepatitis B", "Widal Test", "Voluntary Serology"]:
                # Ensure "Unknown" is preserved as "Unknown"
                if value == "Unknown" or value is None or value == "" or value == "null":
                    processed_data[key] = "Unknown"
                else:
                    processed_data[key] = value
            # Handle None values
            elif value is None or value == "" or value == "null":
                # Skip null values completely - don't add them to processed_data
                # This ensures they won't be evaluated at all
                continue
            # Convert numeric values to float
            else:
                try:
                    processed_data[key] = float(value)
                except (ValueError, TypeError):
                    # If conversion fails, keep the original value
                    processed_data[key] = value

        logging.info(f"Processed health data: {json.dumps(processed_data)}")

        # Initialize the custom health score tool directly
        logging.info("Initializing CustomHealthScoreAnalysisTool")

        # Create a custom subclass to override the generate_report method
        class CustomHealthScoreAnalysisTool(HealthScoreAnalysisTool):
            def generate_report(self, health_data: dict) -> dict:
                total_score = 0
                max_score = 0
                vitals_needing_improvement = []
                improvement_tips = []

                # We've already filtered out null values in the analyze_health_score function
                # This loop will only process non-null values
                for key, value in health_data.items():

                    # Handle test results properly
                    if key in ["Malaria", "Widal Test", "Hepatitis B", "Voluntary Serology"]:
                        max_score += 5
                        if isinstance(value, str):
                            if value.lower() == "negative":
                                total_score += 5
                            elif value.lower() == "unknown":
                                # Don't count Unknown as needing improvement
                                pass
                            else:
                                vitals_needing_improvement.append(f"{key} (Positive)")
                                improvement_tips.append(f"Seek medical attention for {key}.")
                    # Handle other metrics using the parent class logic
                    elif key == "Weight (BMI)":
                        # This field stores the BMI value, not the weight
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
                    elif key == "Glucose":
                        max_score += 10
                        if isinstance(value, (int, float)):
                            if 70 <= value <= 100:
                                total_score += 10
                            elif 100 < value <= 125:
                                total_score += 5
                                vitals_needing_improvement.append(f"{key} (Moderately High)")
                                improvement_tips.append("Monitor glucose levels and reduce sugar intake.")
                            else:
                                vitals_needing_improvement.append(f"{key} (Abnormal)")
                                improvement_tips.append("Consult a doctor about your glucose levels.")
                    elif key == "SpO2":
                        max_score += 10
                        if isinstance(value, (int, float)):
                            if value >= 95:
                                total_score += 10
                            else:
                                vitals_needing_improvement.append(f"{key} (Low)")
                                improvement_tips.append("Improve oxygen saturation with breathing exercises.")
                    elif key == "Temperature":
                        max_score += 10
                        if isinstance(value, (int, float)):
                            if 36.5 <= value <= 37.5:
                                total_score += 10
                            else:
                                vitals_needing_improvement.append(f"{key} (Abnormal)")
                                improvement_tips.append("Monitor your temperature for any signs of illness.")
                    elif key == "ECG (Heart Rate)":
                        max_score += 10
                        if isinstance(value, (int, float)):
                            if 60 <= value <= 100:
                                total_score += 10
                            else:
                                vitals_needing_improvement.append(f"{key} (Abnormal)")
                                improvement_tips.append("Monitor your heart rate and consult a doctor if irregular.")
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
                    elif key == "Perfusion_index":
                        # Only evaluate if value is not None
                        if value is not None:
                            max_score += 5
                            if isinstance(value, (int, float)) and 0.02 <= value <= 20:
                                total_score += 5
                            else:
                                vitals_needing_improvement.append(f"{key} (Abnormal)")
                                improvement_tips.append("Check perfusion with a professional.")
                    elif key == "Fev":
                        # Only evaluate if value is not None
                        if value is not None:
                            max_score += 5
                            if isinstance(value, (int, float)) and value >= 80:
                                total_score += 5
                            else:
                                vitals_needing_improvement.append(f"{key} (Low)")
                                improvement_tips.append("Improve respiratory function with breathing therapy.")

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
                    "Improvement Tips": ". ".join(improvement_tips) if improvement_tips else "Continue maintaining your current health practices."
                }

        # Use the custom tool to generate the report
        custom_tool = CustomHealthScoreAnalysisTool()
        result = custom_tool.generate_report(processed_data)
        logging.info(f"Health score report generated: {json.dumps(result)}")

        return result
    except Exception as e:
        error_msg = f"Failed to analyze health score: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise Exception(error_msg)


def process_vital_signs(vital_signs: Dict[str, float]) -> Dict:
    """Process vital signs data and return analysis"""
    try:
        # Format data for the tool
        vital_signs_json = json.dumps({"data": vital_signs})
        logging.info(f"Processing vital signs: {vital_signs_json}")

        # Use the vital sign monitoring tool
        result = monitor_vital_signs(vital_signs_json)
        logging.info(f"Vital signs monitoring result: {result}")

        # Check for abnormal patterns
        alerts = []
        severity = "Normal"
        if vital_signs.get("Glucose") and vital_signs["Glucose"] > 140:
            alerts.append("⚠️ High glucose levels detected, Possible hyperglycemia. Consider consulting a doctor.")
            severity = "Critical"
        if vital_signs.get("SpO2") and vital_signs["SpO2"] < 95:
            alerts.append("⚠️ Low SpO2 levels detected, Possible hypoxemia. Ensure proper ventilation.")
            severity = "Critical"
        if vital_signs.get("Heart_Rate") and vital_signs["Heart_Rate"] > 100:
            alerts.append("⚠️ High heart rate detected. Practice stress management.")
            severity = "Caution"
        if vital_signs.get("Temperature") and vital_signs["Temperature"] > 37.5:
            alerts.append("⚠️ Fever detected. Stay hydrated and consult a doctor if it persists.")
            severity = "Caution"

        alert_text = "\n".join(alerts) if alerts else "✅ No abnormal patterns detected."

        return {
            "analysis": result,
            "alerts": alert_text,
            "suggest_consultation": len(alerts) > 0,
            "recommendation": "Please consult your doctor." if severity == "Critical" else "Continue monitoring regularly."
        }
    except Exception as e:
        error_msg = f"Failed to process vital signs: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise Exception(error_msg)


def process_kidney_function(data: Dict[str, float]) -> Dict:
    """Analyze kidney function data using the updated tool"""
    try:
        logging.info(f"Processing kidney function data: {json.dumps(data)}")
        # The updated kidney_function_analysis_tool returns a more comprehensive result
        # that includes analysis, overall_health, confidence_level, missing_parameters, and recommendations
        result = kidney_function_analysis_tool(data)

        # No need to format the output as the tool now returns a well-structured result
        return result
    except Exception as e:
        error_msg = f"Failed to process kidney function: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg}

def process_agent_query(query: str, user_id: str, model_name: str) -> Dict:
    """Process a query through the agent, detecting intent and using appropriate tools"""
    try:
        # Get relevant context from vector store
        context = retrieve_context(query, model_name)
        logging.info(f"Retrieved context: {context[:100]}...")

        # Detect intent from the query
        query_lower = query.lower()

        # Prepare system prompt based on detected intent
        system_prompt = "You are Dr. Deuce, a certified and authorized medical assistant. You assist users by analyzing their health data, monitoring vitals, and providing health consultations. When giving recommendations or health advice, ALWAYS use the user's actual health data if available. Personalize your responses based on their specific health metrics rather than giving generic advice."

        # Add context to the system prompt
        if context:
            system_prompt += f"\n\nRelevant information: {context}"

        # Always include user health data in the system prompt
        health_data_context = ""
        if user_id in user_health_data:
            # Add a header for the health data section
            health_data_context += "\n\nUser's Health Data Summary:\n"

            if "reproductive_health" in user_health_data[user_id]:
                rh = user_health_data[user_id]["reproductive_health"]
                health_data_context += "\n\n🌸 Reproductive Health Summary:\n"
                health_data_context += f"Mode: {rh.get('mode')}\n"
                if rh.get("mode") == "cycle":
                    health_data_context += f"- Next Period: {rh.get('next_prediction', {}).get('Next Period Start')}\n"
                    health_data_context += f"- Ovulation Window: {rh.get('next_prediction', {}).get('Ovulation Window')}\n"
                    health_data_context += "Recommendations:\n"
                    for rec in rh.get("recommendations", []):
                        health_data_context += f"  • {rec}\n"
                elif rh.get("mode") == "pregnancy":
                    health_data_context += f"- Gestational Age: {rh.get('gestational_weeks')}\n"
                    health_data_context += f"- Expected Delivery: {rh.get('expected_delivery')}\n"
                    health_data_context += "Diagnosis:\n"
                    for diag in rh.get("diagnosis", []):
                        health_data_context += f"  • {diag}\n"
                    health_data_context += "Recommendations:\n"
                    for rec in rh.get("recommendations", []):
                        health_data_context += f"  • {rec}\n"
                elif rh.get("mode") == "postpartum":
                    health_data_context += f"- Days Since Delivery: {rh.get('days_since_delivery')}\n"
                    health_data_context += f"- Ovulation Info: {rh.get('ovulation_info')}\n"
                    health_data_context += "Flags:\n"
                    for flag in rh.get("flags", []):
                        health_data_context += f"  • {flag}\n"
                    health_data_context += "Recommendations:\n"
                    for rec in rh.get("recommendations", []):
                        health_data_context += f"  • {rec}\n"


            # Add vital signs data if available
            if "vital_signs" in user_health_data[user_id]:
                vital_data = user_health_data[user_id]["vital_signs"]
                health_data_context += "\n📊 Vital Signs:\n"
                health_data_context += f"Test date: {vital_data['timestamp'][:10]}\n"

                # Add vital signs
                for key, value in vital_data['data'].items():
                    health_data_context += f"- {key}: {value}\n"

                # Add alerts if any
                if "alerts" in vital_data['result'] and vital_data['result']['alerts']:
                    health_data_context += f"Alerts: {vital_data['result']['alerts']}\n"

            # Add health score data if available
            if "health_score" in user_health_data[user_id]:
                score_data = user_health_data[user_id]["health_score"]
                health_data_context += "\n🏆 Health Score:\n"
                health_data_context += f"Test date: {score_data['timestamp'][:10]}\n"
                health_data_context += f"Total Score: {score_data['result'].get('Total Score', 'Unknown')}\n"
                health_data_context += f"Health Status: {score_data['result'].get('Health Status', 'Unknown')}\n"

                # Add vitals needing improvement
                if "Vitals Needing Improvement" in score_data['result']:
                    health_data_context += f"Vitals Needing Improvement: {score_data['result']['Vitals Needing Improvement']}\n"

                # Add improvement tips
                if "Improvement Tips" in score_data['result']:
                    health_data_context += f"Improvement Tips: {score_data['result']['Improvement Tips']}\n"

            # Add kidney function data if available
            if "kidney_function" in user_health_data[user_id]:
                kidney_data = user_health_data[user_id]["kidney_function"]
                health_data_context += "\n🧪 Kidney Function:\n"
                health_data_context += f"Test date: {kidney_data['timestamp'][:10]}\n"
                health_data_context += f"Overall health: {kidney_data['result'].get('overall_health', 'Unknown')}\n"
                health_data_context += f"Confidence level: {kidney_data['result'].get('confidence_level', 'Unknown')}\n"

                # Add analysis items
                analysis_items = kidney_data['result'].get('analysis', [])
                if analysis_items:
                    health_data_context += "Analysis:\n"
                    if isinstance(analysis_items, list):
                        for item in analysis_items:
                            health_data_context += f"- {item}\n"
                    else:
                        health_data_context += f"{analysis_items}\n"

                # Add recommendations if available
                if "recommendations" in kidney_data['result'] and kidney_data['result']['recommendations']:
                    health_data_context += "Recommendations:\n"
                    for rec in kidney_data['result']['recommendations']:
                        health_data_context += f"- {rec}\n"

            # Add lipid profile data if available
            if "lipid_profile" in user_health_data[user_id]:
                lipid_data = user_health_data[user_id]["lipid_profile"]
                health_data_context += "\n💉 Lipid Profile:\n"
                health_data_context += f"Test date: {lipid_data['timestamp'][:10]}\n"

                # Add classification
                classification = lipid_data['result'].get('classification', {})
                if classification:
                    health_data_context += "Classification:\n"
                    for component, level in classification.items():
                        health_data_context += f"- {component.replace('_', ' ').title()}: {level.title()}\n"

                # Add risk assessment
                health_data_context += f"ASCVD Risk: {lipid_data['result'].get('ascvd_risk', 'Unknown')}\n"

                # Add recommendations if available
                if "recommendations" in lipid_data['result'] and lipid_data['result']['recommendations']:
                    health_data_context += "Recommendations:\n"
                    for rec in lipid_data['result']['recommendations']:
                        health_data_context += f"- {rec}\n"

            # Add instructions for the agent to use this data
            health_data_context += "\nIMPORTANT: Always use the above health data when providing recommendations or answering health-related questions. Personalize your responses based on this data."

        # Add health data context to system prompt if available
        if health_data_context:
            system_prompt += f"\n\nUser's health data for reference:{health_data_context}"
            logging.info(f"Added health data context to system prompt for user {user_id}")

        # Initialize chat history if first interaction
        if user_id not in chat_histories:
            chat_histories[user_id] = [{"role": "system", "content": system_prompt}]
            chat_titles[user_id] = generate_chat_title(query)
        else:
            # Update system prompt with latest health data context
            chat_histories[user_id][0]["content"] = system_prompt

        # Add user query to history
        chat_histories[user_id].append({"role": "user", "content": query})

        # Keep chat history within limit
        if len(chat_histories[user_id]) > MAX_HISTORY_LENGTH:
            # Keep the system message and the most recent messages
            system_message = chat_histories[user_id][0]
            chat_histories[user_id] = [system_message] + chat_histories[user_id][-(MAX_HISTORY_LENGTH-1):]

        # Get response from Ollama
        response = ollama.chat(model=model_name, messages=chat_histories[user_id])
        model_response = response["message"]["content"]

        # Check for health-related intents
        tool_response = ""
        tools_used = []

        # Recommendation intent - handle this specially to ensure personalized recommendations
        if any(keyword in query_lower for keyword in ["recommendation", "advice", "suggest", "tips", "what should i do"]):
            # Check if user has any health data
            if user_id in user_health_data and user_health_data[user_id]:
                # Create a personalized recommendation response
                recommendation_response = "Based on your health data, here are my personalized recommendations:\n\n"

                # Add recommendations from health score if available
                if "health_score" in user_health_data[user_id]:
                    score_data = user_health_data[user_id]["health_score"]
                    if "Improvement Tips" in score_data["result"] and score_data["result"]["Improvement Tips"]:
                        recommendation_response += "**Health Score Recommendations:**\n"
                        tips = score_data["result"]["Improvement Tips"].split(". ")
                        for tip in tips:
                            if tip.strip():
                                recommendation_response += f"- {tip.strip()}.\n"
                        recommendation_response += "\n"

                # Add recommendations from kidney function if available
                if "kidney_function" in user_health_data[user_id]:
                    kidney_data = user_health_data[user_id]["kidney_function"]
                    if "recommendations" in kidney_data["result"] and kidney_data["result"]["recommendations"]:
                        recommendation_response += "**Kidney Function Recommendations:**\n"
                        for rec in kidney_data["result"]["recommendations"]:
                            recommendation_response += f"- {rec}\n"
                        recommendation_response += "\n"

                # Add recommendations from lipid profile if available
                if "lipid_profile" in user_health_data[user_id]:
                    lipid_data = user_health_data[user_id]["lipid_profile"]
                    if "recommendations" in lipid_data["result"] and lipid_data["result"]["recommendations"]:
                        recommendation_response += "**Lipid Profile Recommendations:**\n"
                        for rec in lipid_data["result"]["recommendations"]:
                            recommendation_response += f"- {rec}\n"
                        recommendation_response += "\n"

                # Add alerts from vital signs if available
                if "vital_signs" in user_health_data[user_id]:
                    vital_data = user_health_data[user_id]["vital_signs"]
                    if "alerts" in vital_data["result"] and vital_data["result"]["alerts"] and vital_data["result"]["alerts"] != "No abnormal patterns detected.":
                        recommendation_response += "**Vital Signs Recommendations:**\n"
                        alerts = vital_data["result"]["alerts"].split("\n")
                        for alert in alerts:
                            if alert.strip():
                                recommendation_response += f"- {alert.strip()}\n"
                        recommendation_response += "\n"

                # Override the model response with our personalized recommendations
                model_response = recommendation_response
                tools_used.append("personalized_recommendations")
            else:
                # If no health data, prompt user to enter some
                tool_response += "\n\nI don't have any health data for you yet. Would you like to enter your health data for personalized recommendations? You can choose from:\n\n- Health Score Analysis\n- Vital Signs Monitoring\n- Kidney Function Test\n- Lipid Profile Test\n\nType the name of the test you'd like to perform."
                tools_used.append("no_health_data")

        # Health score analysis intent
        elif any(keyword in query_lower for keyword in ["health score", "analyze my health", "health analysis"]):
            tool_response += "\n\nWould you like to analyze your health score? Type 'yes' to begin."
            tools_used.append("health_score_intent")

        # Vital signs monitoring intent
        elif any(keyword in query_lower for keyword in ["vital signs", "monitor vitals", "check vitals"]):
            tool_response += "\n\nWould you like to enter your vital signs for monitoring? Type 'yes' to begin."
            tools_used.append("vital_signs_intent")

        # Kidney function test intent
        elif any(keyword in query_lower for keyword in ["kidney function", "kidney test", "renal function"]):
            tool_response += "\n\nWould you like to analyze your kidney function? Type 'yes' to begin."
            tools_used.append("kidney_function_intent")

        # Lipid profile test intent
        elif any(keyword in query_lower for keyword in ["lipid profile", "cholesterol test", "lipid test"]):
            tool_response += "\n\nWould you like to analyze your lipid profile? Type 'yes' to begin."
            tools_used.append("lipid_profile_intent")

        # Health consultation intent
        elif any(keyword in query_lower for keyword in ["health consultation", "consult", "medical advice"]):
            tool_response += "\n\nWould you like to start a health consultation? Type 'yes' to begin."
            tools_used.append("consultation_intent")

        # Add tool response to model response if any
        if tool_response:
            model_response += tool_response

        # Add response to chat history
        chat_histories[user_id].append({"role": "assistant", "content": model_response})

        return {
            "response": model_response,
            "chat_title": chat_titles[user_id],
            "chat_history": chat_histories[user_id],
            "tools_used": tools_used
        }
    except Exception as e:
        error_msg = f"Failed to process agent query: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise Exception(error_msg)

# === ENDPOINTS ===
@app.get("/")
async def root():
    return {"message": "Integrated Health Agent API is running"}

@app.get("/health")
async def health_check():
    """Simple health check endpoint for monitoring"""
    return {"status": "healthy"}



@app.get("/default-health-data")
async def default_health_data():
    """Get default health data"""
    try:
        # Use the exact DEFAULT_HEALTH_DATA structure as specified
        default_data = {
            "Glucose": None,
            "SpO2": None,
            "ECG (Heart Rate)": None,
            "Blood Pressure (Systolic)": None,
            "Blood Pressure (Diastolic)": None,
            "Weight (BMI)": None,  # This field stores the BMI value, not the weight
            "Temperature": None,
            "Malaria": "Unknown",
            "Widal Test": "Unknown",
            "Hepatitis B": "Unknown",
            "Voluntary Serology": "Unknown",
            "Perfusion_index": None,
            "Waist Circumference": None,
            "Fev": None
        }

        logging.info(f"Returning default health data: {json.dumps(default_data)}")
        return default_data
    except Exception as e:
        error_msg = f"Error getting default health data: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg}

@app.get("/user-health-data/{user_id}")
async def get_user_health_data(user_id: str, data_type: Optional[str] = None):
    """Get stored health data for a specific user

    Args:
        user_id: The ID of the user
        data_type: Optional type of health data to retrieve (vital_signs, health_score, kidney_function, lipid_profile)
                  If not provided, returns all health data for the user
    """
    try:
        if user_id not in user_health_data:
            return {"message": "No health data found for this user"}

        if data_type:
            if data_type not in user_health_data[user_id]:
                return {"message": f"No {data_type} data found for this user"}
            return user_health_data[user_id][data_type]

        return user_health_data[user_id]
    except Exception as e:
        error_msg = f"Error retrieving user health data: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg}

@app.get("/status")
async def status():
    """Check if the server is running and models are loaded"""
    models_status = {}
    for model_name in [QWEN_MODEL, DEEPSEEK_MODEL]:
        models_status[model_name] = {
            "vector_index": model_name in vector_indexes,
            "metadata": model_name in vector_docs,
            "embeddings": model_name in embedding_models
        }

    return {
        "status": "running",
        "models": models_status
    }

@app.get("/default-health-data")
async def get_default_health_data_endpoint():
    """Get default health data template"""
    try:
        # Get the default health data
        default_data = get_default_health_data()

        # Extract just the data part without the wrapper
        if "data" in default_data:
            health_data = default_data["data"]
        else:
            health_data = default_data

        # Remove score fields for the form
        clean_data = {k: v for k, v in health_data.items() if not k.endswith("_Score") and not k.endswith("_Unit")}

        return clean_data
    except Exception as e:
        error_msg = f"Error getting default health data: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg}

from fastapi import Query

@app.get("/chat-history")
async def get_chat_history(user_id: str = Query(...), session_id: str = Query(...)):
    """Return chat history and health data for the user"""
    try:
        # Get messages for this session
        messages = chat_histories.get(user_id, [])

        # Get health data (vital signs, health score, etc.)
        health_data = user_health_data.get(user_id, {})

        return {
            "messages": messages,
            "health_data": health_data
        }
    except Exception as e:
        error_msg = f"Error fetching chat history: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}


@app.post("/query")
async def get_response(chat: ChatRequest):
    """Handle chat queries through the agent (Dr. Deuce)"""
    try:
        user_id, query, model = chat.user_id, chat.query, chat.model

        # Validate model selection
        if model not in [QWEN_MODEL, DEEPSEEK_MODEL]:
            return {"error": f"Invalid model selection. Choose either {QWEN_MODEL} or {DEEPSEEK_MODEL}."}

        # Check if query is follow-up to offer_chat prompt
        is_follow_up = query.strip().lower() in ["yes", "yes please", "sure", "okay", "yeah", "ok", "i want to chat"]

        if is_follow_up:
            context_data = user_health_data.get(user_id, {})
            if not context_data:
                return {"error": "No prior health data found. Please complete an assessment first."}

            formatted_context = json.dumps(context_data, indent=2)
            query = f"I would like deeper insight based on my recent reproductive health data.\n\nHere is my data:\n{formatted_context}"

        # Process the query through the Dr. Deuce agent
        result = process_agent_query(query, user_id, model)

        return result

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg}


@app.post("/vital-signs")
async def process_vital_signs_endpoint(request: VitalSignsRequest):
    """Process vital signs data"""
    try:
        user_id = request.user_id
        vital_signs = request.vital_signs

        # Process the vital signs
        result = process_vital_signs(vital_signs)

        # Save vital signs data to user health data store
        if user_id not in user_health_data:
            user_health_data[user_id] = {}
        user_health_data[user_id]["vital_signs"] = {
            "data": vital_signs,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        logging.info(f"Saved vital signs data for user {user_id}")

        # Update chat history if user exists
        if user_id in chat_histories:
            vital_signs_str = ", ".join([f"{k}: {v}" for k, v in vital_signs.items()])
            chat_histories[user_id].append({"role": "user", "content": f"My vital signs are: {vital_signs_str}"})

            response_content = f"I've analyzed your vital signs.\n\n{result.get('analysis', '')}"
            if result.get('alerts'):
                response_content += f"\n\nAlerts: {result['alerts']}"

            chat_histories[user_id].append({"role": "assistant", "content": response_content})

        return result
    except Exception as e:
        error_msg = f"Error processing vital signs: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg}

@app.post("/health-score")
async def analyze_health_score_endpoint(request: HealthScoreRequest):
    """Analyze health score data"""
    try:
        user_id = request.user_id
        health_data = request.health_data

        # Validate health data
        if not health_data:
            error_msg = "No health data provided"
            logging.error(error_msg)
            return {"error": error_msg}

        # Log the original health data
        logging.info(f"Original health data: {json.dumps(health_data)}")

        # Analyze the health score
        result = analyze_health_score(health_data)

        # Save health data to user health data store
        if user_id not in user_health_data:
            user_health_data[user_id] = {}
        user_health_data[user_id]["health_score"] = {
            "data": health_data,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        logging.info(f"Saved health score data for user {user_id}")

        # Format the result for display
        formatted_result = "\n".join([f"**{k}**: {v}" for k, v in result.items()])

        # Update chat history if user exists
        if user_id in chat_histories:
            chat_histories[user_id].append({"role": "user", "content": "Please analyze my health score."})

            response_content = f"I've analyzed your health score.\n\n{formatted_result}"
            chat_histories[user_id].append({"role": "assistant", "content": response_content})

        return {
            "analysis": formatted_result,
            "score": result.get("Total Score", 0),
            "category": result.get("Health Status", "Unknown")
        }
    except Exception as e:
        error_msg = f"Error analyzing health score: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg}

@app.post("/kidney-function")
async def analyze_kidney_function_endpoint(request: KidneyFunctionRequest):
    """Analyze kidney function data using the updated tool"""
    try:
        user_id = request.user_id
        kidney_data = request.kidney_data

        if not kidney_data:
            error_msg = "No kidney data provided"
            logging.error(error_msg)
            return {"error": error_msg}

        # Run analysis tool - the updated tool returns a comprehensive result
        result = kidney_function_analysis_tool(kidney_data)

        # Save kidney data to user health data store
        if user_id not in user_health_data:
            user_health_data[user_id] = {}
        user_health_data[user_id]["kidney_function"] = {
            "data": kidney_data,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        logging.info(f"Saved kidney function data for user {user_id}")

        # Format for chat memory (optional)
        if user_id in chat_histories:
            chat_histories[user_id].append({"role": "user", "content": "Please analyze my kidney function."})

            # Create a summary for the chat history
            analysis_items = result.get("analysis", [])
            formatted_analysis = "\n".join([f"- {item}" for item in analysis_items]) if isinstance(analysis_items, list) else analysis_items

            summary = f"🧪 Kidney Function Analysis\n\n"
            if formatted_analysis:
                summary += f"{formatted_analysis}\n\n"

            summary += f"🩺 Overall Health: {result.get('overall_health', 'Unknown')}\n"
            summary += f"📊 Confidence Level: {result.get('confidence_level', 'Unknown')}"

            if result.get("missing_parameters"):
                summary += f"\n🔍 Missing Parameters: {', '.join(result['missing_parameters'])}"

            # Add recommendations if available
            if result.get("recommendations"):
                summary += "\n\n🔸 Recommendations:\n"
                for rec in result["recommendations"]:
                    summary += f"- {rec}\n"

            chat_histories[user_id].append({"role": "assistant", "content": summary})

        return result

    except Exception as e:
        error_msg = f"Error analyzing kidney function: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg}

@app.post("/lipid-profile")
async def analyze_lipid_profile_endpoint(request: LipidProfileRequest):
    """Analyze lipid profile data"""
    try:
        user_id = request.user_id
        lipid_data = request.lipid_data

        if not lipid_data:
            return {"error": "No lipid data provided."}

        result = analyze_lipid_profile(lipid_data)

        # Save lipid data to user health data store
        if user_id not in user_health_data:
            user_health_data[user_id] = {}
        user_health_data[user_id]["lipid_profile"] = {
            "data": lipid_data,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        logging.info(f"Saved lipid profile data for user {user_id}")

        # Add formatted text for chat history (if integrated)
        if user_id in chat_histories:
            chat_histories[user_id].append({
                "role": "user",
                "content": "Please analyze my lipid profile."
            })

            summary = "\n".join([f"**{k.title().replace('_',' ')}**: {v}" for k, v in result["classification"].items()])
            summary += f"\n\n**ASCVD Risk Level**: {result['ascvd_risk']}"
            summary += "\n\n**Recommendations:**\n- " + "\n- ".join(result["recommendations"])

            chat_histories[user_id].append({
                "role": "assistant",
                "content": summary
            })

        return result
    except Exception as e:
        error_msg = f"Error analyzing lipid profile: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}

@app.post("/track-progress")
async def track_vital_progress(request: ProgressTrackRequest):
    try:
        user_id = request.user_id
        data = request.vital_signs
        timestamp = datetime.now().isoformat()

        if user_id not in user_health_data:
            user_health_data[user_id] = []

        user_health_data[user_id].append({**data, "timestamp": timestamp})
        logging.info(f"Updated progress history for user: {user_id}")

        summary = generate_monthly_summary(user_id, user_health_data)
        recommendations = generate_trend_recommendations(summary.get("trend_analysis", {}))

        return {
            "active_vitals": data,
            "monthly_summary": summary,
            "recommendations": recommendations,
            "raw_data_points": len(user_health_data[user_id])
        }

    except Exception as e:
        logging.error(f"Error tracking vitals for {request.user_id}: {str(e)}")
        return {"error": str(e)}

        
@app.post("/chronic-risk")
async def chronic_risk_endpoint(request: ChronicRiskRequest):
    try:
        user_id = request.user_id
        data = request.chronic_data

        # Run risk prediction
        result = predict_chronic_risk(data)

        # Save to health data
        if user_id not in user_health_data:
            user_health_data[user_id] = {}
        user_health_data[user_id]["chronic_risk"] = {
            "input": data,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        # Log chat interaction if enabled
        if user_id in chat_histories:
            chat_histories[user_id].append({
                "role": "user",
                "content": f"My chronic risk inputs are: {json.dumps(data)}"
            })
            chat_histories[user_id].append({
                "role": "assistant",
                "content": f"Here is your chronic disease risk prediction:\n{json.dumps(result, indent=2)}"
            })

        return result
    except Exception as e:
        logging.error(f"Chronic risk analysis failed for {request.user_id}: {str(e)}")
        return {"error": str(e)}

@app.post("/summarize-medical-doc")
async def summarize_medical_doc(
    user_id: str = Form(...),
    model: Optional[str] = Form("qwen2.5:1.5b"),  # Optional override
    file: UploadFile = File(...)
):
    try:
        raw_text = extract_text_from_upload(file)

        if len(raw_text) < 100:
            return {"error": "Unable to extract enough readable text. Please try another document."}

        summary = summarize_medical_text(raw_text, model=model)

        # Store in chat history
        if user_id in chat_histories:
            chat_histories[user_id].append({"role": "user", "content": f"I've uploaded a medical document."})
            chat_histories[user_id].append({"role": "assistant", "content": summary})

        return {
            "summary": summary,
            "length": len(raw_text),
            "success": True
        }

    except Exception as e:
        logging.error(f"Error summarizing doc: {e}")
        return {"error": str(e)}


@app.post("/track-lifestyle")
async def track_lifestyle_habits(request: LifestyleHabitRequest):
    try:
        user_id = request.user_id
        habits = request.habits

        record_habits(user_id, habits)
        summary = compute_weekly_habit_summary(user_id)
        recommendations = generate_lifestyle_recommendations(summary)

        # Update chat history
        if user_id in chat_histories:
            chat_histories[user_id].append({"role": "user", "content": f"My lifestyle habits are: {habits}"})
            chat_histories[user_id].append({"role": "assistant", "content": "I've logged your habits and generated a weekly summary."})

        return {
            "weekly_summary": summary,
            "recommendations": recommendations
        }

    except Exception as e:
        logging.error(f"Error processing lifestyle habits: {str(e)}")
        return {"error": str(e)}

@app.post("/weekly-digest")
async def summarize_weekly_vitals(request: DigestRequest):
    try:
        user_id = request.user_id
        data = request.vital_signs
        timestamp = datetime.now().isoformat()

        # Add the new vitals to the user's history
        if user_id not in user_health_data:
            user_health_data[user_id] = []

        user_health_data[user_id].append({**data, "timestamp": timestamp})
        
        # Call digest generator tool
        digest = generate_weekly_digest(user_id, user_health_data)

        return {
            "current_vitals": data,
            "weekly_digest": digest,
            "records_logged": len(user_health_data[user_id])
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/mental-health-assessment")
async def mental_health_assessment_endpoint(request: MentalHealthAssessmentRequest):
    """
    Comprehensive Mental Health Assessment endpoint
    Includes stress/burnout, PHQ-9, GAD-7, and ML risk prediction
    """
    try:
        user_id = request.user_id
        assessment_data = request.assessment_data

        logging.info(f"Processing mental health assessment for user {user_id}")
        logging.info(f"Assessment data: {json.dumps(assessment_data)}")

        # Initialize the mental health assessment tool
        mental_health_tool = MentalHealthAssessmentTool()

        # Validate that country is provided
        if "country" not in assessment_data or not assessment_data["country"]:
            return {"error": "Country is required for mental health assessment and crisis resource recommendations"}

        # Perform comprehensive assessment
        result = mental_health_tool.comprehensive_assessment(assessment_data)

        # Save assessment results to user health data
        if user_id not in user_health_data:
            user_health_data[user_id] = {}

        user_health_data[user_id]["mental_health_assessment"] = {
            "input": assessment_data,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        # Update chat history if user exists
        if user_id in chat_histories:
            # Add user message
            chat_histories[user_id].append({
                "role": "user",
                "content": "I'd like to complete a comprehensive mental health assessment."
            })

            # Create a summary for the chat
            summary = result.get("summary", "Mental health assessment completed.")
            risk_level = result.get("assessments", {}).get("ml_risk_prediction", {}).get("risk_level", "Unknown")

            # Add assistant message with assessment summary
            chat_histories[user_id].append({
                "role": "assistant",
                "content": f"**Mental Health Assessment Complete**\n\n{summary}\n\n**Risk Level**: {risk_level}\n\nI've provided personalized recommendations and follow-up reminders based on your assessment. If you're experiencing a mental health crisis, please reach out to the crisis resources provided."
            })

        logging.info(f"Mental health assessment completed for user {user_id}")
        return result

    except Exception as e:
        error_msg = f"Error in mental health assessment: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return {"error": error_msg}


@app.get("/mental-health-countries")
async def get_mental_health_countries():
    """
    Get list of supported countries for mental health crisis resources
    """
    try:
        from tools.tools_mental_health_assessment import MentalHealthAssessmentTool
        mental_health_tool = MentalHealthAssessmentTool()
        countries = mental_health_tool.get_supported_countries()

        return {
            "supported_countries": countries,
            "total_countries": len(countries),
            "note": "These countries have specific crisis resources available. Other countries will receive generic international resources."
        }
    except Exception as e:
        logging.error(f"Error getting supported countries: {str(e)}")
        return {"error": f"Failed to get supported countries: {str(e)}"}


@app.post("/liver-function/manual")
async def analyze_liver_manual(request: LiverFunctionAssessmentRequest):
    """Liver Function Test Analysis via Manual Entry"""
    try:
        user_id = request.user_id
        lft_data = request.lft_data

        result = analyze_liver_function(
            extracted_values=lft_data.to_extracted_values(),
            dietary_habits=lft_data.dietary_habits,
            medications=lft_data.medications,
            symptoms=lft_data.symptoms,
            hepatitis_markers=lft_data.hepatitis_markers,
            smoking_alcohol_use=lft_data.smoking_alcohol_use,
            medical_conditions=lft_data.medical_conditions,
            input_method="Manual Entry"
        )

        # Save to memory
        if user_id not in user_health_data:
            user_health_data[user_id] = {}
        user_health_data[user_id]["liver_function"] = {
            "data": lft_data.dict(),
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        # Add chat summary
        if user_id in chat_histories:
            chat_histories[user_id].append({
                "role": "user",
                "content": "Please assess my liver function."
            })

            summary = "\n".join(result.get("parameter_status", []))
            summary += f"\n\n**Risk Level**: {result['risk_level']}"
            summary += f"\n**Confidence Level**: {result['confidence_level']}"
            summary += "\n\n**Recommendations:**\n- " + "\n- ".join(result["recommendations"])

            chat_histories[user_id].append({
                "role": "assistant",
                "content": summary
            })

        return result

    except Exception as e:
        logging.error(f"Liver Function Error: {str(e)}")
        return {"error": str(e)}


@app.post("/liver-function/pdf")
async def analyze_pdf(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        with pdfplumber.open(file.file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            extracted_values = extract_lft_values(text)

        result = analyze_liver_function(
            extracted_values=extracted_values,
            input_method="Upload PDF"
        )

        # Store in memory
        if user_id not in user_health_data:
            user_health_data[user_id] = {}

        user_health_data[user_id]["liver_function"] = {
            "data": extracted_values,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        # Add to chat memory
        if user_id in chat_histories:
            chat_histories[user_id].append({
                "role": "user",
                "content": "Please analyze my liver function test PDF."
            })

            summary = "\n".join(result["parameter_status"])
            summary += f"\n\n**Risk Level**: {result['risk_level']}"
            summary += f"\n**Confidence**: {result['confidence_level']}"
            summary += "\n\n**Recommendations:**\n- " + "\n- ".join(result["recommendations"])

            chat_histories[user_id].append({
                "role": "assistant",
                "content": summary
            })

        return JSONResponse(content={"extracted_values": extracted_values, **result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/reproductive-health")
async def reproductive_health_endpoint(request: ReproductiveHealthRequest):
    try:
        user_id = request.user_id
        mode = request.mode
        payload = request.payload

        # Run the unified reproductive health agent
        result = run_reproductive_agent(user_id, mode, payload)

        # Save to user health memory
        if user_id not in user_health_data:
            user_health_data[user_id] = {}
        user_health_data[user_id][f"reproductive_{mode}"] = {
            "mode": mode,
            "input": payload,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        # Chat integration suggestion
        if request.user_id in chat_histories:
            chat_histories[request.user_id].append({
                "role": "user",
                "content": f"I want to check my reproductive health ({request.mode})."
            })
            chat_histories[request.user_id].append({
                "role": "assistant",
                "content": result.get("chat_prompt", "")
            })

        return result
            
        # Format brief result summary for chat
        summary = ""
        if mode == "cycle" and isinstance(result, dict):
            summary += f"🩸 **Next Period**: {result.get('Next Period Start Date', 'N/A')}\n"
            summary += f"🧬 **Ovulation Window**: {result.get('Ovulation Window', 'N/A')}"
        elif mode == "pregnancy":
            summary += f"🤰 **Gestational Age**: {result.get('gestational_age')}\n"
            summary += f"👶 **EDD Range**: {result.get('EDD Window')}\n"
            if result.get("Diagnosis"):
                summary += "\n⚠️ **Symptoms Flagged:**\n- " + "\n- ".join(result["Diagnosis"])
        elif mode == "postpartum":
            summary += f"🍼 **Postpartum Day**: {result.get('Days Since Delivery', 'N/A')}\n"
            summary += f"⏳ **Ovulation Info**: {result.get('Ovulation Info', '')}\n"
            if result.get("Flags"):
                summary += "\n🚨 **Health Flags:**\n- " + "\n- ".join(result["Flags"])

        chat_histories[user_id].append({"role": "assistant", "content": summary})

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# === RUN SERVER ===
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
