import streamlit as st
import requests
import json
import uuid
import numpy as np
from datetime import datetime

# === CONSTANTS ===
API_URL = "http://127.0.0.1:8000"
MODELS = ["qwen2.5:1.5b", "deepseek-r1:1.5b"]
DEFAULT_MODEL = "qwen2.5:1.5b"

# === STYLING ===
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    .chat-title {
        margin-bottom: 0;
    }
    .stButton button {
        background-color: #4d4d4d;
        color: white;
        border: none;
    }
    .stButton button:hover {
        background-color: #5d5d5d;
    }
    .health-metrics {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .quick-actions {
        display: flex;
        gap: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# === SESSION STATE INITIALIZATION ===
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm Dr. Deuce. How can I help you today?"}]

if "chat_title" not in st.session_state:
    st.session_state.chat_title = "New Chat"

if "server_status" not in st.session_state:
    st.session_state.server_status = "Checking..."

if "waiting_for_vitals" not in st.session_state:
    st.session_state.waiting_for_vitals = False

if "waiting_for_health_score" not in st.session_state:
    st.session_state.waiting_for_health_score = False

if "waiting_for_kidney_function" not in st.session_state:
    st.session_state.waiting_for_kidney_function = False

if "waiting_for_lipid_profile" not in st.session_state:
    st.session_state.waiting_for_lipid_profile = False

if "waiting_for_confirmation" not in st.session_state:
    st.session_state.waiting_for_confirmation = False

if "confirmation_type" not in st.session_state:
    st.session_state.confirmation_type = None

if "waiting_for_health_consultation" not in st.session_state:
    st.session_state.waiting_for_health_consultation = False

# Health data is stored on the server side

# === HELPER FUNCTIONS ===
# Recommendations are now handled by the server

def check_server_status():
    """Check if the server is running"""
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            return "Online ✅", response.json()
        return "Error ❌", None
    except requests.RequestException:
        return "Offline ❌", None

def get_default_health_data():
    """Get default health data from the server"""
    try:
        response = requests.get(f"{API_URL}/default-health-data", timeout=5)
        if response.status_code == 200:
            return response.json()
        print(f"Error getting default health data: {response.status_code}")
        # Return the exact DEFAULT_HEALTH_DATA structure
        return {
            "Glucose": None,
            "SpO2": None,
            "ECG (Heart Rate)": None,
            "Blood Pressure (Systolic)": None,
            "Blood Pressure (Diastolic)": None,
            "Weight (BMI)": None,
            "Temperature": None,
            "Malaria": "Unknown",
            "Widal Test": "Unknown",
            "Hepatitis B": "Unknown",
            "Voluntary Serology": "Unknown",
            "Perfusion_index": None,
            "Waist Circumference": None,
            "Fev": None
        }
    except requests.RequestException as e:
        print(f"Error connecting to server: {str(e)}")
        # Return the exact DEFAULT_HEALTH_DATA structure
        return {
            "Glucose": None,
            "SpO2": None,
            "ECG (Heart Rate)": None,
            "Blood Pressure (Systolic)": None,
            "Blood Pressure (Diastolic)": None,
            "Weight (BMI)": None,
            "Temperature": None,
            "Malaria": "Unknown",
            "Widal Test": "Unknown",
            "Hepatitis B": "Unknown",
            "Voluntary Serology": "Unknown",
            "Perfusion_index": None,
            "Waist Circumference": None,
            "Fev": None
        }

def query_agent(user_query, model):
    """Send a query to the agent, including chat and health history"""
    try:
        # Fetch chat + health history from MCP server
        chat_context = fetch_chat_history()

        payload = {
            "session_id": st.session_state.session_id,
            "user_id": st.session_state.user_id,
            "query": user_query,
            "model": model,
            "chat_history": chat_context  # 🔗 Include prior chat + health data if backend supports it
        }

        response = requests.post(f"{API_URL}/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                return f"Error: {data['error']}"

            # Update chat title if available
            if "chat_title" in data:
                st.session_state.chat_title = data["chat_title"]

            # Check for tool intents
            if "tools_used" in data:
                tools = data.get("tools_used", [])
                if "health_score_intent" in tools:
                    st.session_state.waiting_for_confirmation = True
                    st.session_state.confirmation_type = "health_score"
                elif "vital_signs_intent" in tools:
                    st.session_state.waiting_for_confirmation = True
                    st.session_state.confirmation_type = "vital_signs"
                elif "kidney_function_intent" in tools:
                    st.session_state.waiting_for_confirmation = True
                    st.session_state.confirmation_type = "kidney_function"
                elif "lipid_profile_intent" in tools:
                    st.session_state.waiting_for_confirmation = True
                    st.session_state.confirmation_type = "lipid_profile"

            return data["response"]
        return f"Error: Server returned status code {response.status_code}"
    except requests.RequestException as e:
        return f"Error connecting to server: {str(e)}"

def submit_vital_signs(vital_signs):
    """Submit vital signs to the server"""
    try:
        # Log the data being sent
        print(f"Submitting vital signs: {json.dumps(vital_signs, indent=2)}")

        payload = {
            "user_id": st.session_state.user_id,
            "vital_signs": vital_signs
        }

        # Make the request
        response = requests.post(f"{API_URL}/vital-signs", json=payload)

        # Handle different status codes
        if response.status_code == 200:
            result = response.json()
            print(f"Received response: {json.dumps(result, indent=2)}")
            return result
        elif response.status_code == 500:
            # Try to parse the error message from the response
            try:
                error_data = response.json()
                error_msg = error_data.get('error', f"Server error (500): {response.text}")
            except:
                error_msg = f"Server error (500): {response.text}"
            print(f"Server error: {error_msg}")
            return {"error": error_msg}
        else:
            error_msg = f"Server returned status code {response.status_code}: {response.text}"
            print(error_msg)
            return {"error": error_msg}
    except requests.RequestException as e:
        error_msg = f"Error connecting to server: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

def submit_health_score(health_data):
    """Submit health data for score analysis"""
    try:
        # Log the data being sent
        print(f"Submitting health data: {json.dumps(health_data, indent=2)}")

        payload = {
            "user_id": st.session_state.user_id,
            "health_data": health_data
        }

        # Make the request
        response = requests.post(f"{API_URL}/health-score", json=payload)

        # Handle different status codes
        if response.status_code == 200:
            result = response.json()
            print(f"Received response: {json.dumps(result, indent=2)}")
            return result
        elif response.status_code == 500:
            # Try to parse the error message from the response
            try:
                error_data = response.json()
                error_msg = error_data.get('error', f"Server error (500): {response.text}")
            except:
                error_msg = f"Server error (500): {response.text}"
            print(f"Server error: {error_msg}")
            return {"error": error_msg}
        else:
            error_msg = f"Server returned status code {response.status_code}: {response.text}"
            print(error_msg)
            return {"error": error_msg}
    except requests.RequestException as e:
        error_msg = f"Error connecting to server: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

def submit_kidney_function(kidney_data):
    """Submit kidney function data for analysis"""
    try:
        # Log the data being sent
        print(f"Submitting kidney function data: {json.dumps(kidney_data, indent=2)}")

        payload = {
            "user_id": st.session_state.user_id,
            "kidney_data": kidney_data
        }

        # Make the request
        response = requests.post(f"{API_URL}/kidney-function", json=payload)

        # Handle different status codes
        if response.status_code == 200:
            result = response.json()
            print(f"Received response: {json.dumps(result, indent=2)}")
            return result
        elif response.status_code == 500:
            # Try to parse the error message from the response
            try:
                error_data = response.json()
                error_msg = error_data.get('error', f"Server error (500): {response.text}")
            except:
                error_msg = f"Server error (500): {response.text}"
            print(f"Server error: {error_msg}")
            return {"error": error_msg}
        else:
            error_msg = f"Server returned status code {response.status_code}: {response.text}"
            print(error_msg)
            return {"error": error_msg}
    except requests.RequestException as e:
        error_msg = f"Error connecting to server: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

# The function is kept as a fallback but will not be used in normal operation
def generate_kidney_recommendations(abnormal_params, kidney_data):
    """Generate personalized recommendations based on abnormal kidney function parameters"""
    recommendations = []

    # Specific recommendations based on abnormal parameters
    if "Serum Creatinine" in abnormal_params or "eGFR" in abnormal_params:
        recommendations.append("🔹 Consider reducing protein intake and consult with a nephrologist")
        recommendations.append("🔹 Monitor blood pressure regularly and keep it under control")

    if "Serum Sodium" in abnormal_params:
        if kidney_data.get("Serum Sodium", 0) > 145:
            recommendations.append("🔹 Reduce salt intake and increase water consumption")
        else:
            recommendations.append("🔹 Consult with your doctor about proper fluid intake")

    if "Serum Potassium" in abnormal_params:
        if kidney_data.get("Serum Potassium", 0) > 5.0:
            recommendations.append("🔹 Limit high-potassium foods like bananas, oranges, and potatoes")
        else:
            recommendations.append("🔹 Include potassium-rich foods in your diet as advised by your doctor")

    if "Serum Calcium" in abnormal_params:
        recommendations.append("🔹 Discuss vitamin D and calcium supplementation with your healthcare provider")

    if "Serum Uric Acid" in abnormal_params:
        recommendations.append("🔹 Limit purine-rich foods like red meat, seafood, and beer")
        recommendations.append("🔹 Increase intake of cherries and vitamin C-rich foods")

    if "ACR" in abnormal_params or "Urine Albumin" in abnormal_params:
        recommendations.append("🔹 Control blood sugar and blood pressure carefully")
        recommendations.append("🔹 Follow up with regular kidney function tests")

    # Format the recommendations as a list
    formatted_recs = "**Personalized Recommendations:**\n"
    for rec in recommendations:
        # Check if the recommendation already starts with a list marker
        if rec.strip().startswith("🔹"):
            # Remove the emoji and add a dash instead
            rec = rec.replace("🔹", "-", 1).strip()
            formatted_recs += f"{rec}\n"
        else:
            formatted_recs += f"- {rec}\n"

    return formatted_recs

def submit_lipid_profile(lipid_data):
    """Submit lipid profile data for analysis using the new tools_lipid_profile2 tool"""
    try:
        # Log the data being sent
        print(f"Submitting lipid profile data: {json.dumps(lipid_data, indent=2)}")

        payload = {
            "user_id": st.session_state.user_id,
            "lipid_data": lipid_data
        }

        # Make the request
        response = requests.post(f"{API_URL}/lipid-profile", json=payload)

        # Handle different status codes
        if response.status_code == 200:
            result = response.json()
            print(f"Received response: {json.dumps(result, indent=2)}")

            # Format the recommendations as a list if they're not already
            if "recommendations" in result and isinstance(result["recommendations"], list):
                formatted_recs = []
                for rec in result["recommendations"]:
                    if not rec.startswith("-"):
                        formatted_recs.append(f"- {rec}")
                    else:
                        formatted_recs.append(rec)
                result["formatted_recommendations"] = formatted_recs

            return result
        elif response.status_code == 500:
            # Try to parse the error message from the response
            try:
                error_data = response.json()
                error_msg = error_data.get('error', f"Server error (500): {response.text}")
            except:
                error_msg = f"Server error (500): {response.text}"
            print(f"Server error: {error_msg}")
            return {"error": error_msg}
        else:
            error_msg = f"Server returned status code {response.status_code}: {response.text}"
            print(error_msg)
            return {"error": error_msg}
    except requests.RequestException as e:
        error_msg = f"Error connecting to server: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

def handle_confirmation(confirmation_type):
    """Handle user confirmation for different actions"""
    if confirmation_type == "vital_signs":
        st.session_state.waiting_for_vitals = True
        st.session_state.waiting_for_confirmation = False
        st.session_state.confirmation_type = None
        return "Please enter your vital signs below:"

    elif confirmation_type == "health_score":
        st.session_state.waiting_for_health_score = True
        st.session_state.waiting_for_confirmation = False
        st.session_state.confirmation_type = None
        return "Please enter your health data below for analysis:"

    elif confirmation_type == "kidney_function":
        st.session_state.waiting_for_kidney_function = True
        st.session_state.waiting_for_confirmation = False
        st.session_state.confirmation_type = None
        return "Please enter your kidney function test results below:"

    elif confirmation_type == "lipid_profile":
        st.session_state.waiting_for_lipid_profile = True
        st.session_state.waiting_for_confirmation = False
        st.session_state.confirmation_type = None
        return "Please enter your lipid profile test results below:"

    elif confirmation_type == "health_consultation":
        # Generate booking URL
        booking_url = generate_booking_url()

        # Reset confirmation state
        st.session_state.waiting_for_confirmation = False
        st.session_state.confirmation_type = None

        # Return response with booking link
        return f"""Thank you for confirming. I've generated a booking link for your health consultation.

**[Click here to book your consultation]({booking_url})**

Your health data including age and sex will be used to prepare for your consultation. Is there anything specific you'd like to discuss during your appointment?"""

    return "I'm not sure what you're confirming. How can I help you?"

def fetch_chat_history():
    """Fetch chat history (including health data) from the MCP server."""
    try:
        response = requests.get(
            f"{API_URL}/chat-history",
            params={
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id
            },
            timeout=5
        )
        if response.status_code == 200:
            return response.json()  # Should return a dict with messages and possibly health data
        else:
            print(f"Failed to fetch chat history. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error fetching chat history: {e}")
        return None

def generate_booking_url():
    """Generate a URL for booking a health consultation."""
    # Generate a unique booking ID
    booking_id = str(uuid.uuid4())

    # Create a booking URL with user ID and booking ID
    # This URL will allow the consultation service to fetch the user's data from the server
    booking_url = f"https://drdeucehealth.com/book-consultation?user_id={st.session_state.user_id}&booking_id={booking_id}"

    return booking_url


# === SIDEBAR ===
with st.sidebar:
    # Server status at the top of the sidebar
    status_text, status_data = check_server_status()
    st.session_state.server_status = status_text

    st.markdown(f"### Agent Server: {st.session_state.server_status}")

    # User ID
    st.markdown(f"**User ID**: {st.session_state.user_id[:8]}...")

    # Model selection
    selected_model = st.selectbox(
        "Choose Model",
        MODELS,
        index=MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in MODELS else 0
    )

    # Chat title
    st.markdown("### Chat")
    st.markdown(f"**Title**: {st.session_state.chat_title}")

    # New chat button
    if st.button("New Chat", key="new_chat"):
        # Reset session state
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm Dr. Deuce. How can I help you today?"}]
        st.session_state.chat_title = "New Chat"
        st.session_state.waiting_for_vitals = False
        st.session_state.waiting_for_health_score = False
        st.session_state.waiting_for_kidney_function = False
        st.session_state.waiting_for_lipid_profile = False
        st.session_state.waiting_for_confirmation = False
        st.session_state.waiting_for_health_consultation = False
        st.session_state.confirmation_type = None

        # Health data is preserved on the server without notification

        st.rerun()



# === MAIN CONTENT ===
st.title("🩺 Dr. Deuce Health Assistant")
st.caption("Your AI Assistant for Healthcare Issues")

# Quick action buttons above the chat
st.markdown("### Quick Actions")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Health Score Analysis", key="btn_health_score"):
        st.session_state.waiting_for_confirmation = True
        st.session_state.confirmation_type = "health_score"
        st.session_state.message_log.append({"role": "ai", "content": "Would you like to analyze your health score? Type 'yes' to begin."})
        st.rerun()

with col2:
    if st.button("Monitor Vital Signs", key="btn_vitals"):
        st.session_state.waiting_for_confirmation = True
        st.session_state.confirmation_type = "vital_signs"
        st.session_state.message_log.append({"role": "ai", "content": "Would you like to enter your vital signs for monitoring? Type 'yes' to begin."})
        st.rerun()

with col3:
    if st.button("Kidney Function Test", key="btn_kidney_function"):
        st.session_state.waiting_for_confirmation = True
        st.session_state.confirmation_type = "kidney_function"
        st.session_state.message_log.append({"role": "ai", "content": "Would you like to analyze your kidney function? Type 'yes' to begin."})
        st.rerun()

with col4:
    if st.button("Lipid Profile Test", key="btn_lipid_profile"):
        st.session_state.waiting_for_confirmation = True
        st.session_state.confirmation_type = "lipid_profile"
        st.session_state.message_log.append({"role": "ai", "content": "Would you like to analyze your lipid profile? Type 'yes' to begin."})
        st.rerun()

with col5:
    if st.button("Health Consultation", key="btn_consultation"):
        st.session_state.waiting_for_confirmation = True
        st.session_state.confirmation_type = "health_consultation"
        st.session_state.message_log.append({"role": "user", "content": "I'd like a health consultation"})
        st.session_state.message_log.append({"role": "ai", "content": "Would you like to book a health consultation with one of our healthcare professionals? Your health data including age and sex will be pulled from our system for this consultation. Type 'yes' to confirm."})
        st.rerun()

# Chat history
st.markdown("### Chat History")
chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Vital signs form (only shown when waiting for vitals)
if st.session_state.waiting_for_vitals:
    # Get default health data from the server
    default_data = get_default_health_data()

    with st.form(key="vital_signs_form"):
        st.markdown("### Enter Your Vital Signs")
        col1, col2 = st.columns(2)

        # Add a note about N/A values
        # st.info("For any field you don't have data for, select 'N/A' from the dropdown.")

        # Helper function to create options for numeric fields
        def create_numeric_options(start, end, step=1, default=None):
            options = [f"{i:.1f}" if step < 1 else str(i) for i in np.arange(start, end + step, step)]
            options.append("N/A")
            # Set default index
            if default is None or default == "N/A":
                default_index = len(options) - 1  # N/A is the last option
            else:
                # Find the closest value to default
                closest_val = min(options[:-1], key=lambda x: abs(float(x) - float(default)))
                default_index = options.index(closest_val)
            return options, default_index

        with col1:
            # Blood Pressure (Systolic) with N/A option
            bp_sys_default = "120" if default_data.get("Blood Pressure (Systolic)") is None else str(default_data.get("Blood Pressure (Systolic)"))
            bp_sys_options, bp_sys_default_index = create_numeric_options(70, 200, 1, bp_sys_default)
            bp_sys_selected = st.selectbox("Blood Pressure (Systolic)",
                                         options=bp_sys_options,
                                         index=bp_sys_default_index,
                                         key="vital_bp_sys_select")
            blood_pressure_systolic = None if bp_sys_selected == "N/A" else float(bp_sys_selected)

            # Blood Pressure (Diastolic) with N/A option
            bp_dia_default = "80" if default_data.get("Blood Pressure (Diastolic)") is None else str(default_data.get("Blood Pressure (Diastolic)"))
            bp_dia_options, bp_dia_default_index = create_numeric_options(40, 120, 1, bp_dia_default)
            bp_dia_selected = st.selectbox("Blood Pressure (Diastolic)",
                                         options=bp_dia_options,
                                         index=bp_dia_default_index,
                                         key="vital_bp_dia_select")
            blood_pressure_diastolic = None if bp_dia_selected == "N/A" else float(bp_dia_selected)

            # Heart Rate with N/A option
            hr_default = "75" if default_data.get("ECG (Heart Rate)") is None else str(default_data.get("ECG (Heart Rate)"))
            hr_options, hr_default_index = create_numeric_options(40, 200, 1, hr_default)
            hr_selected = st.selectbox("Heart Rate (bpm)",
                                     options=hr_options,
                                     index=hr_default_index,
                                     key="vital_hr_select")
            heart_rate = None if hr_selected == "N/A" else float(hr_selected)
            
            # Waist Circumference with N/A option
            waist_default = "85" if default_data.get("Waist Circumference") is None else str(default_data.get("Waist Circumference"))
            waist_options, waist_default_index = create_numeric_options(50, 150, 1, waist_default)
            waist_selected = st.selectbox("Waist Circumference (cm)",
                                       options=waist_options,
                                       index=waist_default_index,
                                       key="vital_waist_select")
            waist_circumference = None if waist_selected == "N/A" else float(waist_selected)

        with col2:
            # Temperature with N/A option
            temp_default = "36.8" if default_data.get("Temperature") is None else str(default_data.get("Temperature"))
            temp_options, temp_default_index = create_numeric_options(35.0, 42.0, 0.1, temp_default)
            temp_selected = st.selectbox("Temperature (°C)",
                                       options=temp_options,
                                       index=temp_default_index,
                                       key="vital_temp_select")
            temperature = None if temp_selected == "N/A" else float(temp_selected)

            # Glucose with N/A option
            glucose_default = "100" if default_data.get("Glucose") is None else str(default_data.get("Glucose"))
            glucose_options, glucose_default_index = create_numeric_options(50, 300, 1, glucose_default)
            glucose_selected = st.selectbox("Glucose (mg/dL)",
                                          options=glucose_options,
                                          index=glucose_default_index,
                                          key="vital_glucose_select")
            glucose = None if glucose_selected == "N/A" else float(glucose_selected)

            # SpO2 with N/A option
            spo2_default = "98" if default_data.get("SpO2") is None else str(default_data.get("SpO2"))
            spo2_options, spo2_default_index = create_numeric_options(80, 100, 1, spo2_default)
            spo2_selected = st.selectbox("SpO2 (%)",
                                       options=spo2_options,
                                       index=spo2_default_index,
                                       key="vital_spo2_select")
            spo2 = None if spo2_selected == "N/A" else float(spo2_selected)

            # BMI with N/A option
            bmi_default = "22.0" if default_data.get("BMI") is None else str(default_data.get("BMI"))
            bmi_options, bmi_default_index = create_numeric_options(15.0, 50.0, 0.1, bmi_default)
            bmi_selected = st.selectbox("BMI (kg/m²)",
                                      options=bmi_options,
                                      index=bmi_default_index,
                                      key="vital_bmi_select")
            bmi = None if bmi_selected == "N/A" else float(bmi_selected)

        submit_button = st.form_submit_button(label="Submit Vital Signs")

        if submit_button:
            vital_signs = {
                "Blood_Pressure_Systolic": blood_pressure_systolic,
                "Blood_Pressure_Diastolic": blood_pressure_diastolic,
                "Heart_Rate": heart_rate,
                "Temperature": temperature,
                "Glucose": glucose,
                "SpO2": spo2
                #"Waist_Circumference": waist_circumference,
                #"BMI": bmi
            }

            result = submit_vital_signs(vital_signs)

            if "error" in result:
                response = f"Error processing vital signs: {result['error']}"
            else:
                # Data is automatically saved on the server side in the submit_vital_signs function

                response = f"**Vital Signs Analysis**\n\n{result.get('analysis', '')}"
                if result.get('alerts'):
                    response += f"\n\n**Alerts**:\n{result['alerts']}"

                # Add note about saved data
                # response += "\n\n*Your vital signs data has been saved on the server. You can ask for personalized recommendations anytime.*"

            st.session_state.message_log.append({"role": "user", "content": f"I've submitted my vital signs: {json.dumps(vital_signs, indent=2)}"})
            st.session_state.message_log.append({"role": "ai", "content": response})
            st.session_state.waiting_for_vitals = False
            st.rerun()

# Health score form (only shown when waiting for health score)
if st.session_state.waiting_for_health_score:
    # Get default health data from the server
    default_data = get_default_health_data()

    with st.form(key="health_score_form"):
        st.markdown("### Enter Your Health Data")
        col1, col2 = st.columns(2)

        with col1:
            # Add a note about N/A values
            # st.info("For any field you don't have data for, select 'N/A' from the dropdown.")

            # Helper function to create options for numeric fields
            def create_numeric_options(start, end, step=1, default=None):
                options = [f"{i:.1f}" if step < 1 else str(i) for i in np.arange(start, end + step, step)]
                options.append("N/A")
                # Set default index
                if default is None or default == "N/A":
                    default_index = len(options) - 1  # N/A is the last option
                else:
                    # Find the closest value to default
                    closest_val = min(options[:-1], key=lambda x: abs(float(x) - float(default)))
                    default_index = options.index(closest_val)
                return options, default_index

            # BMI input with N/A option
            bmi_options, bmi_default_index = create_numeric_options(10.0, 50.0, 0.1, "24.2")
            bmi_selected = st.selectbox("BMI Value (normal range: 18.5-24.9)",
                                      options=bmi_options,
                                      index=bmi_default_index,
                                      key="bmi_select")
            bmi = None if bmi_selected == "N/A" else float(bmi_selected)

            # Blood Pressure (Systolic) with N/A option
            bp_sys_default = "120" if default_data.get("Blood Pressure (Systolic)") is None else str(default_data.get("Blood Pressure (Systolic)"))
            bp_sys_options, bp_sys_default_index = create_numeric_options(70, 200, 1, bp_sys_default)
            bp_sys_selected = st.selectbox("Blood Pressure (Systolic)",
                                         options=bp_sys_options,
                                         index=bp_sys_default_index,
                                         key="bp_sys_select")
            blood_pressure_systolic = None if bp_sys_selected == "N/A" else float(bp_sys_selected)

            # Blood Pressure (Diastolic) with N/A option
            bp_dia_default = "80" if default_data.get("Blood Pressure (Diastolic)") is None else str(default_data.get("Blood Pressure (Diastolic)"))
            bp_dia_options, bp_dia_default_index = create_numeric_options(40, 120, 1, bp_dia_default)
            bp_dia_selected = st.selectbox("Blood Pressure (Diastolic)",
                                         options=bp_dia_options,
                                         index=bp_dia_default_index,
                                         key="bp_dia_select")
            blood_pressure_diastolic = None if bp_dia_selected == "N/A" else float(bp_dia_selected)

            # Heart Rate with N/A option
            hr_default = "75" if default_data.get("ECG (Heart Rate)") is None else str(default_data.get("ECG (Heart Rate)"))
            hr_options, hr_default_index = create_numeric_options(40, 200, 1, hr_default)
            hr_selected = st.selectbox("Heart Rate (bpm)",
                                     options=hr_options,
                                     index=hr_default_index,
                                     key="hr_select")
            heart_rate = None if hr_selected == "N/A" else float(hr_selected)

            # Test results with Unknown option already included
            malaria = st.selectbox("Malaria", ["Positive", "Negative", "Unknown"], index=2)
            widal_test = st.selectbox("Widal Test", ["Positive", "Negative", "Unknown"], index=2)
            hepatitis_b = st.selectbox("Hepatitis B", ["Positive", "Negative", "Unknown"], index=2)
            voluntary_serology = st.selectbox("Voluntary Serology", ["Positive", "Negative", "Unknown"], index=2)

        with col2:
            # Glucose with N/A option
            glucose_default = "100" if default_data.get("Glucose") is None else str(default_data.get("Glucose"))
            glucose_options, glucose_default_index = create_numeric_options(50, 300, 1, glucose_default)
            glucose_selected = st.selectbox("Glucose (mg/dL)",
                                          options=glucose_options,
                                          index=glucose_default_index,
                                          key="glucose_select")
            glucose = None if glucose_selected == "N/A" else float(glucose_selected)

            # SpO2 with N/A option
            spo2_default = "98" if default_data.get("SpO2") is None else str(default_data.get("SpO2"))
            spo2_options, spo2_default_index = create_numeric_options(80, 100, 1, spo2_default)
            spo2_selected = st.selectbox("SpO2 (%)",
                                       options=spo2_options,
                                       index=spo2_default_index,
                                       key="spo2_select")
            spo2 = None if spo2_selected == "N/A" else float(spo2_selected)

            # Temperature with N/A option
            temp_default = "36.8" if default_data.get("Temperature") is None else str(default_data.get("Temperature"))
            temp_options, temp_default_index = create_numeric_options(35.0, 42.0, 0.1, temp_default)
            temp_selected = st.selectbox("Temperature (°C)",
                                       options=temp_options,
                                       index=temp_default_index,
                                       key="temp_select")
            temperature = None if temp_selected == "N/A" else float(temp_selected)

            # Perfusion Index with N/A option
            pi_default = "5.0" if default_data.get("Perfusion_index") is None else str(default_data.get("Perfusion_index"))
            pi_options, pi_default_index = create_numeric_options(0.1, 20.0, 0.1, pi_default)
            pi_selected = st.selectbox("Perfusion Index",
                                     options=pi_options,
                                     index=pi_default_index,
                                     key="pi_select")
            perfusion_index = None if pi_selected == "N/A" else float(pi_selected)

            # Waist Circumference with N/A option
            wc_default = "80.0" if default_data.get("Waist Circumference") is None else str(default_data.get("Waist Circumference"))
            wc_options, wc_default_index = create_numeric_options(50.0, 200.0, 0.5, wc_default)
            wc_selected = st.selectbox("Waist Circumference (cm)",
                                     options=wc_options,
                                     index=wc_default_index,
                                     key="wc_select")
            waist_circumference = None if wc_selected == "N/A" else float(wc_selected)

            # FEV with N/A option
            fev_default = "85.0" if default_data.get("Fev") is None else str(default_data.get("Fev"))
            fev_options, fev_default_index = create_numeric_options(20.0, 150.0, 0.5, fev_default)
            fev_selected = st.selectbox("FEV (%) - Forced Expiratory Volume",
                                      options=fev_options,
                                      index=fev_default_index,
                                      key="fev_select")
            fev = None if fev_selected == "N/A" else float(fev_selected)

        submit_button = st.form_submit_button(label="Analyze Health Score")

        if submit_button:
            # Use the exact structure from DEFAULT_HEALTH_DATA
            health_data = {
                "Glucose": glucose,
                "SpO2": spo2,
                "ECG (Heart Rate)": heart_rate,
                "Blood Pressure (Systolic)": blood_pressure_systolic,
                "Blood Pressure (Diastolic)": blood_pressure_diastolic,
                "Weight (BMI)": bmi,  # This field stores the BMI value directly
                "Temperature": temperature,
                "Malaria": malaria,
                "Widal Test": widal_test,
                "Hepatitis B": hepatitis_b,
                "Voluntary Serology": voluntary_serology,
                "Perfusion_index": perfusion_index,
                "Waist Circumference": waist_circumference,
                "Fev": fev
            }

            result = submit_health_score(health_data)

            if "error" in result:
                response = f"Error analyzing health score: {result['error']}"
            else:
                # Data is automatically saved on the server side in the submit_health_score function

                response = f"**Health Score Analysis**\n\n{result.get('analysis', '')}"

            st.session_state.message_log.append({"role": "user", "content": f"I've submitted my health data for analysis: {json.dumps(health_data, indent=2)}"})
            st.session_state.message_log.append({"role": "ai", "content": response})
            st.session_state.waiting_for_health_score = False
            st.rerun()

# Kidney function test form (only shown when waiting for kidney function test)
if st.session_state.waiting_for_kidney_function:
    with st.form(key="kidney_function_form"):
        st.markdown("### Enter Your Kidney Function Test Results")
        col1, col2 = st.columns(2)

        # Add a note about N/A values
        # st.info("For any field you don't have data for, select 'N/A' from the dropdown.")

        # Helper function to create options for numeric fields
        def create_numeric_options(start, end, step=1, default=None):
            options = [f"{i:.1f}" if step < 1 else str(i) for i in np.arange(start, end + step, step)]
            options.append("N/A")
            # Set default index
            if default is None or default == "N/A":
                default_index = len(options) - 1  # N/A is the last option
            else:
                # Find the closest value to default
                closest_val = min(options[:-1], key=lambda x: abs(float(x) - float(default)))
                default_index = options.index(closest_val)
            return options, default_index

        with col1:
            # Serum Urea with N/A option
            urea_options, urea_default_index = create_numeric_options(0.0, 200.0, 0.1, "5.0")
            urea_selected = st.selectbox("Serum Urea",
                                       options=urea_options,
                                       index=urea_default_index,
                                       key="kidney_urea_select")
            serum_urea = None if urea_selected == "N/A" else float(urea_selected)

            # Serum Creatinine with N/A option
            creatinine_options, creatinine_default_index = create_numeric_options(0.0, 20.0, 0.1, "1.0")
            creatinine_selected = st.selectbox("Serum Creatinine",
                                             options=creatinine_options,
                                             index=creatinine_default_index,
                                             key="kidney_creatinine_select")
            serum_creatinine = None if creatinine_selected == "N/A" else float(creatinine_selected)

            # Serum Sodium with N/A option
            sodium_options, sodium_default_index = create_numeric_options(100.0, 200.0, 0.1, "140.0")
            sodium_selected = st.selectbox("Serum Sodium",
                                         options=sodium_options,
                                         index=sodium_default_index,
                                         key="kidney_sodium_select")
            serum_sodium = None if sodium_selected == "N/A" else float(sodium_selected)

            # Serum Potassium with N/A option
            potassium_options, potassium_default_index = create_numeric_options(1.0, 10.0, 0.1, "4.0")
            potassium_selected = st.selectbox("Serum Potassium",
                                            options=potassium_options,
                                            index=potassium_default_index,
                                            key="kidney_potassium_select")
            serum_potassium = None if potassium_selected == "N/A" else float(potassium_selected)

            # Serum Calcium with N/A option
            calcium_options, calcium_default_index = create_numeric_options(5.0, 15.0, 0.1, "9.5")
            calcium_selected = st.selectbox("Serum Calcium",
                                          options=calcium_options,
                                          index=calcium_default_index,
                                          key="kidney_calcium_select")
            serum_calcium = None if calcium_selected == "N/A" else float(calcium_selected)

            # Serum Uric Acid with N/A option
            uric_acid_options, uric_acid_default_index = create_numeric_options(1.0, 20.0, 0.1, "5.0")
            uric_acid_selected = st.selectbox("Serum Uric Acid",
                                            options=uric_acid_options,
                                            index=uric_acid_default_index,
                                            key="kidney_uric_acid_select")
            serum_uric_acid = None if uric_acid_selected == "N/A" else float(uric_acid_selected)

        with col2:
            # Urine Albumin with N/A option
            albumin_options, albumin_default_index = create_numeric_options(0.0, 1000.0, 1.0, "10.0")
            albumin_selected = st.selectbox("Urine Albumin",
                                          options=albumin_options,
                                          index=albumin_default_index,
                                          key="kidney_albumin_select")
            urine_albumin = None if albumin_selected == "N/A" else float(albumin_selected)

            # Urine Creatinine with N/A option
            urine_creat_options, urine_creat_default_index = create_numeric_options(0.0, 500.0, 1.0, "100.0")
            urine_creat_selected = st.selectbox("Urine Creatinine",
                                              options=urine_creat_options,
                                              index=urine_creat_default_index,
                                              key="kidney_urine_creat_select")
            urine_creatinine = None if urine_creat_selected == "N/A" else float(urine_creat_selected)

            # Chloride with N/A option
            chloride_options, chloride_default_index = create_numeric_options(50.0, 150.0, 0.1, "100.0")
            chloride_selected = st.selectbox("Chloride",
                                           options=chloride_options,
                                           index=chloride_default_index,
                                           key="kidney_chloride_select")
            chloride = None if chloride_selected == "N/A" else float(chloride_selected)

            # Bicarbonate with N/A option
            bicarb_options, bicarb_default_index = create_numeric_options(10.0, 50.0, 0.1, "25.0")
            bicarb_selected = st.selectbox("Bicarbonate",
                                         options=bicarb_options,
                                         index=bicarb_default_index,
                                         key="kidney_bicarb_select")
            bicarbonate = None if bicarb_selected == "N/A" else float(bicarb_selected)

            # Age with N/A option
            age_options, age_default_index = create_numeric_options(1, 120, 1, "40")
            age_selected = st.selectbox("Age",
                                      options=age_options,
                                      index=age_default_index,
                                      key="kidney_age_select")
            age = None if age_selected == "N/A" else int(age_selected)

            # Sex selection (no N/A option needed as it's already a dropdown)
            sex = st.selectbox("Sex", ["Male", "Female", "N/A"], index=0, key="kidney_sex_select")

        submit_button = st.form_submit_button(label="Analyze Kidney Function")

        if submit_button:
            kidney_data = {
                "Serum Urea": serum_urea,
                "Serum Creatinine": serum_creatinine,
                "Serum Sodium": serum_sodium,
                "Serum Potassium": serum_potassium,
                "Serum Calcium": serum_calcium,
                "Serum Uric Acid": serum_uric_acid,
                "Urine Albumin": urine_albumin,
                "Urine Creatinine": urine_creatinine,
                "Chloride": chloride,
                "Bicarbonate": bicarbonate,
                "Age": age,
                "Sex": sex
            }

            result = submit_kidney_function(kidney_data)

            if "error" in result:
                response = f"Error analyzing kidney function: {result['error']}"
            else:
                # Data is automatically saved on the server side in the submit_kidney_function function

                # Format the response
                analysis_items = result.get("analysis", "")
                formatted_analysis = ""
                if isinstance(analysis_items, list):
                    formatted_analysis = "**Analysis:**\n"
                    for item in analysis_items:
                        formatted_analysis += f"- {item}\n"
                elif isinstance(analysis_items, str):
                    # If it's a string, try to split by newlines and format as list
                    lines = analysis_items.split("\n")
                    formatted_analysis = "**Analysis Results:**\n"
                    for line in lines:
                        if line.strip():  # Skip empty lines
                            formatted_analysis += f"- {line}\n"

                overall_health = result.get("overall_health", "Unknown")
                confidence_level = result.get("confidence_level", "Unknown")
                missing_parameters = result.get("missing_parameters", [])
                recommendations = result.get("recommendations", [])

                response = f"**Kidney Function Analysis**\n\n"

                # Add analysis results
                if formatted_analysis:
                    response += f"{formatted_analysis}\n"

                # Add overall health
                response += f"**Findings**:\n{overall_health}\n\n"

                # Add confidence level
                response += f"**Confidence Level**: {confidence_level} "

                if missing_parameters:
                    response += f"(Due to missing parameters: {', '.join(missing_parameters)})\n\n"
                    response += f"Some parameters necessary for a more complete analysis were not provided, which may affect the accuracy of this assessment.\n\n"
                else:
                    response += "(Due to complete data)\n\n"

                if recommendations:
                    response += "**Personalized Recommendations:**\n"
                    for rec in recommendations:
                        if not rec.startswith("-"):
                            response += f"- {rec}\n"
                        else:
                            response += f"{rec}\n"
                    response += "\n"

                # response += "You can ask me for more specific recommendations based on your test results at any time."

                # Add note about saved data
                # response += "\n\n*Your kidney function data has been saved on the server. You can ask for personalized recommendations anytime.*"

            st.session_state.message_log.append({"role": "user", "content": f"I've submitted my kidney function test results for analysis: {json.dumps(kidney_data, indent=2)}"})
            st.session_state.message_log.append({"role": "ai", "content": response})
            st.session_state.waiting_for_kidney_function = False
            st.rerun()

# Lipid profile test form (only shown when waiting for lipid profile test)
if st.session_state.waiting_for_lipid_profile:
    with st.form(key="lipid_profile_form"):
        st.markdown("### Enter Your Lipid Profile Test Results")
        col1, col2 = st.columns(2)

        # Add a note about N/A values
        # st.info("For any field you don't have data for, select 'N/A' from the dropdown.")

        # Helper function to create options for numeric fields
        def create_numeric_options(start, end, step=1, default=None):
            options = [f"{i:.1f}" if step < 1 else str(i) for i in np.arange(start, end + step, step)]
            options.append("N/A")
            # Set default index
            if default is None or default == "N/A":
                default_index = len(options) - 1  # N/A is the last option
            else:
                # Find the closest value to default
                closest_val = min(options[:-1], key=lambda x: abs(float(x) - float(default)))
                default_index = options.index(closest_val)
            return options, default_index

        with col1:
            # Total Cholesterol with N/A option
            total_chol_options, total_chol_default_index = create_numeric_options(100, 400, 1, "200")
            total_chol_selected = st.selectbox("Total Cholesterol (mg/dL)",
                                             options=total_chol_options,
                                             index=total_chol_default_index,
                                             key="lipid_total_chol_select")
            total_chol = None if total_chol_selected == "N/A" else float(total_chol_selected)

            # LDL Cholesterol with N/A option
            ldl_options, ldl_default_index = create_numeric_options(30, 300, 1, "130")
            ldl_selected = st.selectbox("LDL Cholesterol (mg/dL)",
                                      options=ldl_options,
                                      index=ldl_default_index,
                                      key="lipid_ldl_select")
            ldl = None if ldl_selected == "N/A" else float(ldl_selected)

            # HDL Cholesterol with N/A option
            hdl_options, hdl_default_index = create_numeric_options(20, 100, 1, "50")
            hdl_selected = st.selectbox("HDL Cholesterol (mg/dL)",
                                      options=hdl_options,
                                      index=hdl_default_index,
                                      key="lipid_hdl_select")
            hdl = None if hdl_selected == "N/A" else float(hdl_selected)

            # Triglycerides with N/A option
            trig_options, trig_default_index = create_numeric_options(50, 1000, 1, "150")
            trig_selected = st.selectbox("Triglycerides (mg/dL)",
                                       options=trig_options,
                                       index=trig_default_index,
                                       key="lipid_trig_select")
            triglycerides = None if trig_selected == "N/A" else float(trig_selected)

            # Non-HDL Cholesterol with N/A option
            non_hdl_options, non_hdl_default_index = create_numeric_options(50, 200, 1, "110")
            non_hdl_selected = st.selectbox("Non-HDL Cholesterol (mg/dL)",
                                          options=non_hdl_options,
                                          index=non_hdl_default_index,
                                          key="lipid_non_hdl_select")
            non_hdl = None if non_hdl_selected == "N/A" else float(non_hdl_selected)

            # VLDL Cholesterol with N/A option
            vldl_options, vldl_default_index = create_numeric_options(20, 50, 1, "45")
            vldl_selected = st.selectbox("VLDL Cholesterol (mg/dL)",
                                       options=vldl_options,
                                       index=vldl_default_index,
                                       key="lipid_vldl_select")
            vldl = None if vldl_selected == "N/A" else float(vldl_selected)

        with col2:
            # Age with N/A option
            age_options, age_default_index = create_numeric_options(18, 120, 1, "40")
            age_selected = st.selectbox("Age",
                                      options=age_options,
                                      index=age_default_index,
                                      key="lipid_age_select")
            age = None if age_selected == "N/A" else int(age_selected)

            # Sex selection (no N/A option needed as it's already a dropdown)
            sex = st.selectbox("Sex", ["Male", "Female", "N/A"], index=0, key="lipid_sex")

            # Other risk factors with N/A option
            smoker = st.selectbox("Smoker", ["Yes", "No", "N/A"], index=1, key="lipid_smoker")
            hypertension = st.selectbox("Hypertension", ["Yes", "No", "N/A"], index=1, key="lipid_hypertension")
            diabetes = st.selectbox("Diabetes", ["Yes", "No", "N/A"], index=1, key="lipid_diabetes")
            family_history = st.selectbox("Family History of Heart Disease", ["Yes", "No", "N/A"], index=1, key="lipid_family_history")

        submit_button = st.form_submit_button(label="Analyze Lipid Profile")

        if submit_button:
            # Convert N/A selections to None for the backend
            sex_value = None if sex == "N/A" else sex
            smoker_value = None if smoker == "N/A" else smoker
            hypertension_value = None if hypertension == "N/A" else hypertension
            diabetes_value = None if diabetes == "N/A" else diabetes
            family_history_value = None if family_history == "N/A" else family_history

            lipid_data = {
                "total_chol": total_chol,
                "ldl": ldl,
                "hdl": hdl,
                "triglycerides": triglycerides,
                "non_hdl": non_hdl,
                "vldl": vldl,
                "age": age,
                "sex": sex_value,
                "smoker": smoker_value,
                "hypertension": hypertension_value,
                "diabetes": diabetes_value,
                "family_history": family_history_value
            }

            result = submit_lipid_profile(lipid_data)

            if "error" in result:
                response = f"Error analyzing lipid profile: {result['error']}"
            else:
                # Data is automatically saved on the server side in the submit_lipid_profile function

                # Format the response
                classification = result.get("classification", {})
                risk = result.get("ascvd_risk", "Unknown")
                recommendations = result.get("recommendations", [])
                formatted_recs = result.get("formatted_recommendations", [])
                ref_ranges = result.get("ref_ranges", {})

                # Create a formatted response
                response = "**Lipid Profile Analysis**\n\n"

                # Add classification results
                response += "**Results:**\n"
                for component, level in classification.items():
                    component_name = component.replace('_', ' ').title()
                    response += f"- {component_name}: {level.title()} "

                    # Add reference range if available
                    if component in ref_ranges:
                        ranges = ref_ranges[component]
                        if level in ranges:
                            response += f"({ranges[level]})"
                    response += "\n"

                # Add risk assessment
                response += f"\n**ASCVD Risk Assessment**: {risk}\n"

                # Add recommendations
                if formatted_recs:
                    response += "\n**Recommendations:**\n"
                    for rec in formatted_recs:
                        response += f"{rec}\n"
                elif recommendations:
                    response += "\n**Recommendations:**\n"
                    for rec in recommendations:
                        response += f"- {rec}\n"

                # Add note about saved data
                # response += "\n\n*Your lipid profile data has been saved on the server. You can ask for personalized recommendations anytime.*"

            st.session_state.message_log.append({"role": "user", "content": f"I've submitted my lipid profile test results for analysis: {json.dumps(lipid_data, indent=2)}"})
            st.session_state.message_log.append({"role": "ai", "content": response})
            st.session_state.waiting_for_lipid_profile = False
            st.rerun()

# Chat input
user_query = st.chat_input("Type your message here...")

# Process user input
if user_query:
    # Add user message to chat history
    st.session_state.message_log.append({"role": "user", "content": user_query})

    # Check if waiting for confirmation
    if st.session_state.waiting_for_confirmation and user_query.lower() == "yes":
        response = handle_confirmation(st.session_state.confirmation_type)
        st.session_state.message_log.append({"role": "ai", "content": response})
    # Check if user is asking for recommendations - let the server handle this
    elif any(keyword in user_query.lower() for keyword in ["recommendations", "what should i improve", "how can i improve", "what needs improvement", "vitals that need improvement", "advice", "suggest", "tips", "what should i do"]):
        # Let the server generate personalized recommendations based on saved health data
        response = query_agent(user_query, selected_model)
        st.session_state.message_log.append({"role": "ai", "content": response})
    else:
        # Get response from agent - the server will now handle including health data in responses
        response = query_agent(user_query, selected_model)
        st.session_state.message_log.append({"role": "ai", "content": response})

    # Rerun to update UI
    st.rerun()

# Footer
st.markdown("---")
st.caption(f"© {datetime.now().year} Dr. Deuce Health Assistant | Last updated: {datetime.now().strftime('%Y-%m-%d')}")
