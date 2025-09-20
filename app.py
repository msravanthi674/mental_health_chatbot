import streamlit as st
import requests
import uuid

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="AI Mental Health Chatbot",
    page_icon="💙"
)
st.title("AI Mental Health Chatbot")
st.markdown(
    """
    Hello! I'm here to help you navigate your thoughts and feelings. 
    You can chat with me, and I'll do my best to provide a supportive and understanding space. 
    In case of an emergency or a crisis, please use the provided helpline numbers.
    """
)

# --- Session State Management ---
# Generate a unique session ID for this user session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat History Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and API Interaction ---
if user_query := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare data for the API request
    payload = {
        "session_id": st.session_state.session_id,
        "query": user_query
    }
    
    # Send request to the FastAPI backend
    try:
        # Note: Replace with the correct backend URL if not running locally
        response = requests.post("http://127.0.0.1:8000/chat", json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        backend_response = response.json()["response"]

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": backend_response})
        with st.chat_message("assistant"):
            st.markdown(backend_response)

    except requests.exceptions.RequestException as e:
        error_message = f"**Error:** Could not connect to the backend. Please ensure the backend server is running. Error details: {e}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown(error_message)