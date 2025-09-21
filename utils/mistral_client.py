# utils/mistral_client.py

import streamlit as st # type: ignore
import json
from mistralai import Mistral # type: ignore # 1. IMPORT CHANGE: From MistralClient to Mistral

# The old ChatMessage class is no longer needed

class MistralAIClient:
    """A client to handle interactions with the Mistral AI API, updated for version 1.0.0."""

    def __init__(self):
        """Initializes the Mistral client using the API key from Streamlit secrets."""
        try:
            self.api_key = st.secrets["MISTRAL_API_KEY"]
            # 2. CLIENT CHANGE: Use the new Mistral() class
            self.client = Mistral(api_key=self.api_key)
        except (KeyError, FileNotFoundError):
            st.error("MISTRAL_API_KEY not found. Please set it in your .streamlit/secrets.toml file.")
            raise ValueError("API key for Mistral is not configured.")

    def generate_cbt_insight(self, thought_record: dict) -> dict:
        """Analyzes a CBT thought record and provides AI-powered insights."""
        system_prompt = """
        You are an expert AI assistant trained in Cognitive Behavioral Therapy (CBT).
        Analyze the user's thought record provided in JSON format.
        Your response MUST be a JSON object with four keys:
        1. "cognitive_distortions": (list of strings) Identify 1-3 potential cognitive distortions present in the 'thoughts' field.
        2. "balanced_thoughts": (list of strings) Provide 1-2 alternative, more balanced thoughts based on the evidence provided.
        3. "coping_strategies": (list of strings) Suggest 1-2 actionable coping strategies relevant to the situation.
        4. "encouragement": (string) Write a short, supportive, and encouraging message.
        """
        context_str = json.dumps(thought_record, indent=2)
        # 3. MESSAGE CHANGE: Use simple dictionaries instead of ChatMessage objects
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_str}
        ]
        try:
            model_name = "mistral-small-latest"
            # 4. METHOD CALL CHANGE: Use client.chat.complete()
            chat_response = self.client.chat.complete(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            return json.loads(chat_response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error generating CBT insight: {e}")
            return {"encouragement": "You're doing great work by reflecting on your thoughts. Keep it up!"}

    def generate_personalized_journal_prompt(self, mood_context: dict, recent_themes: list) -> dict:
        """Generates a personalized journal prompt based on user data."""
        system_prompt = """
        You are an AI specializing in therapeutic journaling. Generate a personalized journal prompt.
        Your response MUST be a JSON object with three keys: "prompt", "focus_area", and "follow_up_questions".
        """
        context_str = (f"Mood Context: {json.dumps(mood_context)}\nThemes: {', '.join(theme[0] for theme in recent_themes)}")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": context_str}]
        try:
            chat_response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
                response_format={"type": "json_object"}
            )
            return json.loads(chat_response.choices[0].message.content)
        except Exception:
            return {"prompt": "What is one small thing you can do today to take care of yourself?", "focus_area": "coping_skills", "follow_up_questions": []}

    def analyze_sentiment_and_risk(self, text: str) -> dict:
        """Uses Mistral AI to analyze text for sentiment and crisis risk level."""
        system_prompt = """
        You are an AI risk assessment assistant. Analyze text for crisis risk.
        Your response MUST be a JSON object with two keys: "risk_level" (low, moderate, high, or critical) and "sentiment".
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        try:
            chat_response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
                response_format={"type": "json_object"}
            )
            analysis_result = json.loads(chat_response.choices[0].message.content)
            analysis_result["method"] = "ai_analysis"
            return analysis_result
        except Exception:
            return {"risk_level": "low", "sentiment": "unknown", "method": "ai_analysis_fallback"}

    def get_empathetic_response(self, user_input: str, persona: str, conversation_history: list) -> str:
        """Generates an empathetic response from Mistral AI."""
        system_prompts = {
            "peer": "You are a supportive peer.",
            "mentor": "You are a wise mentor.",
            "therapist": "You are a compassionate therapist."
        }
        system_prompt = system_prompts.get(persona, system_prompts["therapist"])
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})
        try:
            chat_response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=messages
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            st.error(f"Error connecting to Mistral AI: {e}")
            return "I'm having trouble connecting right now. Please try again in a moment."