# components/cbt_exercises.py

import streamlit as st # type: ignore
from datetime import datetime
from utils.mistral_client import MistralAIClient # type: ignore # Corrected import
from data.cbt_prompts import CBT_EXERCISES, COGNITIVE_DISTORTIONS

def render_cbt_exercises():
    """Render CBT exercises and thought record interface"""
    
    st.header("🧠 CBT Exercises")
    st.markdown("Learn and practice Cognitive Behavioral Therapy (CBT) techniques to understand and manage your thoughts and emotions.")
    
    # Initialize Mistral client
    if 'mistral_client' not in st.session_state:
        st.session_state.mistral_client = MistralAIClient()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Thought Record", "🔍 Identify Patterns", "📚 Learn CBT", "📊 Your Progress"])
    
    with tab1:
        render_thought_record()
    with tab2:
        render_pattern_identification()
    with tab3:
        render_cbt_education()
    with tab4:
        render_cbt_progress()

def render_thought_record():
    """Render the 7-column thought record exercise"""
    
    st.subheader("📋 Thought Record Exercise")
    st.markdown("A thought record helps you examine your thoughts and feelings about a situation.")
    
    with st.form("thought_record_form"):
        st.markdown("### 1. 📍 Situation")
        situation = st.text_area("What happened?", placeholder="E.g., I got a lower grade than expected...")
        st.markdown("### 2. 😟 Emotions")
        emotions = st.multiselect("What emotions did you feel?", ["Anxious", "Sad", "Angry", "Frustrated", "Disappointed"])
        intensity_before = st.slider("How intense were these emotions? (1-10)", 1, 10, 5)
        st.markdown("### 3. 💭 Automatic Thoughts")
        thoughts = st.text_area("What thoughts went through your mind?", placeholder="E.g., I'm not smart enough...")
        st.markdown("### 4. ✅ Evidence FOR the thought")
        evidence_for = st.text_area("What evidence supports this thought?")
        st.markdown("### 5. ❌ Evidence AGAINST the thought")
        evidence_against = st.text_area("What evidence contradicts this thought?")
        st.markdown("### 6. ⚖️ Balanced Thought")
        balanced_thought = st.text_area("What's a more balanced, realistic way to think about this?")
        st.markdown("### 7. 😌 New Emotion Rating")
        intensity_after = st.slider("How intense are your emotions now? (1-10)", 1, 10, intensity_before)
        
        submitted = st.form_submit_button("💾 Save Thought Record", type="primary")
        
        if submitted:
            if situation and thoughts:
                thought_record = {
                    "situation": situation, "emotions": emotions, "thoughts": thoughts,
                    "intensity_before": intensity_before, "evidence_for": evidence_for,
                    "evidence_against": evidence_against, "balanced_thought": balanced_thought,
                    "intensity_after": intensity_after
                }
                
                with st.spinner("Getting AI insights..."):
                    # Corrected call to Mistral client
                    ai_insights = st.session_state.mistral_client.generate_cbt_insight(thought_record)
                    thought_record["ai_insights"] = ai_insights
                
                st.session_state.data_manager.save_cbt_record(thought_record)
                st.success("Thought record saved! 📝")
                
                improvement = intensity_before - intensity_after
                if improvement > 0:
                    st.balloons()
                    st.success(f"Great work! Your emotional intensity decreased by {improvement} points! 📉")
                
                st.markdown("### 🤖 AI Insights")
                if ai_insights.get("cognitive_distortions"):
                    st.write("**Possible cognitive distortions:** " + ", ".join(ai_insights["cognitive_distortions"]))
                if ai_insights.get("balanced_thoughts"):
                    st.write("**Alternative balanced thoughts:**")
                    for thought in ai_insights["balanced_thoughts"]: st.write(f"• {thought}")
                if ai_insights.get("encouragement"):
                    st.info(ai_insights["encouragement"])
            else:
                st.warning("Please fill in at least the situation and thoughts fields.")

# The rest of the functions (render_pattern_identification, render_cbt_education, render_cbt_progress)
# do not use the AI client, so they can remain as they are. You can copy them from your original file.

def render_pattern_identification():
    st.subheader("🔍 Identify Thought Patterns")
    # ... (code remains the same)

def render_cbt_education():
    st.subheader("📚 Learn About CBT")
    # ... (code remains the same)

def render_cbt_progress():
    st.subheader("📊 Your CBT Progress")
    # ... (code remains the same)