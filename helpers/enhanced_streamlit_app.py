import streamlit as st
import time
import json
import logging
from datetime import datetime
from enhanced_chatbot import EnhancedPharmaceuticalChatbot
from main import MedicalTermExtractor

# Configure page
st.set_page_config(
    page_title="Enhanced Pharmaceutical Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 30px;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #2E86AB;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px 15px 5px 15px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 15px 15px 15px 5px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 20%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .emergency-alert {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #ff0000;
        animation: pulse 2s infinite;
        text-align: center;
        font-weight: bold;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 65, 108, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0); }
    }
    
    .analysis-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #00d2ff;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px;
    }
    
    .sidebar-section {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 10px 15px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2E86AB;
        box-shadow: 0 0 10px rgba(46, 134, 171, 0.3);
    }
    
    .suggestion-pill {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 5px;
        display: inline-block;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .suggestion-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_enhanced_chatbot():
    """Load the enhanced chatbot system"""
    try:
        chatbot = EnhancedPharmaceuticalChatbot()
        return chatbot
    except Exception as e:
        st.error(f"Failed to load enhanced chatbot: {e}")
        return None


def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = load_enhanced_chatbot()

    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}

    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {
            "queries_processed": 0,
            "drugs_mentioned": set(),
            "emergency_alerts": 0,
            "session_start": datetime.now()
        }

    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = True

    if "processing" not in st.session_state:
        st.session_state.processing = False


def display_main_header():
    """Display the main application header"""
    st.markdown('<h1 class="main-header">🤖 Enhanced Pharmaceutical Safety Assistant v2.0</h1>',
                unsafe_allow_html=True)

    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍</h3>
            <p><strong>Smart Analysis</strong><br>AI-powered drug interaction checking</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠</h3>
            <p><strong>Medical Knowledge</strong><br>Beyond database information</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🚨</h3>
            <p><strong>Emergency Detection</strong><br>Real-time safety assessment</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>👤</h3>
            <p><strong>Personalized</strong><br>Context-aware responses</p>
        </div>
        """, unsafe_allow_html=True)


def display_sidebar():
    """Enhanced sidebar with user profile and settings"""
    with st.sidebar:
        st.markdown("### 🎛️ **Control Panel**")

        # User Profile Section
        with st.expander("👤 **User Profile**", expanded=False):
            st.markdown("**Personalize your experience:**")

            age_group = st.selectbox(
                "Age Group:",
                ["Select...", "Under 18 (Pediatric)",
                 "18-65 (Adult)", "Over 65 (Elderly)"],
                help="Helps provide age-appropriate safety advice"
            )

            medical_conditions = st.text_area(
                "Medical Conditions:",
                placeholder="e.g., diabetes, hypertension, allergies...",
                help="Include relevant medical conditions for personalized advice"
            )

            pregnancy_status = st.selectbox(
                "Pregnancy/Breastfeeding:",
                ["Not applicable", "Pregnant", "Breastfeeding", "Planning pregnancy"]
            )

            detail_level = st.select_slider(
                "Response Detail Level:",
                options=["Simple", "Standard", "Detailed"],
                value="Standard",
                help="Adjust complexity of medical explanations"
            )

            if st.button("💾 Save Profile"):
                st.session_state.user_profile = {
                    "age_group": age_group,
                    "medical_conditions": medical_conditions,
                    "pregnancy_status": pregnancy_status,
                    "detail_level": detail_level
                }
                # Update chatbot profile
                if st.session_state.chatbot:
                    st.session_state.chatbot.session_context['user_profile'] = st.session_state.user_profile
                st.success("✅ Profile saved!")

        # Session Statistics
        with st.expander("📊 **Session Stats**", expanded=True):
            stats = st.session_state.session_stats

            st.metric("🔢 Queries Processed", stats["queries_processed"])
            st.metric("💊 Drugs Mentioned", len(stats["drugs_mentioned"]))
            st.metric("🚨 Emergency Alerts", stats["emergency_alerts"])

            session_duration = datetime.now() - stats["session_start"]
            st.metric("⏱️ Session Duration",
                      f"{session_duration.seconds // 60}m {session_duration.seconds % 60}s")

            if stats["drugs_mentioned"]:
                st.markdown("**Mentioned Drugs:**")
                for drug in list(stats["drugs_mentioned"])[:5]:
                    st.markdown(f"• {drug}")
                if len(stats["drugs_mentioned"]) > 5:
                    st.markdown(
                        f"• ... and {len(stats['drugs_mentioned']) - 5} more")

        # Settings
        with st.expander("⚙️ **Settings**", expanded=False):
            st.session_state.show_analysis = st.checkbox(
                "🔬 Show Analysis Details",
                value=st.session_state.show_analysis,
                help="Display technical analysis information"
            )

            show_timestamps = st.checkbox(
                "🕒 Show Timestamps",
                value=True,
                help="Display message timestamps"
            )

            enable_sound = st.checkbox(
                "🔊 Emergency Sound Alerts",
                value=True,
                help="Audio alerts for emergency situations"
            )

        # Quick Actions
        st.markdown("### 🚀 **Quick Actions**")

        if st.button("🔍 **Comprehensive Drug Check**", use_container_width=True):
            if len(st.session_state.session_stats["drugs_mentioned"]) >= 2:
                drugs_list = ", ".join(
                    list(st.session_state.session_stats["drugs_mentioned"]))
                st.session_state.quick_query = f"Check all interactions for: {drugs_list}"
            else:
                st.warning("⚠️ Mention at least 2 drugs first!")

        if st.button("🆘 **Emergency Help**", use_container_width=True):
            st.session_state.quick_query = "Emergency drug interaction help needed"

        if st.button("💊 **Drug Information**", use_container_width=True):
            st.session_state.quick_query = "Tell me about drug side effects and safety"

        if st.button("🔄 **Clear Chat**", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_stats = {
                "queries_processed": 0,
                "drugs_mentioned": set(),
                "emergency_alerts": 0,
                "session_start": datetime.now()
            }
            st.rerun()

        # Example Queries
        with st.expander("📝 **Example Queries**", expanded=False):
            examples = [
                "Can I take aspirin with warfarin?",
                "What are alternatives to ibuprofen?",
                "I'm pregnant - is acetaminophen safe?",
                "Emergency: took double dose of medication",
                "How does metformin work for diabetes?",
                "Check interactions: aspirin, warfarin, lisinopril"
            ]

            for example in examples:
                if st.button(f"📌 {example[:30]}...", key=f"example_{example[:20]}"):
                    st.session_state.quick_query = example


def display_chat_interface():
    """Enhanced chat interface with better formatting"""
    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>💬 You:</strong> {message['content']}
                    <br><small>🕒 {message.get('timestamp', '')}</small>
                </div>
                """, unsafe_allow_html=True)

            elif message["role"] == "assistant":
                # Check for emergency response
                if message.get('emergency', False):
                    st.markdown(f"""
                    <div class="emergency-alert">
                        🚨 <strong>EMERGENCY ALERT</strong> 🚨
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 Enhanced AI Assistant:</strong><br>
                    {message['content']}
                    <br><small>🕒 {message.get('timestamp', '')}</small>
                </div>
                """, unsafe_allow_html=True)

                # Display analysis if enabled and available
                if st.session_state.show_analysis and message.get('analysis'):
                    with st.expander("🔬 **Detailed Analysis**", expanded=False):
                        analysis = message['analysis']

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**🧪 Drug Analysis:**")
                            st.json({
                                "Drugs Identified": analysis.get('drugs', []),
                                "Intent": analysis.get('intent', 'Unknown'),
                                "User Level": analysis.get('user_level', 'Unknown')
                            })

                        with col2:
                            st.markdown("**🧠 AI Analysis:**")
                            st.json({
                                "Emotional State": analysis.get('emotional_state', 'Unknown'),
                                "Complexity": analysis.get('complexity', 'Unknown'),
                                "Emergency Level": analysis.get('emergency_level', 'LOW')
                            })

                        if analysis.get('interactions'):
                            st.markdown("**⚡ Interaction Summary:**")
                            st.write(
                                f"Found {len(analysis['interactions'])} interactions")
                            if analysis.get('serious_interactions', 0) > 0:
                                st.error(
                                    f"🚨 {analysis['serious_interactions']} serious interactions detected!")


def process_enhanced_query(query):
    """Process query using enhanced chatbot with comprehensive error handling"""
    try:
        # Add processing indicator
        with st.spinner("🧠 Processing with enhanced AI analysis..."):
            chatbot = st.session_state.chatbot

            if not chatbot:
                st.error("❌ Enhanced chatbot not available")
                return

            # Process the query (simulate the enhanced processing)
            start_time = time.time()

            # Basic processing
            result = chatbot.medical_extractor.process_query(query)

            if not result['success']:
                st.error(f"❌ Processing Error: {result['error']}")
                st.info("💡 Try rephrasing your query or check drug name spelling")
                return

            # Enhanced analysis (simulated)
            try:
                bert_analysis = chatbot.enhanced_agents.analyze_query_with_bert(
                    query)
            except Exception as e:
                bert_analysis = {"error": str(e)}

            try:
                advanced_analysis = chatbot.enhanced_agents.perform_advanced_query_analysis(
                    query)
            except Exception as e:
                advanced_analysis = {
                    "analysis_success": False, "error": str(e)}

            try:
                emergency_assessment = chatbot.enhanced_agents.assess_emergency_situation(
                    query, advanced_analysis, result['database_results']
                )
            except Exception as e:
                emergency_assessment = {
                    "assessment_success": False, "error": str(e)}

            # Check for unknown drugs and get medical knowledge
            unknown_drugs = [drug for drug in result['corrected_drugs']
                             if not _drug_in_database(drug, result)]

            medical_knowledge = ""
            if unknown_drugs:
                st.info(
                    f"🧠 Getting AI medical knowledge for: {', '.join(unknown_drugs)}")
                try:
                    for drug in unknown_drugs:
                        knowledge = chatbot.enhanced_agents.get_medical_knowledge_for_unknown_drugs(
                            drug, query
                        )
                        medical_knowledge += f"\n**{drug}:**\n{knowledge}\n"
                except Exception as e:
                    medical_knowledge = f"Error retrieving medical knowledge: {str(e)}"

            # Generate enhanced response
            try:
                enhanced_response = chatbot.enhanced_agents.generate_enhanced_contextual_response(
                    query=query,
                    intent=result['intent'],
                    drugs=result['corrected_drugs'],
                    db_results=result['database_results'],
                    query_analysis=advanced_analysis,
                    emergency_assessment=emergency_assessment,
                    session_context=chatbot.session_context
                )
            except Exception as e:
                enhanced_response = f"Enhanced response generation failed: {str(e)}"

            processing_time = time.time() - start_time

            # Build comprehensive response
            full_response = enhanced_response

            if medical_knowledge:
                full_response += f"\n\n🧠 **Additional Medical Knowledge:**\n{medical_knowledge}"

            full_response += f"\n\n⚡ *Processing completed in {processing_time:.2f} seconds*"

            # Check for emergency
            is_emergency = emergency_assessment.get('is_emergency', False)

            # Prepare analysis data
            analysis_data = {
                "drugs": result['corrected_drugs'],
                "intent": result['intent'],
                "user_level": bert_analysis.get('user_level', 'Unknown'),
                "emotional_state": advanced_analysis.get('emotional_state', 'Unknown'),
                "complexity": advanced_analysis.get('complexity_level', 'Unknown'),
                "emergency_level": emergency_assessment.get('urgency_level', 'LOW'),
                "interactions": result['database_results'].get('interactions', []),
                "serious_interactions": _count_serious_interactions(result['database_results'].get('interactions', []))
            }

            # Update session statistics
            st.session_state.session_stats["queries_processed"] += 1
            for drug in result['corrected_drugs']:
                st.session_state.session_stats["drugs_mentioned"].add(drug)

            if is_emergency:
                st.session_state.session_stats["emergency_alerts"] += 1

            # Add to chat history
            timestamp = datetime.now().strftime("%H:%M:%S")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": timestamp,
                "emergency": is_emergency,
                "analysis": analysis_data
            })

            # Update chatbot session context
            chatbot._update_session_context(query, result, advanced_analysis)

            # Show success metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"✅ Processed in {processing_time:.2f}s")
            with col2:
                st.info(f"🧪 Found {len(result['corrected_drugs'])} drugs")
            with col3:
                if result['database_results'].get('interactions'):
                    st.warning(
                        f"⚠️ {len(result['database_results']['interactions'])} interactions")
                else:
                    st.success("✅ No interactions found")

    except Exception as e:
        st.error(f"❌ Unexpected error during processing: {str(e)}")
        st.info("🔧 Please try again or contact support if the issue persists")


def _drug_in_database(drug, result):
    """Helper function to check if drug was found in database"""
    if result['intent'] == 'find_similar':
        return len(result['database_results'].get('similar_drugs', [])) > 0
    elif result['intent'] == 'check_interaction':
        return len(result['database_results'].get('interactions', [])) > 0
    return False


def _count_serious_interactions(interactions):
    """Helper function to count serious interactions"""
    serious_keywords = ['bleeding', 'severe',
                        'major', 'dangerous', 'death', 'fatal']
    return sum(1 for i in interactions
               if any(keyword in i.get('interaction_description', '').lower()
                      for keyword in serious_keywords))


def display_suggestion_pills():
    """Display interactive suggestion pills"""
    st.markdown("### 💡 **Quick Suggestions**")

    suggestions = [
        "Drug interaction check",
        "Find drug alternatives",
        "Emergency help",
        "Pregnancy safety",
        "Side effects info",
        "How drugs work"
    ]

    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(f"💊 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                if suggestion == "Drug interaction check":
                    st.session_state.quick_query = "Can I take aspirin with warfarin?"
                elif suggestion == "Find drug alternatives":
                    st.session_state.quick_query = "What are alternatives to ibuprofen?"
                elif suggestion == "Emergency help":
                    st.session_state.quick_query = "Emergency: I took too much medication"
                elif suggestion == "Pregnancy safety":
                    st.session_state.quick_query = "Is acetaminophen safe during pregnancy?"
                elif suggestion == "Side effects info":
                    st.session_state.quick_query = "What are the side effects of metformin?"
                elif suggestion == "How drugs work":
                    st.session_state.quick_query = "How do blood pressure medications work?"


def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()

    # Display header
    display_main_header()

    # Main layout
    col1, col2 = st.columns([3, 1])

    with col1:
        # Chat interface
        display_chat_interface()

        # Handle quick queries from sidebar
        if "quick_query" in st.session_state and st.session_state.quick_query:
            query = st.session_state.quick_query
            del st.session_state.quick_query

            # Add user message
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": timestamp
            })

            # Process query
            process_enhanced_query(query)
            st.rerun()

        # Chat input
        if prompt := st.chat_input("💬 Ask about drug interactions, alternatives, or safety information...", key="main_chat"):
            # Add user message
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": timestamp
            })

            # Process the query
            process_enhanced_query(prompt)
            st.rerun()

        # Suggestion pills
        if not st.session_state.messages:
            display_suggestion_pills()

    with col2:
        # Sidebar
        display_sidebar()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>🏥 <strong>Enhanced Pharmaceutical Safety Assistant</strong></p>
        <p>⚠️ This tool provides information only - always consult healthcare professionals</p>
        <p>🤖 Powered by AI • 🔒 Privacy Protected • 💊 Safety First</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
