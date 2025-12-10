from agno.agent import Agent
from agno.models.google import Gemini
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import torch
from time import time

GEMINI_API_KEY = "AIzaSyD6eGB9TogKi-lWX5u2PY0528hWvSHZsgk"


class EnhancedContextualResponse(BaseModel):
    primary_response: str = Field(...,
                                  description="Main contextual response to user query")
    context_level: str = Field(
        ..., description="Response complexity: BEGINNER, INTERMEDIATE, ADVANCED, PROFESSIONAL")
    safety_warnings: List[str] = Field(
        ..., description="Specific safety warnings relevant to query")
    additional_insights: List[str] = Field(...,
                                           description="Extra relevant medical information")
    follow_up_questions: List[str] = Field(...,
                                           description="Suggested follow-up questions")
    confidence_score: float = Field(...,
                                    description="Response confidence 0.0-1.0")
    medical_disclaimer: str = Field(...,
                                    description="Appropriate medical disclaimer")


class MedicalKnowledgeResponse(BaseModel):
    drug_information: str = Field(...,
                                  description="Comprehensive drug information")
    mechanism_of_action: str = Field(..., description="How the drug works")
    common_side_effects: List[str] = Field(...,
                                           description="Common side effects")
    contraindications: List[str] = Field(...,
                                         description="Who should avoid this drug")
    special_populations: str = Field(
        ..., description="Pregnancy, elderly, pediatric considerations")
    lifestyle_considerations: str = Field(
        ..., description="Food, alcohol, activity interactions")


class EmergencyAssessment(BaseModel):
    is_emergency: bool = Field(...,
                               description="Whether immediate medical attention needed")
    urgency_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    immediate_actions: List[str] = Field(...,
                                         description="Steps to take right now")
    when_to_seek_help: str = Field(...,
                                   description="Specific triggers for medical help")
    emergency_contacts: str = Field(..., description="Who to contact and when")


class QueryAnalysis(BaseModel):
    medical_entities: List[str] = Field(...,
                                        description="All medical terms found")
    user_intent_detailed: str = Field(...,
                                      description="Detailed intent classification")
    complexity_level: str = Field(...,
                                  description="Query complexity assessment")
    urgency_indicators: List[str] = Field(...,
                                          description="Signs of urgency in query")
    emotional_state: str = Field(
        ..., description="User's emotional state: CALM, CONCERNED, ANXIOUS, PANICKED")


class EnhancedPharmaceuticalAgentSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_enhanced_agents()
        self._initialize_bert_models()
        self.session_context = {
            'user_profile': {},
            'conversation_history': [],
            'mentioned_drugs': set(),
            'medical_conditions': set(),
            'safety_alerts': []
        }

    def _initialize_enhanced_agents(self):
        """Initialize all enhanced specialized agents"""

        # Enhanced Contextual Response Agent
        self.contextual_response_agent = Agent(
            name="Enhanced Contextual Response Generator",
            role="Generate intelligent, context-aware pharmaceutical responses",
            model=Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY),
            description="""Advanced pharmaceutical response generator that:
            
            CONTEXT ANALYSIS:
            - Analyzes user's medical literacy level from query complexity
            - Considers conversation history and session context
            - Incorporates safety warnings based on interaction severity
            - Adapts response style to user needs
            
            RESPONSE GENERATION:
            - Provides clear, actionable medical information
            - Includes appropriate safety warnings and disclaimers
            - Suggests relevant follow-up questions
            - Maintains professional medical standards
            
            PERSONALIZATION:
            - Adjusts technical detail based on user level
            - Considers previous questions and context
            - Provides graduated safety warnings
            - Includes relevant additional insights""",
            response_model=EnhancedContextualResponse,
            markdown=True
        )

        # Medical Knowledge Agent for non-database queries
        self.medical_knowledge_agent = Agent(
            name="Medical Knowledge Expert",
            role="Provide comprehensive medical information beyond database scope",
            model=Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY),
            description="""Expert medical knowledge agent providing:
            
            DRUG INFORMATION:
            - Mechanism of action and pharmacology
            - Comprehensive side effect profiles
            - Contraindications and warnings
            - Special population considerations
            
            CLINICAL GUIDANCE:
            - Dosing information and schedules
            - Monitoring requirements
            - Drug class information
            - Therapeutic alternatives
            
            SAFETY INFORMATION:
            - Food and alcohol interactions
            - Activity restrictions
            - Signs of adverse reactions
            - When to contact healthcare providers""",
            response_model=MedicalKnowledgeResponse,
        )

        # Emergency Assessment Agent
        self.emergency_assessment_agent = Agent(
            name="Emergency Medical Assessment",
            role="Assess medical emergency situations and provide guidance",
            model=Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY),
            description="""Emergency assessment specialist that:
            
            EMERGENCY DETECTION:
            - Identifies life-threatening drug interactions
            - Recognizes overdose symptoms and signs
            - Detects allergic reaction indicators
            - Assesses severity of reported symptoms
            
            RESPONSE PROTOCOLS:
            - Provides immediate action steps
            - Determines appropriate level of care needed
            - Guides to emergency services when necessary
            - Offers poison control contact information
            
            SAFETY GUIDANCE:
            - Clear, step-by-step emergency instructions
            - Prioritizes most critical actions first
            - Provides reassurance while ensuring safety""",
            response_model=EmergencyAssessment,
        )

        # Advanced Query Analysis Agent
        self.query_analysis_agent = Agent(
            name="Advanced Query Analyzer",
            role="Perform deep analysis of pharmaceutical queries",
            model=Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY),
            description="""Advanced query analysis system that:
            
            LINGUISTIC ANALYSIS:
            - Extracts all medical entities and terms
            - Identifies implicit medical concerns
            - Analyzes query complexity and sophistication
            - Detects emotional state and urgency
            
            INTENT CLASSIFICATION:
            - Expands beyond basic interaction/similarity checking
            - Identifies specific medical concerns
            - Recognizes emergency situations
            - Classifies user expertise level
            
            CONTEXT BUILDING:
            - Builds comprehensive query context
            - Identifies missing information needs
            - Suggests clarifying questions
            - Assesses information completeness""",
            response_model=QueryAnalysis,
        )

    def _initialize_bert_models(self):
        """Initialize BERT models for enhanced NLP capabilities"""
        try:
            # BioBERT for medical entity recognition
            self.biobert_tokenizer = AutoTokenizer.from_pretrained(
                'dmis-lab/biobert-base-cased-v1.1')
            self.biobert_model = AutoModel.from_pretrained(
                'dmis-lab/biobert-base-cased-v1.1')

            # Clinical BERT for medical context understanding
            self.clinical_bert_tokenizer = AutoTokenizer.from_pretrained(
                'emilyalsentzer/Bio_ClinicalBERT')
            self.clinical_bert_model = AutoModel.from_pretrained(
                'emilyalsentzer/Bio_ClinicalBERT')

            print("✅ Enhanced BERT models loaded successfully")
        except Exception as e:
            self.logger.error(f"BERT model loading failed: {e}")
            self.biobert_model = None
            self.clinical_bert_model = None

    def analyze_query_with_bert(self, query: str) -> Dict[str, Any]:
        """Enhanced query analysis using BERT models"""
        if not self.biobert_model or not self.clinical_bert_model:
            return {"error": "BERT models not available"}

        try:
            # BioBERT for medical entity extraction
            bio_inputs = self.biobert_tokenizer(
                query, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                bio_outputs = self.biobert_model(**bio_inputs)
                bio_embeddings = bio_outputs.last_hidden_state.mean(dim=1)

            # Clinical BERT for context understanding
            clinical_inputs = self.clinical_bert_tokenizer(
                query, return_tensors="pt", truncation=True)
            with torch.no_grad():
                clinical_outputs = self.clinical_bert_model(**clinical_inputs)
                clinical_embeddings = clinical_outputs.last_hidden_state.mean(
                    dim=1)

            # Simple medical entity detection (can be enhanced with proper NER)
            medical_keywords = [
                'drug', 'medication', 'pill', 'tablet', 'capsule', 'dose', 'dosage',
                'side effect', 'interaction', 'allergy', 'reaction', 'symptoms',
                'overdose', 'emergency', 'urgent', 'pain', 'bleeding', 'nausea'
            ]

            found_entities = [
                keyword for keyword in medical_keywords if keyword in query.lower()]

            # Urgency detection
            urgency_keywords = ['emergency', 'urgent', 'help',
                                'immediately', 'now', 'asap', 'critical']
            urgency_score = sum(
                1 for keyword in urgency_keywords if keyword in query.lower())

            # Complexity assessment
            complex_terms = ['contraindication', 'pharmacokinetics',
                             'bioavailability', 'metabolism', 'cytochrome']
            complexity_score = sum(
                1 for term in complex_terms if term in query.lower())

            return {
                'medical_entities': found_entities,
                'urgency_score': urgency_score,
                'complexity_score': complexity_score,
                'bio_embedding_norm': float(bio_embeddings.norm().item()),
                'clinical_embedding_norm': float(clinical_embeddings.norm().item()),
                'user_level': 'ADVANCED' if complexity_score > 0 else 'INTERMEDIATE' if len(found_entities) > 2 else 'BEGINNER'
            }

        except Exception as e:
            self.logger.error(f"BERT analysis failed: {e}")
            return {"error": str(e)}

    def perform_advanced_query_analysis(self, query: str) -> Dict[str, Any]:
        """Perform comprehensive query analysis using Gemini agent"""
        try:
            analysis_prompt = f"""
            COMPREHENSIVE PHARMACEUTICAL QUERY ANALYSIS
            
            Query: "{query}"
            
            Perform detailed analysis including:
            1. Extract ALL medical entities (drugs, conditions, symptoms)
            2. Classify detailed user intent beyond basic categories
            3. Assess query complexity and user medical literacy level
            4. Identify urgency indicators and emotional state
            5. Detect any emergency or safety concerns
            
            Provide comprehensive analysis for context-aware response generation.
            """

            response = self.query_analysis_agent.run(analysis_prompt)

            if hasattr(response, 'content'):
                return {
                    'medical_entities': response.content.medical_entities,
                    'detailed_intent': response.content.user_intent_detailed,
                    'complexity_level': response.content.complexity_level,
                    'urgency_indicators': response.content.urgency_indicators,
                    'emotional_state': response.content.emotional_state,
                    'analysis_success': True
                }

            return {'analysis_success': False, 'error': 'No response content'}

        except Exception as e:
            self.logger.error(f"Advanced query analysis failed: {e}")
            return {'analysis_success': False, 'error': str(e)}

    def get_medical_knowledge_for_unknown_drugs(self, drug_name: str, specific_question: str = None) -> str:
        """Get medical knowledge for drugs not in database using Gemini"""
        try:
            knowledge_prompt = f"""
            MEDICAL KNOWLEDGE REQUEST
            
            Drug: {drug_name}
            Specific Question: {specific_question or "General information"}
            
            Provide comprehensive medical information including:
            1. What this medication is used for (indications)
            2. How it works (mechanism of action)
            3. Common side effects and adverse reactions
            4. Important contraindications and warnings
            5. Special considerations for different populations
            6. Food, alcohol, and lifestyle interactions
            
            Focus on safety, accuracy, and practical clinical information.
            Include appropriate medical disclaimers.
            """

            response = self.medical_knowledge_agent.run(knowledge_prompt)

            if hasattr(response, 'content'):
                # Build comprehensive response
                knowledge_response = f"**{drug_name} - Medical Information**\n\n"
                knowledge_response += f"**Primary Uses:** {response.content.drug_information}\n\n"
                knowledge_response += f"**How it Works:** {response.content.mechanism_of_action}\n\n"

                if response.content.common_side_effects:
                    knowledge_response += f"**Common Side Effects:**\n"
                    for effect in response.content.common_side_effects:
                        knowledge_response += f"• {effect}\n"
                    knowledge_response += "\n"

                if response.content.contraindications:
                    knowledge_response += f"**⚠️ Important Warnings:**\n"
                    for warning in response.content.contraindications:
                        knowledge_response += f"• {warning}\n"
                    knowledge_response += "\n"

                knowledge_response += f"**Special Populations:** {response.content.special_populations}\n\n"
                knowledge_response += f"**Lifestyle Considerations:** {response.content.lifestyle_considerations}\n\n"

                return knowledge_response

            return f"Unable to retrieve detailed information about {drug_name}"

        except Exception as e:
            self.logger.error(f"Medical knowledge retrieval failed: {e}")
            return f"Error retrieving information about {drug_name}: {str(e)}"

    def assess_emergency_situation(self, query: str, query_analysis: Dict, db_results: Dict) -> Dict[str, Any]:
        """Assess if query indicates emergency situation"""
        try:
            # Check for emergency indicators
            emergency_keywords = ['overdose', 'poisoning',
                                  'allergic reaction', 'emergency', 'urgent help']
            urgency_indicators = query_analysis.get('urgency_indicators', [])
            emotional_state = query_analysis.get('emotional_state', 'CALM')

            # Check database results for critical interactions
            interactions = db_results.get('interactions', [])
            critical_interactions = [
                i for i in interactions
                if any(word in i.get('interaction_description', '').lower()
                       for word in ['death', 'fatal', 'life-threatening', 'emergency', 'critical'])
            ]

            emergency_prompt = f"""
            EMERGENCY MEDICAL ASSESSMENT
            
            User Query: "{query}"
            Emergency Keywords Detected: {any(keyword in query.lower() for keyword in emergency_keywords)}
            Urgency Indicators: {urgency_indicators}
            User Emotional State: {emotional_state}
            Critical Interactions Found: {len(critical_interactions)}
            
            Critical Interaction Details:
            {critical_interactions[:2] if critical_interactions else 'None detected'}
            
            Assess:
            1. Is this a medical emergency requiring immediate attention?
            2. What is the appropriate urgency level?
            3. What immediate actions should the user take?
            4. When and who should they contact for help?
            5. What emergency resources are most appropriate?
            
            Provide clear, actionable emergency guidance.
            """

            response = self.emergency_assessment_agent.run(emergency_prompt)

            if hasattr(response, 'content'):
                return {
                    'is_emergency': response.content.is_emergency,
                    'urgency_level': response.content.urgency_level,
                    'immediate_actions': response.content.immediate_actions,
                    'when_to_seek_help': response.content.when_to_seek_help,
                    'emergency_contacts': response.content.emergency_contacts,
                    'assessment_success': True
                }

            return {'assessment_success': False, 'is_emergency': False}

        except Exception as e:
            self.logger.error(f"Emergency assessment failed: {e}")
            return {'assessment_success': False, 'is_emergency': False, 'error': str(e)}

    def generate_enhanced_contextual_response(self,
                                              query: str,
                                              intent: str,
                                              drugs: List[str],
                                              db_results: Dict[str, Any],
                                              query_analysis: Dict[str, Any],
                                              emergency_assessment: Dict[str, Any],
                                              session_context: Dict[str, Any] = None) -> str:
        """Generate the most advanced contextual response possible"""
        try:
            # Build comprehensive context for response generation
            comprehensive_context = f"""
            ENHANCED CONTEXTUAL PHARMACEUTICAL RESPONSE GENERATION
            
            === USER QUERY ===
            Original Query: "{query}"
            User Intent: {intent}
            Mentioned Drugs: {drugs}
            
            === ADVANCED ANALYSIS ===
            User Medical Literacy: {query_analysis.get('complexity_level', 'UNKNOWN')}
            Emotional State: {query_analysis.get('emotional_state', 'CALM')}
            Urgency Indicators: {query_analysis.get('urgency_indicators', [])}
            Medical Entities Found: {query_analysis.get('medical_entities', [])}
            
            === DATABASE RESULTS ===
            {self._format_database_results_for_context(intent, db_results)}
            
            === EMERGENCY ASSESSMENT ===
            Emergency Status: {emergency_assessment.get('is_emergency', False)}
            Urgency Level: {emergency_assessment.get('urgency_level', 'LOW')}
            
            === SESSION CONTEXT ===
            Previous Drugs Mentioned: {list(self.session_context.get('mentioned_drugs', set()))}
            Conversation History: {len(self.session_context.get('conversation_history', []))} previous interactions
            User Profile: {session_context or 'No specific profile'}
            
            === RESPONSE REQUIREMENTS ===
            1. Generate response appropriate for user's medical literacy level
            2. Include specific safety warnings based on emergency assessment
            3. Provide actionable, clear medical information
            4. Consider conversation history and context
            5. Include appropriate follow-up suggestions
            6. Add proper medical disclaimers
            7. Use emojis and formatting for clarity
            8. Maintain professional medical standards
            
            Generate a comprehensive, contextually intelligent pharmaceutical response.
            """

            response = self.contextual_response_agent.run(
                comprehensive_context)

            if hasattr(response, 'content'):
                # Build final enhanced response
                final_response = ""

                # Add emergency alert if needed
                if emergency_assessment.get('is_emergency', False):
                    final_response += "🚨 **EMERGENCY ALERT** 🚨\n\n"
                    for action in emergency_assessment.get('immediate_actions', []):
                        final_response += f"• {action}\n"
                    final_response += f"\n📞 {emergency_assessment.get('emergency_contacts', '')}\n\n"

                # Add main response
                final_response += response.content.primary_response + "\n\n"

                # Add safety warnings if any
                if response.content.safety_warnings:
                    final_response += "⚠️ **Important Safety Information:**\n"
                    for warning in response.content.safety_warnings:
                        final_response += f"• {warning}\n"
                    final_response += "\n"

                # Add additional insights
                if response.content.additional_insights:
                    final_response += "💡 **Additional Information:**\n"
                    for insight in response.content.additional_insights:
                        final_response += f"• {insight}\n"
                    final_response += "\n"

                # Add follow-up questions
                if response.content.follow_up_questions:
                    final_response += "🤔 **You might also want to ask:**\n"
                    for question in response.content.follow_up_questions:
                        final_response += f"• {question}\n"
                    final_response += "\n"

                # Add medical disclaimer
                final_response += f"⚕️ **Medical Disclaimer:** {response.content.medical_disclaimer}\n"

                # Add confidence indicator
                confidence = response.content.confidence_score
                if confidence < 0.7:
                    final_response += f"\n🔍 **Note:** Response confidence: {confidence:.1%} - Consider consulting a healthcare professional for more specific guidance."

                return final_response

            return "Unable to generate enhanced contextual response"

        except Exception as e:
            self.logger.error(
                f"Enhanced contextual response generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def _format_database_results_for_context(self, intent: str, db_results: Dict[str, Any]) -> str:
        """Format database results for agent context"""
        if intent == 'check_interaction':
            interactions = db_results.get('interactions', [])
            if interactions:
                formatted = f"Database Interaction Results ({len(interactions)} found):\n"
                for i, interaction in enumerate(interactions[:3], 1):
                    drug1 = interaction.get(
                        'entity1', {}).get('name', 'Unknown')
                    drug2 = interaction.get(
                        'entity2', {}).get('name', 'Unknown')
                    desc = interaction.get(
                        'interaction_description', 'No description')
                    formatted += f"  {i}. {drug1} ↔ {drug2}: {desc[:150]}...\n"
                if len(interactions) > 3:
                    formatted += f"  ... and {len(interactions) - 3} more interactions\n"
                return formatted
            else:
                return "Database Results: No interactions found"

        elif intent == 'find_similar':
            similar_drugs = db_results.get('similar_drugs', [])
            if similar_drugs:
                formatted = f"Database Similarity Results ({len(similar_drugs)} found):\n"
                for i, drug in enumerate(similar_drugs[:5], 1):
                    name = drug.get('entity_name', 'Unknown')
                    score = drug.get('similarity_score', 0)
                    formatted += f"  {i}. {name} (similarity: {score:.3f})\n"
                return formatted
            else:
                return "Database Results: No similar drugs found"

        return "Database Results: No relevant data available"

    def update_session_context(self, query: str, drugs: List[str], intent: str):
        """Update session context with new information"""
        # Add to conversation history
        self.session_context['conversation_history'].append({
            'query': query,
            'intent': intent,
            'drugs': drugs,
            'timestamp': time.time()
        })

        # Update mentioned drugs
        self.session_context['mentioned_drugs'].update(drugs)

        # Keep only last 10 conversations
        if len(self.session_context['conversation_history']) > 10:
            self.session_context['conversation_history'] = self.session_context['conversation_history'][-10:]

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session context"""
        return {
            'total_queries': len(self.session_context['conversation_history']),
            'unique_drugs_mentioned': len(self.session_context['mentioned_drugs']),
            'drugs_list': list(self.session_context['mentioned_drugs']),
            'recent_intents': [conv['intent'] for conv in self.session_context['conversation_history'][-5:]],
            'safety_alerts_issued': len(self.session_context['safety_alerts'])
        }
