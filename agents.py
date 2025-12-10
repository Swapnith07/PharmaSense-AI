from agno.agent import Agent
from agno.models.google import Gemini
from typing import List, Dict, Any
import logging
from pydantic import BaseModel, Field

GEMINI_API_KEY = "AIzaSyBHiDJHNXqXU_q2JLq_mNma20UO0UOVq2Q"


class DrugExtractionResponse(BaseModel):
    drugs: List[str] = Field(
        ..., description="List of drug names found in the query. Return empty list if no drugs found.")


class IntentClassificationResponse(BaseModel):
    intent: str = Field(..., description="Classification of user intent. Must be one of: check_interaction, find_similar, general_query")
    confidence: float = Field(...,
                              description="Confidence score between 0.0 and 1.0")


class PharmaceuticalResponse(BaseModel):
    response: str = Field(
        ..., description="Safe, informative pharmaceutical response with medical disclaimers")
    severity_level: str = Field(
        ..., description="Safety level: SAFE, CAUTION, MAJOR_INTERACTION, UNKNOWN, or INFO")
    disclaimer_included: bool = Field(...,
                                      description="Whether medical disclaimer is included")


class PharmaceuticalAgentSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("google_genai.models").setLevel(logging.WARNING)

        # NER Agent for drug extraction
        self.ner_agent = Agent(
            name="Medical NER Agent",
            role="Extract drug names from pharmaceutical queries",
            model=Gemini(id="gemini-2.5-flash", api_key=GEMINI_API_KEY),
            description="Extract ALL drug names, active ingredients, and medication names from user queries. Include brand names, generic names, and chemical compounds.",
            response_model=DrugExtractionResponse,
        )

        # Enhanced Intent Classifier
        # - now supports 3 intents
        self.intent_agent = Agent(
            name="Pharmaceutical Intent Classifier",
            role="Classify pharmaceutical queries into three categories with high accuracy",
            model=Gemini(id="gemini-2.5-flash", api_key=GEMINI_API_KEY),
            description="""You are an expert pharmaceutical intent classifier. Analyze the user query and classify into exactly ONE category:

**PRIORITY CLASSIFICATION RULES:**

1. **'check_interaction'** - HIGHEST PRIORITY when query involves:
   - Multiple drugs mentioned (2 or more)
   - Keywords: 'can I take...with', 'take...together', 'combine', 'mix', 'safe to take', 'interaction', 'together with'
   - Question format asking about safety of drug combinations
   - Phrases like: "Can I take X with Y?", "Is it safe to combine X and Y?", "X and Y together"
   
2. **'find_similar'** - When query asks for alternatives:
   - Single drug mentioned typically
   - Keywords: 'similar to', 'alternative to', 'like', 'substitute for', 'replace', 'equivalent', 'instead of'
   - Phrases like: "What is similar to X?", "Alternatives to X", "Replace X with"
   
3. **'general_query'** - Educational information requests:
   - Single drug mentioned usually
   - Keywords: 'what is', 'tell me about', 'how does X work', 'side effects of', 'used for'
   - NO mention of combining with other drugs
   - Phrases like: "What is X?", "Tell me about X", "How does X work?"

**CRITICAL DECISION FACTORS:**
- If 2+ drugs mentioned AND asking about taking them = check_interaction
- Words "with", "and", "together" between drug names = check_interaction  
- "Can I take" + multiple drugs = check_interaction (NOT general_query)

**EXAMPLES:**
✅ "Can I take Lepirudin with Apixaban?" → check_interaction (2 drugs + "with")
✅ "Is Aspirin safe with Warfarin?" → check_interaction (2 drugs + safety question)
✅ "What drugs are similar to Metformin?" → find_similar (alternatives requested)
✅ "What is Lepirudin used for?" → general_query (single drug info)
✅ "Side effects of Aspirin" → general_query (educational)
✅ "Combinational effets of considering Lepirudin?" → check_interaction (understanding interaction)


**PRIORITY ORDER:** check_interaction > find_similar > general_query""",
            response_model=IntentClassificationResponse,
        )

        # Enhanced Response Generator with better formatting
        self.response_agent = Agent(
            name="Pharmaceutical Response Specialist",
            role="Generate comprehensive, safe pharmaceutical responses",
            model=Gemini(id="gemini-2.5-flash", api_key=GEMINI_API_KEY),
            description="""Generate safe, comprehensive pharmaceutical responses with:
            
            RESPONSE STRUCTURE:
            1. **Clear Summary** with appropriate emoji (✅🚨⚠️💊)
            2. **Specific Details** from database findings
            3. **Risk Assessment** when applicable
            4. **Actionable Recommendations**
            5. **Medical Disclaimer** (always required)
            
            SAFETY LEVELS:
            - SAFE ✅: No known issues, generally safe
            - CAUTION ⚠️: Monitor closely, potential mild interactions
            - MAJOR_INTERACTION 🚨: Serious interaction risk, avoid combination
            - INFO 💊: General information provided
            - UNKNOWN ❓: Insufficient data, consult healthcare provider
            
            FORMATTING:
            - Use clear headings with **bold text**
            - Include relevant emojis for visual clarity
            - Provide numbered lists for multiple items
            - Always end with medical disclaimer
            - Keep language accessible but professional""",
            response_model=PharmaceuticalResponse,
            markdown=True
        )

    def extract_drugs(self, query: str) -> List[str]:
        try:
            prompt = f"Extract all drug names from: '{query}'"
            response = self.ner_agent.run(prompt)

            if hasattr(response, 'content') and hasattr(response.content, 'drugs'):
                return [drug.lower().strip() for drug in response.content.drugs if drug.strip()]
            return []
        except Exception as e:
            self.logger.error(f"Drug extraction error: {e}")
            return []

    def classify_intent(self, query: str, drugs: List[str]) -> str:
        try:
            # Pre-analysis for better context
            query_lower = query.lower()
            drug_count = len(drugs)

            # Strong interaction indicators
            interaction_patterns = [
                'can i take', 'can you take', 'safe to take', 'safe to combine',
                'take with', 'take together', 'together with', 'combine with',
                'mix with', 'interaction', 'dangerous to take',
            ]

            # Check for interaction patterns
            has_interaction_pattern = any(
                pattern in query_lower for pattern in interaction_patterns)
            has_connecting_words = any(word in query_lower for word in [
                                       ' with ', ' and ', ' together',])

            context = f"""
            QUERY ANALYSIS:
            Original Query: '{query}'
            Identified Drugs: {drugs}
            Number of Drugs: {drug_count}
            
            PATTERN ANALYSIS:
            - Has interaction keywords: {has_interaction_pattern}
            - Has connecting words (with/and): {has_connecting_words}
            - Multiple drugs present: {drug_count >= 2}
            
            CLASSIFICATION LOGIC:
            If multiple drugs (2+) AND (interaction keywords OR connecting words) → check_interaction
            If asking for alternatives/similar → find_similar  
            If single drug info request → general_query
            
            Apply the priority rules strictly. Multiple drugs with connecting words = interaction query.
            """

            response = self.intent_agent.run(context)

            if hasattr(response, 'content') and hasattr(response.content, 'intent'):
                classified_intent = response.content.intent

                # Additional validation logic as backup
                if drug_count >= 2 and (has_interaction_pattern or has_connecting_words):
                    if classified_intent != "check_interaction":
                        self.logger.warning(
                            f"Intent correction: {classified_intent} → check_interaction for query: {query}")
                        return "check_interaction"

                return classified_intent
            return "general_query"

        except Exception as e:
            self.logger.error(f"Intent classification error: {e}")
            return "general_query"

    def generate_response(self, intent: str, drugs: List[str], db_results: Dict[str, Any]) -> str:
        try:
            # Build comprehensive context based on intent
            if intent == "check_interaction":
                context = self._build_comprehensive_interaction_context(
                    drugs, db_results)
            elif intent == "find_similar":
                context = self._build_comprehensive_similarity_context(
                    drugs, db_results)
            else:  # general_query
                context = self._build_general_query_context(drugs, db_results)

            # Enhanced prompt for better responses
            prompt = f"""
            PHARMACEUTICAL QUERY ANALYSIS:
            Intent: {intent}
            
            {context}
            
            INSTRUCTIONS:
            Generate a comprehensive, safe pharmaceutical response following the structured format.
            Include appropriate safety level assessment and always end with medical disclaimer.
            Use clear formatting with emojis and bold headings for readability.
            """

            response = self.response_agent.run(prompt)

            if hasattr(response, 'content') and hasattr(response.content, 'response'):
                return response.content.response
            return "Unable to generate response"
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            return "Error generating response. Please consult with a healthcare professional."

    def _build_comprehensive_interaction_context(self, drugs: List[str], db_results: Dict[str, Any]) -> str:
        """Build detailed context for interaction checking"""
        interactions = db_results.get('interactions', [])

        context = f"""
        DRUG INTERACTION ANALYSIS:
        Drugs being checked: {', '.join(drugs)} ({len(drugs)} total)
        
        DATABASE FINDINGS:
        """

        if not interactions:
            context += f"""
            - No direct interactions found between {', '.join(drugs)}
            - Database searched: {len(drugs)} drugs cross-referenced
            - This suggests potential safety for concurrent use
            
            IMPORTANT CONSIDERATIONS:
            1. Absence of data doesn't guarantee safety
            2. Individual patient factors may still create risks
            3. Timing of administration may be important
            4. Dosage adjustments might be needed
            """
        else:
            # Categorize interactions by severity
            bleeding_interactions = [i for i in interactions if any(word in i.get('interaction_description', '').lower()
                                                                    for word in ['bleeding', 'hemorrhage', 'anticoagulant'])]
            severe_interactions = [i for i in interactions if any(word in i.get('interaction_description', '').lower()
                                                                  for word in ['severe', 'contraindicated', 'avoid', 'dangerous'])]

            context += f"""
            - TOTAL INTERACTIONS FOUND: {len(interactions)}
            - Bleeding-related interactions: {len(bleeding_interactions)}
            - Severe/contraindicated interactions: {len(severe_interactions)}
            
            DETAILED INTERACTION ANALYSIS:
            """

            for i, interaction in enumerate(interactions[:5], 1):  # Show top 5
                drug1 = interaction.get('entity1', {}).get('name', 'Unknown')
                drug2 = interaction.get('entity2', {}).get('name', 'Unknown')
                desc = interaction.get(
                    'interaction_description', 'No description available')

                context += f"""
            {i}. {drug1} ↔ {drug2}
               Description: {desc}
               """

            if len(interactions) > 5:
                context += f"\n   ... and {len(interactions) - 5} additional interactions found"

        return context

    def _build_comprehensive_similarity_context(self, drugs: List[str], db_results: Dict[str, Any]) -> str:
        """Build detailed context for drug similarity"""
        similar_drugs = db_results.get('similar_drugs', [])
        query_drug = drugs[0] if drugs else 'unknown'

        context = f"""
        DRUG SIMILARITY ANALYSIS:
        Target drug: {query_drug}
        
        DATABASE FINDINGS:
        """

        if not similar_drugs:
            context += f"""
            - No similar drugs found for '{query_drug}'
            - This could indicate:
              1. Unique mechanism of action
              2. Specialized therapeutic use  
              3. Limited database coverage
              4. Potential misspelling of drug name
            
            RECOMMENDATIONS:
            - Verify correct drug spelling
            - Consult healthcare provider for alternatives
            - Consider therapeutic class substitutions
            """
        else:
            # Filter out the input drug and limit to 5
            filtered_drugs = []
            input_drug_lower = query_drug.lower().strip()

            for drug in similar_drugs:
                drug_name = drug.get('entity_name', '').lower().strip()
                similarity = drug.get('similarity_score', 0)

                if (drug_name != input_drug_lower and
                    similarity < 0.98 and
                        len(filtered_drugs) < 5):
                    filtered_drugs.append(drug)

            if not filtered_drugs:
                context += f"""
                - No alternative drugs found for '{query_drug}'
                - All similar entries were the same drug
                """
            else:
                context += f"""
                - TOTAL ALTERNATIVE DRUGS FOUND: {len(filtered_drugs)}
                
                TOP 5 ALTERNATIVES:
                """

                for i, drug in enumerate(filtered_drugs, 1):
                    name = drug.get('entity_name', 'Unknown')
                    score = drug.get('similarity_score', 0)
                    similarity_percent = score * 100

                    if score > 0.8:
                        level = "Very High"
                    elif score > 0.6:
                        level = "Moderate"
                    else:
                        level = "Lower"

                    context += f"""
            {i}. {name}
               Similarity: {similarity_percent:.1f}% ({level})
               """

        return context

    def _build_general_query_context(self, drugs: List[str], db_results: Dict[str, Any]) -> str:
        """Build context for general pharmaceutical queries"""
        context = f"""
        GENERAL PHARMACEUTICAL QUERY:
        Drugs mentioned: {', '.join(drugs) if drugs else 'None specified'}
        
        AVAILABLE INFORMATION:
        """

        if not drugs:
            context += """
            - No specific drugs identified in query
            - Providing general pharmaceutical guidance
            - Response will focus on educational information
            """
        else:
            # Check if we have any database info
            interactions = db_results.get('interactions', [])
            similar_drugs = db_results.get('similar_drugs', [])

            context += f"""
            - Primary drug focus: {drugs[0] if drugs else 'N/A'}
            - Related interactions in database: {len(interactions)}
            - Similar drugs available: {len(similar_drugs)}
            
            CONTEXT FOR RESPONSE:
            - Provide educational information about the drug(s)
            - Include general safety considerations
            - Mention common uses and precautions
            - Avoid specific medical advice
            """

        return context

    def process_query(self, query: str, db_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main method to process queries through the agent pipeline"""

        try:
            # Step 1: Extract drugs using NER agent
            drugs = self.extract_drugs(query)

            # Step 2: Classify intent using enhanced intent agent
            intent = self.classify_intent(query, drugs)

            # Step 3: Generate response using enhanced response agent
            if db_results:
                response = self.generate_response(intent, drugs, db_results)
            else:
                response = "Processed by agent system - awaiting database results"

            return {
                "drugs": drugs,
                "intent": intent,
                "response": response
            }

        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            raise Exception(f"Query processing failed: {e}")
