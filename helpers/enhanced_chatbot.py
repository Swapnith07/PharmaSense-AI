from agents import PharmaceuticalAgentSystem
from main import MedicalTermExtractor
from enhanced_agents import EnhancedPharmaceuticalAgentSystem
import logging
import time


class EnhancedPharmaceuticalChatbot:
    def __init__(self):
        """Initialize enhanced chatbot with advanced context-aware capabilities"""
        self.agent_system = PharmaceuticalAgentSystem()  # Original agent system
        self.enhanced_agents = EnhancedPharmaceuticalAgentSystem()  # New enhanced agents
        self.medical_extractor = MedicalTermExtractor()
        self.logger = logging.getLogger(__name__)

        # Session management
        self.session_context = {
            'user_profile': {},
            'conversation_history': [],
            'mentioned_drugs': set(),
            'safety_alerts': [],
            'user_preferences': {
                'detail_level': 'standard',  # brief, standard, detailed
                'show_technical_info': False,
                'emergency_notifications': True
            }
        }

    def chat(self):
        """Enhanced chat loop with context-aware responses"""
        self._display_enhanced_welcome()

        while True:
            try:
                user_query = input("💬 You: ").strip()

                # Handle special commands
                if self._handle_special_commands(user_query):
                    continue

                if not user_query:
                    continue

                # Process with enhanced context awareness
                self._process_enhanced_query(user_query)

            except KeyboardInterrupt:
                self._graceful_exit()
                break
            except Exception as e:
                self.logger.error(f"Chat error: {e}")
                print(f"❌ Unexpected error: {e}")

    def _display_enhanced_welcome(self):
        """Enhanced welcome with capabilities overview"""
        print("🤖 Enhanced Pharmaceutical Safety Assistant v2.0")
        print("=" * 55)
        print("🎯 Enhanced Capabilities:")
        print("• 🔍 Smart drug interaction checking with severity analysis")
        print("• 💊 Intelligent drug alternatives with detailed comparisons")
        print("• 🧠 Medical knowledge beyond database (powered by AI)")
        print("• 🚨 Real-time emergency situation assessment")
        print("• 📊 Personalized responses based on your medical literacy")
        print("• 🎭 Context-aware conversations with session memory")
        print("• 🔧 Advanced drug name correction and suggestions")
        print("\n🎮 Smart Commands:")
        print("  'settings' - Customize response preferences")
        print("  'profile' - Set your medical profile for personalized advice")
        print("  'session' - View conversation summary and mentioned drugs")
        print("  'emergency' - Quick emergency drug interaction check")
        print("  'history' - View your query history")
        print("  'help' - Detailed help and examples")
        print("  'quit' - Exit safely")
        print("\n📝 Example Queries:")
        print("  'Can I take aspirin with warfarin while pregnant?'")
        print("  'What are safer alternatives to ibuprofen for elderly?'")
        print("  'I accidentally took double dose of metformin - help!'")
        print("  'Explain how blood thinners work in simple terms'")
        print("  'Check all my medications for interactions'")
        print("=" * 55)

    def _handle_special_commands(self, user_input):
        """Handle enhanced special commands"""
        command = user_input.lower().strip()

        if command in ['quit', 'exit', 'bye']:
            self._graceful_exit()
            return True

        elif command == 'settings':
            self._show_settings_menu()
            return True

        elif command == 'profile':
            self._setup_user_profile()
            return True

        elif command == 'session':
            self._show_session_summary()
            return True

        elif command == 'emergency':
            self._emergency_check_mode()
            return True

        elif command == 'history':
            self._show_conversation_history()
            return True

        elif command == 'help':
            self._show_detailed_help()
            return True

        elif command.startswith('set '):
            self._handle_setting_command(command)
            return True

        return False

    def _process_enhanced_query(self, user_query):
        """Process query with full enhanced context awareness"""
        print(f"\n🔍 Processing with enhanced AI analysis...")
        start_time = time.time()

        try:
            # Step 1: Basic processing (existing system)
            result = self.medical_extractor.process_query(user_query)

            if not result['success']:
                print(f"❌ Error: {result['error']}")
                self._suggest_query_improvements(user_query)
                return

            # Step 2: Enhanced BERT analysis
            bert_analysis = self.enhanced_agents.analyze_query_with_bert(
                user_query)

            # Step 3: Advanced query analysis with Gemini
            advanced_analysis = self.enhanced_agents.perform_advanced_query_analysis(
                user_query)

            # Step 4: Emergency assessment
            emergency_assessment = self.enhanced_agents.assess_emergency_situation(
                user_query, advanced_analysis, result['database_results']
            )

            # Step 5: Handle emergency situations first
            if emergency_assessment.get('is_emergency', False):
                self._handle_emergency_response(emergency_assessment)

            # Step 6: Check if medical knowledge needed for unknown drugs
            unknown_drugs = [drug for drug in result['corrected_drugs']
                             if not self._drug_in_database(drug, result)]

            medical_knowledge = ""
            if unknown_drugs:
                print(f"🧠 Getting AI medical knowledge for: {unknown_drugs}")
                for drug in unknown_drugs:
                    knowledge = self.enhanced_agents.get_medical_knowledge_for_unknown_drugs(
                        drug, user_query
                    )
                    medical_knowledge += f"\n{knowledge}\n"

            # Step 7: Generate enhanced contextual response
            enhanced_response = self.enhanced_agents.generate_enhanced_contextual_response(
                query=user_query,
                intent=result['intent'],
                drugs=result['corrected_drugs'],
                db_results=result['database_results'],
                query_analysis=advanced_analysis,
                emergency_assessment=emergency_assessment,
                session_context=self.session_context
            )

            # Step 8: Display comprehensive results
            self._display_enhanced_analysis(
                result, bert_analysis, advanced_analysis, emergency_assessment)

            # Step 9: Show enhanced response
            print(f"\n🤖 **Enhanced AI Response:**")
            print(enhanced_response)

            # Step 10: Show medical knowledge if available
            if medical_knowledge:
                print(f"\n🧠 **Additional Medical Knowledge:**")
                print(medical_knowledge)

            # Step 11: Update session context
            self._update_session_context(user_query, result, advanced_analysis)

            # Step 12: Show processing time
            processing_time = time.time() - start_time
            print(f"\n⚡ Processing completed in {processing_time:.2f} seconds")

            # Step 13: Suggest follow-up questions
            self._suggest_followup_questions(result, advanced_analysis)

        except Exception as e:
            self.logger.error(f"Enhanced processing error: {e}")
            print(f"❌ Enhanced processing failed: {e}")

        print("\n" + "=" * 70)

    def _display_enhanced_analysis(self, result, bert_analysis, advanced_analysis, emergency_assessment):
        """Display comprehensive analysis results"""
        print(f"\n📊 **Enhanced Analysis Results:**")

        # Basic analysis
        print(f"🧪 Drugs identified: {result['corrected_drugs']}")
        print(f"🎯 Intent: {result['intent']}")

        # BERT analysis
        if bert_analysis and not bert_analysis.get('error'):
            print(
                f"🧠 User level (BERT): {bert_analysis.get('user_level', 'UNKNOWN')}")
            print(
                f"🔍 Medical entities: {bert_analysis.get('medical_entities', [])}")
            if bert_analysis.get('urgency_score', 0) > 0:
                print(
                    f"⚠️ Urgency indicators: {bert_analysis.get('urgency_score')}")

        # Advanced analysis
        if advanced_analysis.get('analysis_success'):
            print(
                f"🎭 Emotional state: {advanced_analysis.get('emotional_state', 'CALM')}")
            print(
                f"📈 Complexity: {advanced_analysis.get('complexity_level', 'UNKNOWN')}")
            if advanced_analysis.get('urgency_indicators'):
                print(
                    f"🚨 Urgency flags: {advanced_analysis.get('urgency_indicators')}")

        # Emergency assessment
        if emergency_assessment.get('assessment_success'):
            urgency = emergency_assessment.get('urgency_level', 'LOW')
            print(f"🚨 Emergency level: {urgency}")
            if emergency_assessment.get('is_emergency'):
                print("⚠️ **EMERGENCY SITUATION DETECTED**")

        # Database results summary
        if result['intent'] == 'check_interaction':
            interactions = result['database_results'].get('interactions', [])
            if interactions:
                print(f"⚡ Interactions found: {len(interactions)}")
                serious_count = self._count_serious_interactions(interactions)
                if serious_count > 0:
                    print(f"🚨 Serious interactions: {serious_count}")
            else:
                print("✅ No database interactions found")

        elif result['intent'] == 'find_similar':
            similar = result['database_results'].get('similar_drugs', [])
            if similar:
                print(f"🔍 Similar drugs found: {len(similar)}")
                high_quality = sum(1 for d in similar if d.get(
                    'similarity_score', 0) > 0.8)
                if high_quality > 0:
                    print(f"⭐ High-quality matches: {high_quality}")
            else:
                print("❌ No similar drugs in database")

    def _handle_emergency_response(self, emergency_assessment):
        """Handle emergency situations with immediate response"""
        print("\n" + "🚨" * 20)
        print("**EMERGENCY SITUATION DETECTED**")
        print("🚨" * 20)

        print(
            f"\n⚠️ **Urgency Level: {emergency_assessment.get('urgency_level', 'HIGH')}**")

        immediate_actions = emergency_assessment.get('immediate_actions', [])
        if immediate_actions:
            print("\n🆘 **IMMEDIATE ACTIONS:**")
            for i, action in enumerate(immediate_actions, 1):
                print(f"  {i}. {action}")

        emergency_contacts = emergency_assessment.get('emergency_contacts', '')
        if emergency_contacts:
            print(f"\n📞 **EMERGENCY CONTACTS:**")
            print(f"  {emergency_contacts}")

        when_to_seek = emergency_assessment.get('when_to_seek_help', '')
        if when_to_seek:
            print(f"\n🏥 **WHEN TO SEEK HELP:**")
            print(f"  {when_to_seek}")

        print("\n" + "🚨" * 20)

        # Add to safety alerts
        self.session_context['safety_alerts'].append({
            'timestamp': time.time(),
            'urgency_level': emergency_assessment.get('urgency_level'),
            'actions_taken': immediate_actions
        })

    def _setup_user_profile(self):
        """Interactive user profile setup for personalized responses"""
        print("\n👤 **User Profile Setup for Personalized Advice**")
        print("This helps provide more relevant and safe recommendations.")
        print("(All information is kept locally and not shared)")

        # Age group
        print("\n📅 Age group:")
        print("1. Under 18 (pediatric)")
        print("2. 18-65 (adult)")
        print("3. Over 65 (elderly)")
        print("4. Prefer not to say")

        age_choice = input("Choose (1-4): ").strip()
        age_groups = {'1': 'pediatric', '2': 'adult',
                      '3': 'elderly', '4': 'not_specified'}
        self.session_context['user_profile']['age_group'] = age_groups.get(
            age_choice, 'not_specified')

        # Medical conditions
        conditions = input(
            "\n🏥 Any medical conditions? (e.g., diabetes, hypertension, allergies): ").strip()
        if conditions:
            self.session_context['user_profile']['conditions'] = conditions

        # Pregnancy/breastfeeding
        if age_choice == '2':  # Adult
            pregnancy = input(
                "\n🤱 Pregnancy/breastfeeding status (if applicable): ").strip()
            if pregnancy:
                self.session_context['user_profile']['pregnancy_status'] = pregnancy

        # Medical literacy preference
        print("\n📚 Preferred explanation level:")
        print("1. Simple (no medical jargon)")
        print("2. Standard (balanced)")
        print("3. Detailed (medical terms OK)")

        literacy_choice = input("Choose (1-3): ").strip()
        literacy_levels = {'1': 'simple', '2': 'standard', '3': 'detailed'}
        self.session_context['user_preferences']['detail_level'] = literacy_levels.get(
            literacy_choice, 'standard')

        print("\n✅ Profile updated! Responses will now be personalized for you.")

    def _show_session_summary(self):
        """Show comprehensive session summary"""
        summary = self.enhanced_agents.get_session_summary()

        print("\n📊 **Session Summary:**")
        print(f"  💬 Total queries: {summary.get('total_queries', 0)}")
        print(
            f"  💊 Unique drugs mentioned: {summary.get('unique_drugs_mentioned', 0)}")

        if summary.get('drugs_list'):
            print(f"  📋 Drugs: {', '.join(list(summary['drugs_list'])[:5])}")
            if len(summary['drugs_list']) > 5:
                print(f"       ... and {len(summary['drugs_list']) - 5} more")

        if summary.get('recent_intents'):
            print(f"  🎯 Recent intents: {summary['recent_intents']}")

        if summary.get('safety_alerts_issued') > 0:
            print(
                f"  🚨 Safety alerts issued: {summary['safety_alerts_issued']}")

        # User profile summary
        if self.session_context['user_profile']:
            print(f"\n👤 **User Profile:**")
            profile = self.session_context['user_profile']
            if profile.get('age_group'):
                print(f"  📅 Age group: {profile['age_group']}")
            if profile.get('conditions'):
                print(f"  🏥 Conditions: {profile['conditions']}")
            if profile.get('pregnancy_status'):
                print(f"  🤱 Pregnancy status: {profile['pregnancy_status']}")

        # Suggest comprehensive interaction check if multiple drugs
        if summary.get('unique_drugs_mentioned', 0) >= 3:
            print(
                f"\n💡 **Suggestion:** You've mentioned {summary['unique_drugs_mentioned']} drugs.")
            check = input(
                "Would you like a comprehensive interaction check? (y/n): ").lower().strip()
            if check == 'y':
                self._comprehensive_interaction_check()

    def _comprehensive_interaction_check(self):
        """Perform comprehensive interaction check for all session drugs"""
        all_drugs = list(self.session_context['mentioned_drugs'])

        if len(all_drugs) < 2:
            print("Need at least 2 drugs for interaction checking.")
            return

        print(f"\n🔍 **Comprehensive Interaction Analysis**")
        print(f"Checking {len(all_drugs)} drugs: {', '.join(all_drugs)}")

        # Use existing interaction checking
        interactions = self.medical_extractor.check_drug_interactions(
            all_drugs)

        if interactions:
            print(
                f"\n⚠️ **Found {len(interactions)} potential interactions:**")

            # Categorize by severity
            critical = self._filter_interactions_by_keywords(
                interactions, ['death', 'fatal', 'life-threatening'])
            major = self._filter_interactions_by_keywords(
                interactions, ['bleeding', 'severe', 'major'])
            moderate = self._filter_interactions_by_keywords(
                interactions, ['monitor', 'caution', 'adjustment'])

            if critical:
                print(
                    f"🚨 **CRITICAL ({len(critical)}):** Life-threatening interactions!")
                for interaction in critical[:2]:
                    self._display_interaction_detail(interaction)

            if major:
                print(
                    f"⚠️ **MAJOR ({len(major)}):** Serious interactions requiring attention")
                for interaction in major[:2]:
                    self._display_interaction_detail(interaction)

            if moderate:
                print(
                    f"💡 **MODERATE ({len(moderate)}):** May require monitoring")

            # Emergency assessment for all interactions
            emergency_assessment = self.enhanced_agents.assess_emergency_situation(
                f"Comprehensive check: {', '.join(all_drugs)}",
                {'urgency_indicators': []},
                {'interactions': interactions}
            )

            if emergency_assessment.get('is_emergency'):
                self._handle_emergency_response(emergency_assessment)
        else:
            print("✅ **No interactions found** between your medications in our database.")
            print(
                "💡 This doesn't guarantee safety - always consult healthcare providers.")

    def _suggest_followup_questions(self, result, advanced_analysis):
        """Suggest intelligent follow-up questions based on context"""
        suggestions = []

        intent = result['intent']
        drugs = result['corrected_drugs']

        if intent == 'check_interaction':
            interactions = result['database_results'].get('interactions', [])
            if interactions:
                suggestions.extend([
                    "What are the signs I should watch for with these interactions?",
                    "How can I minimize the risks of these drug interactions?",
                    "Are there safer alternatives to any of these medications?"
                ])
            else:
                suggestions.extend([
                    "What are the individual side effects of these drugs?",
                    "Are there any food or alcohol interactions I should avoid?"
                ])

        elif intent == 'find_similar':
            suggestions.extend([
                "What are the differences between these alternative drugs?",
                "Which alternative has the fewest side effects?",
                "How do I safely switch from one medication to another?"
            ])

        # Add context-specific suggestions
        if advanced_analysis.get('emotional_state') in ['ANXIOUS', 'CONCERNED']:
            suggestions.append(
                "What questions should I ask my doctor about these medications?")

        if self.session_context['user_profile'].get('age_group') == 'elderly':
            suggestions.append(
                "Are there special considerations for elderly patients with these drugs?")

        if suggestions:
            print(f"\n🤔 **You might also want to ask:**")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"  {i}. {suggestion}")

    def _update_session_context(self, query, result, advanced_analysis):
        """Update session context with new information"""
        # Add to conversation history
        self.session_context['conversation_history'].append({
            'query': query,
            'intent': result['intent'],
            'drugs': result['corrected_drugs'],
            'emotional_state': advanced_analysis.get('emotional_state', 'CALM'),
            'timestamp': time.time()
        })

        # Update mentioned drugs
        for drug in result['corrected_drugs']:
            self.session_context['mentioned_drugs'].add(drug)

        # Update enhanced agents session context
        self.enhanced_agents.update_session_context(
            query, result['corrected_drugs'], result['intent'])

        # Keep only last 10 conversations
        if len(self.session_context['conversation_history']) > 10:
            self.session_context['conversation_history'] = self.session_context['conversation_history'][-10:]

    def _drug_in_database(self, drug, result):
        """Check if drug was found in database results"""
        if result['intent'] == 'find_similar':
            similar_drugs = result['database_results'].get('similar_drugs', [])
            return len(similar_drugs) > 0
        elif result['intent'] == 'check_interaction':
            interactions = result['database_results'].get('interactions', [])
            return len(interactions) > 0
        return False

    def _count_serious_interactions(self, interactions):
        """Count serious interactions"""
        serious_keywords = ['bleeding', 'severe',
                            'major', 'dangerous', 'death', 'fatal']
        return sum(1 for i in interactions
                   if any(keyword in i.get('interaction_description', '').lower()
                          for keyword in serious_keywords))

    def _filter_interactions_by_keywords(self, interactions, keywords):
        """Filter interactions by severity keywords"""
        return [i for i in interactions
                if any(keyword in i.get('interaction_description', '').lower()
                       for keyword in keywords)]

    def _display_interaction_detail(self, interaction):
        """Display detailed interaction information"""
        drug1 = interaction.get('entity1', {}).get('name', 'Unknown')
        drug2 = interaction.get('entity2', {}).get('name', 'Unknown')
        desc = interaction.get('interaction_description',
                               'No description available')
        print(f"    • **{drug1} ↔ {drug2}:** {desc[:120]}...")

    def _suggest_query_improvements(self, failed_query):
        """Suggest improvements for failed queries"""
        print("\n💡 **Query Suggestions:**")
        print("• Try using generic drug names instead of brand names")
        print("• Check spelling of drug names")
        print("• Use complete drug names (e.g., 'acetaminophen' not 'Tylenol')")
        print("\n📝 **Example formats:**")
        print("  'Can I take [drug1] with [drug2]?'")
        print("  'What are alternatives to [drug name]?'")
        print("  'Is [drug name] safe during pregnancy?'")

    def _show_settings_menu(self):
        """Show interactive settings menu"""
        print("\n⚙️ **Settings Menu:**")
        print("1. Response detail level")
        print("2. Show technical information")
        print("3. Emergency notifications")
        print("4. Reset all settings")
        print("5. Back to chat")

        choice = input("Choose option (1-5): ").strip()

        if choice == "1":
            self._set_detail_level()
        elif choice == "2":
            self._toggle_technical_info()
        elif choice == "3":
            self._toggle_emergency_notifications()
        elif choice == "4":
            self._reset_settings()

    def _set_detail_level(self):
        """Set response detail level"""
        print("\n📊 **Response Detail Level:**")
        print("1. Brief (quick answers)")
        print("2. Standard (balanced)")
        print("3. Detailed (comprehensive)")

        choice = input("Choose (1-3): ").strip()
        levels = {'1': 'brief', '2': 'standard', '3': 'detailed'}

        if choice in levels:
            self.session_context['user_preferences']['detail_level'] = levels[choice]
            print(f"✅ Set to {levels[choice]} responses")

    def _graceful_exit(self):
        """Enhanced graceful exit with session summary"""
        print("\n👋 **Session Complete!**")

        # Show session statistics
        total_queries = len(self.session_context['conversation_history'])
        unique_drugs = len(self.session_context['mentioned_drugs'])
        safety_alerts = len(self.session_context['safety_alerts'])

        if total_queries > 0:
            print(f"📊 Session Stats:")
            print(f"  • {total_queries} queries processed")
            print(f"  • {unique_drugs} unique drugs discussed")
            if safety_alerts > 0:
                print(f"  • {safety_alerts} safety alerts issued")

        print("\n💊 **Important Reminders:**")
        print("• Always consult healthcare professionals for medical decisions")
        print("• This tool provides information, not medical advice")
        print("• Report adverse drug reactions to your doctor")
        print("• Keep your medication list updated")

        print("\n🤖 Stay safe and take care! Goodbye! 👋")


# Enhanced test functions with new capabilities
def test_enhanced_queries():
    """Test enhanced chatbot with complex queries"""
    chatbot = EnhancedPharmaceuticalChatbot()

    enhanced_test_queries = [
        # Emergency scenarios
        "I accidentally took double dose of warfarin - what should I do?",
        "Having severe bleeding after taking aspirin and warfarin together",

        # Knowledge-based queries (not in database)
        "How does metformin work for diabetes?",
        "What are the side effects of lisinopril?",

        # Context-aware scenarios
        "I'm pregnant and need pain relief - what's safe?",
        "My elderly father takes 8 medications - check for interactions",

        # Complex multi-drug scenarios
        "Can I take aspirin, ibuprofen, and warfarin together while on blood pressure medication?",
    ]

    print("🧪 **Testing Enhanced Chatbot Capabilities:**\n")

    for i, query in enumerate(enhanced_test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {query}")
        print('='*60)

        # Simulate user input
        print(f"💬 You: {query}")

        # Process with enhanced system
        chatbot._process_enhanced_query(query)

        print("\n" + "⏸️ " * 20)
        input("Press Enter to continue to next test...")


def main():
    """Enhanced main function with new capabilities"""
    print("🏥 **Enhanced Pharmaceutical Intelligence System v2.0**")
    print("Choose option:")
    print("1. 🤖 Interactive enhanced chatbot")
    print("2. 🧪 Test enhanced capabilities")
    print("3. 📊 Test original functionality")
    print("4. 🔍 Test bleeding interactions")
    print("5. 🔧 Test drug corrections")
    print("6. 🚀 Run all tests")

    choice = input("Enter 1-6: ").strip()

    if choice == "1":
        chatbot = EnhancedPharmaceuticalChatbot()
        chatbot.chat()
    elif choice == "2":
        test_enhanced_queries()
    elif choice == "3":
        # Use original test from your existing code
        from chatbot import test_sample_queries
        test_sample_queries()
    elif choice == "4":
        from chatbot import test_bleeding_interactions
        test_bleeding_interactions()
    elif choice == "5":
        from chatbot import test_drug_corrections
        test_drug_corrections()
    elif choice == "6":
        print("🚀 Running comprehensive test suite...")
        test_enhanced_queries()
        print("\n" + "="*60)
        from chatbot import test_sample_queries, test_bleeding_interactions, test_drug_corrections
        test_sample_queries()
        test_bleeding_interactions()
        test_drug_corrections()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
