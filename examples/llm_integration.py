"""
LLM Integration Examples with Agentic Memory Framework

Demonstrates real agent interactions using:
- OpenAI (ChatGPT)
- Google Gemini

Set your API keys as environment variables or in the code below.
"""

import sys
sys.path.insert(0, '/Users/jchen65/dev/ai_playground/agentic_memory')

import os
from typing import List, Dict
from agentic_memory import AgenticMemory, MemoryType, MemoryScope


# ============================================================================
# API KEY CONFIGURATION
# ============================================================================

# Option 1: Set as environment variables (recommended)
# export OPENAI_API_KEY="your-key-here"
# export GOOGLE_API_KEY="your-key-here"

# Option 2: Set directly in code (for testing)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-google-api-key-here")


# ============================================================================
# LLM CLIENT WRAPPERS
# ============================================================================

class OpenAIClient:
    """Wrapper for OpenAI API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._initialize()
    
    def _initialize(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            print("⚠️  OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            print(f"⚠️  Failed to initialize OpenAI: {e}")
    
    def generate(self, messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
        """Generate response from OpenAI"""
        if not self.client:
            return "OpenAI client not initialized. Please install openai library."
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {e}"


class GeminiClient:
    """Wrapper for Google Gemini API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self._initialize()
    
    def _initialize(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        except ImportError:
            print("⚠️  Google Generative AI library not installed. Run: pip install google-generativeai")
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini: {e}")
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from Gemini"""
        if not self.model:
            return "Gemini client not initialized. Please install google-generativeai library."
        
        try:
            # Convert messages to Gemini format
            prompt = self._format_messages(messages)
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error calling Gemini: {e}"
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Gemini prompt"""
        formatted = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted.append(f"Instructions: {content}")
            elif role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        return "\n\n".join(formatted)


# ============================================================================
# MEMORY-ENHANCED AGENT
# ============================================================================

class MemoryEnhancedAgent:
    """Agent that uses agentic memory with LLM"""
    
    def __init__(self, llm_client, memory: AgenticMemory, user_id: str = "user_default"):
        self.llm = llm_client
        self.memory = memory
        self.user_id = user_id
        self.conversation_history = []
        
        # Create isolated context for this user
        self.memory.create_isolated_context(user_id, MemoryScope.SESSION)
    
    def chat(self, user_message: str, task_type: str = "conversational") -> str:
        """Chat with memory-enhanced context"""
        
        # 1. Retrieve relevant memories
        memory_result = self.memory.retrieve(
            query=user_message,
            task_type=task_type,
            scope=MemoryScope.SESSION,
            context_id=self.user_id,
            max_tokens=1500
        )
        
        # 2. Build context from memories
        context_parts = []
        if memory_result.memories:
            context_parts.append("=== Relevant Context ===")
            for mem in memory_result.memories[:5]:  # Top 5 memories
                context_parts.append(f"[{mem.type.value}] {mem.content}")
            context_parts.append("=== End Context ===\n")
        
        context_str = "\n".join(context_parts)
        
        # 3. Build messages for LLM
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful AI assistant with access to conversation memory.
                
Use the provided context to give informed, personalized responses.
If the context contains relevant information, reference it naturally in your response.
If no relevant context is available, respond based on your general knowledge.

{context_str}"""
            }
        ]
        
        # Add recent conversation history (last 3 exchanges)
        messages.extend(self.conversation_history[-6:])
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # 4. Generate response
        response = self.llm.generate(messages)
        
        # 5. Store conversation in memory
        self._store_conversation(user_message, response)
        
        # 6. Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _store_conversation(self, user_message: str, assistant_response: str):
        """Store conversation turn in memory"""
        
        # Store user message as episodic memory
        self.memory.store_episodic(
            source=self.user_id,
            content=f"User said: {user_message}",
            scope=MemoryScope.SESSION,
            importance=0.6,
            tags=[self.user_id, "conversation"]
        )
        
        # Store assistant response
        self.memory.store_episodic(
            source="assistant",
            content=f"Assistant responded: {assistant_response}",
            scope=MemoryScope.SESSION,
            importance=0.5,
            tags=[self.user_id, "conversation"]
        )
    
    def remember_fact(self, fact: str, importance: float = 0.8):
        """Explicitly store a fact in memory"""
        self.memory.store_semantic(
            entity=self.user_id,
            content=fact,
            scope=MemoryScope.GLOBAL,
            importance=importance,
            tags=[self.user_id, "fact"]
        )
    
    def remember_preference(self, preference: str, importance: float = 0.9):
        """Store user preference"""
        self.memory.store_semantic(
            entity=f"{self.user_id}_preference",
            content=preference,
            scope=MemoryScope.GLOBAL,
            importance=importance,
            tags=[self.user_id, "preference"]
        )
    
    def get_memory_stats(self):
        """Get statistics about stored memories"""
        return self.memory.get_statistics()


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

def demo_chatgpt_basic_conversation():
    """Demo: Basic conversation with ChatGPT and memory"""
    print("=" * 70)
    print("DEMO 1: Basic Conversation with ChatGPT + Memory")
    print("=" * 70)
    
    # Initialize
    llm = OpenAIClient(OPENAI_API_KEY)
    memory = AgenticMemory()
    agent = MemoryEnhancedAgent(llm, memory, user_id="alice")
    
    print("\n🤖 Starting conversation with memory-enhanced ChatGPT agent...\n")
    
    # Conversation 1: User shares information
    print("👤 User: Hi! I'm working on a Python project using FastAPI.")
    response = agent.chat("Hi! I'm working on a Python project using FastAPI.")
    print(f"🤖 Agent: {response}\n")
    
    # Explicitly store as fact
    agent.remember_fact("User is working on a Python project using FastAPI")
    
    # Conversation 2: User shares preference
    print("👤 User: I prefer detailed code examples when you help me.")
    response = agent.chat("I prefer detailed code examples when you help me.")
    print(f"🤖 Agent: {response}\n")
    
    agent.remember_preference("User prefers detailed code examples")
    
    # Conversation 3: Ask about something else
    print("👤 User: What's the weather like?")
    response = agent.chat("What's the weather like?")
    print(f"🤖 Agent: {response}\n")
    
    # Conversation 4: Return to project (should remember context)
    print("👤 User: Can you help me with authentication in my project?")
    response = agent.chat("Can you help me with authentication in my project?", task_type="procedural")
    print(f"🤖 Agent: {response}\n")
    
    # Show memory stats
    stats = agent.get_memory_stats()
    print(f"📊 Memory Stats: {stats['total_memories']} memories stored")
    print(f"   By type: {stats['by_type']}")


def demo_gemini_multi_session():
    """Demo: Multi-session conversation with Gemini"""
    print("\n" + "=" * 70)
    print("DEMO 2: Multi-Session Conversation with Gemini + Memory")
    print("=" * 70)
    
    # Initialize
    llm = GeminiClient(GOOGLE_API_KEY)
    memory = AgenticMemory()
    
    print("\n📅 Session 1: Initial conversation")
    print("-" * 70)
    
    agent_session1 = MemoryEnhancedAgent(llm, memory, user_id="bob")
    
    print("👤 User: I'm learning machine learning. I'm interested in neural networks.")
    response = agent_session1.chat("I'm learning machine learning. I'm interested in neural networks.")
    print(f"🤖 Agent: {response}\n")
    
    agent_session1.remember_fact("User is learning machine learning, interested in neural networks")
    
    print("👤 User: I'm using PyTorch for my experiments.")
    response = agent_session1.chat("I'm using PyTorch for my experiments.")
    print(f"🤖 Agent: {response}\n")
    
    agent_session1.remember_preference("User prefers PyTorch framework")
    
    print("\n📅 Session 2: New conversation (next day)")
    print("-" * 70)
    
    # New session, same user - memory should persist
    agent_session2 = MemoryEnhancedAgent(llm, memory, user_id="bob")
    
    print("👤 User: Hi! Can you recommend a good tutorial?")
    response = agent_session2.chat("Hi! Can you recommend a good tutorial?", task_type="recommendation")
    print(f"🤖 Agent: {response}\n")
    
    print("✅ Notice: Agent remembers user's interests from previous session!")


def demo_chatgpt_code_assistant():
    """Demo: Code assistant with project memory"""
    print("\n" + "=" * 70)
    print("DEMO 3: Code Assistant with Project Memory (ChatGPT)")
    print("=" * 70)
    
    # Initialize
    llm = OpenAIClient(OPENAI_API_KEY)
    memory = AgenticMemory()
    agent = MemoryEnhancedAgent(llm, memory, user_id="developer_1")
    
    # Store project context
    memory.store_semantic(
        entity="project_config",
        content="Project: E-commerce API. Tech stack: FastAPI, PostgreSQL, Redis, Docker. Authentication: JWT tokens.",
        scope=MemoryScope.PROJECT,
        importance=0.9,
        tags=["project", "config"]
    )
    
    memory.store_procedural(
        workflow_name="deployment",
        content="Deployment process: 1) Run tests with pytest, 2) Build Docker image, 3) Push to registry, 4) Deploy to Kubernetes",
        scope=MemoryScope.PROJECT,
        importance=0.8,
        tags=["project", "deployment"]
    )
    
    print("\n🔧 Project context stored in memory")
    print("\n👤 User: How should I structure my authentication endpoints?")
    
    response = agent.chat(
        "How should I structure my authentication endpoints?",
        task_type="procedural"
    )
    print(f"🤖 Agent: {response}\n")
    
    print("👤 User: What's our deployment process again?")
    response = agent.chat("What's our deployment process again?", task_type="procedural")
    print(f"🤖 Agent: {response}\n")
    
    print("✅ Agent uses project-specific context from memory!")


def demo_gemini_customer_support():
    """Demo: Customer support with isolated contexts"""
    print("\n" + "=" * 70)
    print("DEMO 4: Customer Support with Context Isolation (Gemini)")
    print("=" * 70)
    
    # Initialize
    llm = GeminiClient(GOOGLE_API_KEY)
    memory = AgenticMemory()
    
    print("\n👥 Handling two customers simultaneously...\n")
    
    # Customer 1
    print("📞 Customer 1 (Alice)")
    print("-" * 70)
    agent_alice = MemoryEnhancedAgent(llm, memory, user_id="customer_alice")
    
    print("👤 Alice: I can't log into my account. Email: alice@example.com")
    response = agent_alice.chat("I can't log into my account. Email: alice@example.com")
    print(f"🤖 Agent: {response}\n")
    
    agent_alice.remember_fact("Customer Alice has login issue. Email: alice@example.com")
    
    # Customer 2
    print("📞 Customer 2 (Bob)")
    print("-" * 70)
    agent_bob = MemoryEnhancedAgent(llm, memory, user_id="customer_bob")
    
    print("👤 Bob: My payment failed. Order #12345")
    response = agent_bob.chat("My payment failed. Order #12345")
    print(f"🤖 Agent: {response}\n")
    
    agent_bob.remember_fact("Customer Bob has payment issue. Order #12345")
    
    # Continue with Alice
    print("📞 Back to Customer 1 (Alice)")
    print("-" * 70)
    print("👤 Alice: I tried resetting my password but didn't get the email.")
    response = agent_alice.chat("I tried resetting my password but didn't get the email.")
    print(f"🤖 Agent: {response}\n")
    
    print("✅ Context isolation: Agent remembers Alice's login issue, not Bob's payment issue!")


def demo_chatgpt_learning_preferences():
    """Demo: Agent learning user preferences over time"""
    print("\n" + "=" * 70)
    print("DEMO 5: Learning User Preferences (ChatGPT)")
    print("=" * 70)
    
    # Initialize
    llm = OpenAIClient(OPENAI_API_KEY)
    memory = AgenticMemory()
    agent = MemoryEnhancedAgent(llm, memory, user_id="learner")
    
    print("\n📚 Agent learns preferences through conversation...\n")
    
    # Initial interaction
    print("👤 User: Explain how neural networks work.")
    response = agent.chat("Explain how neural networks work.")
    print(f"🤖 Agent: {response}\n")
    
    # User gives feedback
    print("👤 User: That's too technical. Can you explain it more simply?")
    response = agent.chat("That's too technical. Can you explain it more simply?")
    print(f"🤖 Agent: {response}\n")
    
    agent.remember_preference("User prefers simple, non-technical explanations")
    
    # Later question
    print("👤 User: What is gradient descent?")
    response = agent.chat("What is gradient descent?")
    print(f"🤖 Agent: {response}\n")
    
    print("✅ Agent adapts explanation style based on learned preference!")


def demo_gemini_conflict_resolution():
    """Demo: Handling conflicting information"""
    print("\n" + "=" * 70)
    print("DEMO 6: Conflict Resolution (Gemini)")
    print("=" * 70)
    
    # Initialize
    llm = GeminiClient(GOOGLE_API_KEY)
    memory = AgenticMemory()
    agent = MemoryEnhancedAgent(llm, memory, user_id="user_conflict")
    
    print("\n⚠️  Testing conflict resolution...\n")
    
    # Initial preference
    print("👤 User: I prefer working with React for frontend development.")
    response = agent.chat("I prefer working with React for frontend development.")
    print(f"🤖 Agent: {response}\n")
    
    agent.remember_preference("User prefers React for frontend")
    
    # Changed preference
    print("👤 User: Actually, I've switched to Vue. I like it better now.")
    response = agent.chat("Actually, I've switched to Vue. I like it better now.")
    print(f"🤖 Agent: {response}\n")
    
    agent.remember_preference("User prefers Vue for frontend (updated)")
    
    # Query preference
    print("👤 User: What frontend framework should I use for my new project?")
    response = agent.chat("What frontend framework should I use for my new project?", task_type="recommendation")
    print(f"🤖 Agent: {response}\n")
    
    print("✅ Agent uses most recent preference (Vue), resolving conflict!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all LLM integration demos"""
    
    print("\n" + "=" * 70)
    print("🚀 LLM Integration with Agentic Memory Framework")
    print("=" * 70)
    
    print("\n📝 Configuration:")
    print(f"   OpenAI API Key: {'✓ Set' if OPENAI_API_KEY != 'your-openai-api-key-here' else '✗ Not set'}")
    print(f"   Google API Key: {'✓ Set' if GOOGLE_API_KEY != 'your-google-api-key-here' else '✗ Not set'}")
    
    if OPENAI_API_KEY == "your-openai-api-key-here" and GOOGLE_API_KEY == "your-google-api-key-here":
        print("\n⚠️  Please set your API keys before running demos!")
        print("\nOption 1: Set environment variables:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export GOOGLE_API_KEY='your-key'")
        print("\nOption 2: Edit this file and set the keys directly")
        return
    
    demos = []
    
    # Add OpenAI demos if key is set
    if OPENAI_API_KEY != "your-openai-api-key-here":
        demos.extend([
            ("ChatGPT Basic Conversation", demo_chatgpt_basic_conversation),
            ("ChatGPT Code Assistant", demo_chatgpt_code_assistant),
            ("ChatGPT Learning Preferences", demo_chatgpt_learning_preferences),
        ])
    
    # Add Gemini demos if key is set
    if GOOGLE_API_KEY != "your-google-api-key-here":
        demos.extend([
            ("Gemini Multi-Session", demo_gemini_multi_session),
            ("Gemini Customer Support", demo_gemini_customer_support),
            ("Gemini Conflict Resolution", demo_gemini_conflict_resolution),
        ])
    
    # Run demos
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ All demos complete!")
    print("=" * 70)
    
    print("\n💡 Key Takeaways:")
    print("   • Memory provides context across conversations")
    print("   • Preferences and facts are remembered")
    print("   • Context isolation prevents data mixing")
    print("   • Conflicts are automatically resolved")
    print("   • Agent responses are more personalized and informed")


if __name__ == "__main__":
    main()
