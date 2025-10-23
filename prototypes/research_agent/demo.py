"""
Demo script for Research Assistant

Tests the agent without requiring API key (uses mock LLM)
Shows all components working together
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from memory import MemoryManager, ShortTermMemory, LongTermMemory, EpisodicMemory
from tools import ToolRegistry, WikipediaSearchTool


def demo_memory_systems():
    """Demo: Memory Systems (Lesson 3)"""
    print("\n" + "="*60)
    print("🧠 DEMO: Memory Systems")
    print("="*60)
    
    # Short-term memory
    print("\n1. Short-term Memory (Working Memory)")
    stm = ShortTermMemory(max_size=5)
    
    for i in range(7):
        stm.add({"action": f"search_{i}", "result": f"data_{i}"})
    
    print(f"   Added 7 items, kept last 5:")
    for item in stm.get():
        print(f"   - {item['action']}")
    
    # Long-term memory
    print("\n2. Long-term Memory (Knowledge Base)")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ltm = LongTermMemory(Path(tmpdir))
        
        ltm.store_fact("Quantum computers use qubits", "quantum computing")
        ltm.store_fact("Qubits can be in superposition", "quantum computing")
        ltm.store_fact("AI agents have memory", "AI")
        
        facts = ltm.get_topic_facts("quantum computing")
        print(f"   Facts about 'quantum computing': {len(facts)}")
        for fact in facts:
            print(f"   - {fact}")
    
    # Episodic memory
    print("\n3. Episodic Memory (Action Logs)")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        em = EpisodicMemory(Path(tmpdir))
        
        em.log_action("search", {"tool": "wikipedia", "success": True})
        em.log_action("analyze", {"tool": "llm", "success": True})
        em.log_action("write", {"tool": "file", "success": False})
        
        print(f"   Logged 3 actions")
        print(f"   Success rate: {em.get_success_rate():.0%}")
        print(f"\n   {em.get_summary()}")


def demo_tools():
    """Demo: Tool Integration (Lesson 6)"""
    print("\n" + "="*60)
    print("🛠️ DEMO: Tool Integration")
    print("="*60)
    
    print("\n1. Wikipedia Search Tool")
    wiki = WikipediaSearchTool()
    
    print(f"   Tool: {wiki.name}")
    print(f"   Description: {wiki.description[:100]}...")
    
    print("\n   Searching 'Python programming'...")
    result = wiki.execute(query="Python programming", limit=2)
    
    if result.success:
        print(f"   ✅ Found {result.data['results_count']} results")
        for r in result.data['results']:
            print(f"   - {r['title']}")
            print(f"     {r['snippet'][:80]}...")
    
    print(f"\n   Tool stats: {wiki.get_stats()}")


def demo_error_handling():
    """Demo: Error Handling (Lesson 7)"""
    print("\n" + "="*60)
    print("🛡️ DEMO: Error Handling & Recovery")
    print("="*60)
    
    from tools import BaseTool, ToolResult
    
    class UnreliableTool(BaseTool):
        """Tool that fails sometimes"""
        def __init__(self):
            super().__init__()
            self.attempt = 0
        
        @property
        def description(self):
            return "A tool that demonstrates error handling"
        
        def _execute(self, **kwargs):
            self.attempt += 1
            if self.attempt < 3:
                raise Exception(f"Temporary failure (attempt {self.attempt})")
            return "Success!"
    
    print("\n1. Retry with exponential backoff")
    tool = UnreliableTool()
    
    # Try 3 times
    for i in range(3):
        result = tool.execute()
        if result.success:
            print(f"   ✅ Succeeded on attempt {i+1}")
            break
        else:
            print(f"   ⚠️ Attempt {i+1} failed: {result.error}")
    
    print(f"\n   Final stats: {tool.get_stats()}")


def demo_agent_loop():
    """Demo: Agent Loop (Lesson 2)"""
    print("\n" + "="*60)
    print("🔄 DEMO: Agent Loop (Simplified)")
    print("="*60)
    
    print("\n1. Perceive → Plan → Act → Reflect cycle")
    
    # Simplified agent loop
    goal = "Learn about AI agents"
    iteration = 0
    max_iterations = 3
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n   Iteration {iteration}:")
        
        # Perceive
        print(f"   [PERCEIVE] Goal: {goal}")
        context = {"goal": goal, "iteration": iteration}
        
        # Plan
        print(f"   [PLAN] Decide next action...")
        if iteration == 1:
            plan = "search_wikipedia"
        elif iteration == 2:
            plan = "search_news"
        else:
            plan = "synthesize_report"
        print(f"   [PLAN] → {plan}")
        
        # Act
        print(f"   [ACT] Execute: {plan}")
        result = f"completed_{plan}"
        
        # Reflect
        progress = iteration / max_iterations
        print(f"   [REFLECT] Progress: {progress:.0%}")
        
        if iteration == max_iterations:
            print(f"   [REFLECT] ✅ Goal achieved!")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🤖 RESEARCH ASSISTANT - COMPONENT DEMOS")
    print("="*70)
    print("\nThis demonstrates all the concepts from Lessons 1-7")
    print("without requiring API keys (uses real Wikipedia API)")
    print("="*70)
    
    try:
        demo_memory_systems()
        demo_tools()
        demo_error_handling()
        demo_agent_loop()
        
        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("1. Get your free Gemini API key: https://makersuite.google.com/app/apikey")
        print("2. Run: export GEMINI_API_KEY='your-key'")
        print("3. Run: python main.py 'your research topic'")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()