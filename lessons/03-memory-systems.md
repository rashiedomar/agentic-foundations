# Lesson 3: Memory Systems

> **What you'll learn:** How AI agents remember information, the different types of memory, and how to build memory systems that make agents truly intelligent.

---

## 🎯 Learning Objectives

By the end of this lesson, you'll understand:
- Why memory is crucial for agent autonomy
- The three types of agent memory (short-term, long-term, episodic)
- How vector databases enable semantic memory
- How to implement a basic memory system
- When to use each type of memory

---

## 🧠 Why Agents Need Memory

Imagine having a conversation with someone who forgets everything after each sentence. Frustrating, right?

**That's what language models are like without memory.**

Agents need memory to:
- **Learn from experience** - "I tried this tool before and it failed"
- **Maintain context** - "The user's name is Omar and he's learning agentic AI"
- **Build knowledge** - "I've researched 50 papers on this topic"
- **Avoid repetition** - "I already checked that source"
- **Improve over time** - "Last time this approach worked better"

**Without memory, agents can't be truly autonomous.**

---

## 📖 The Three Types of Memory

Just like humans, agents have different memory systems for different purposes:

```
┌─────────────────────────────────────────────────────┐
│                   AGENT MEMORY                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐  │
│  │  SHORT-TERM MEMORY (Working Memory)         │  │
│  │  "What am I thinking about right now?"      │  │
│  │  • Current conversation                      │  │
│  │  • Immediate context (last few turns)       │  │
│  │  • Active task state                        │  │
│  │  Duration: Current session only             │  │
│  └─────────────────────────────────────────────┘  │
│                                                     │
│  ┌─────────────────────────────────────────────┐  │
│  │  LONG-TERM MEMORY (Knowledge Base)          │  │
│  │  "What do I know in general?"               │  │
│  │  • Facts and knowledge                      │  │
│  │  • Semantic understanding                   │  │
│  │  • Patterns and insights                    │  │
│  │  Duration: Permanent                        │  │
│  └─────────────────────────────────────────────┘  │
│                                                     │
│  ┌─────────────────────────────────────────────┐  │
│  │  EPISODIC MEMORY (Experience Log)           │  │
│  │  "What have I done before?"                 │  │
│  │  • Past actions and results                 │  │
│  │  • Historical conversations                 │  │
│  │  • Success/failure cases                    │  │
│  │  Duration: Permanent archive                │  │
│  └─────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

Let's explore each type in detail.

---

## 1️⃣ Short-Term Memory (Working Memory)

### What Is It?

Short-term memory is what the agent is "thinking about" right now. It's the immediate context needed for the current task.

**Human analogy:** When you're solving a math problem, you keep the numbers and steps in your head temporarily. Once done, you forget the details.

### What Goes In Short-Term Memory?

- Current conversation messages
- The current goal/task
- Recent actions (last 5-10 steps)
- Intermediate results
- Active reasoning

### Implementation: The Context Window

```python
class ShortTermMemory:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.current_context = []
        
    def add(self, item):
        """Add to short-term memory"""
        self.current_context.append(item)
        
        # Keep only recent items (sliding window)
        if len(self.current_context) > self.max_size:
            self.current_context = self.current_context[-self.max_size:]
    
    def get_context(self):
        """Get current context for LLM"""
        return self.current_context
    
    def clear(self):
        """Clear working memory (new session)"""
        self.current_context = []

# Usage
stm = ShortTermMemory(max_size=5)

stm.add({"role": "user", "content": "Research AI agents"})
stm.add({"role": "assistant", "content": "I'll search for papers"})
stm.add({"role": "system", "content": "Found 10 papers"})

# Gets only last 5 items
context = stm.get_context()
```

### Characteristics

✅ **Fast** - O(1) access  
✅ **Limited size** - Only keeps recent items  
✅ **Temporary** - Cleared between sessions  
⚠️ **No search** - Sequential access only  

### When to Use

- Current conversation state
- Active task tracking
- Immediate decision making
- Quick reference to recent actions

---

## 2️⃣ Long-Term Memory (Knowledge Base)

### What Is It?

Long-term memory is the agent's permanent knowledge store. It remembers facts, patterns, and information across sessions.

**Human analogy:** You remember that Paris is the capital of France, even though you learned it years ago.

### What Goes In Long-Term Memory?

- Important facts learned during tasks
- User preferences and information
- Domain knowledge
- Successful strategies and patterns
- Extracted insights from experiences

### The Challenge: Finding Relevant Memories

Imagine storing 1 million facts. How do you find the right one quickly?

**Solution: Vector Databases**

Instead of exact keyword matching, vector databases use **semantic similarity**:

```
Query: "How do agents plan?"

Traditional DB: No exact match for "plan" ❌

Vector DB: Finds semantically similar items:
  - "Agent reasoning strategies" ✅
  - "Task decomposition in agents" ✅
  - "Decision making process" ✅
```

### How Vector Embeddings Work

```python
# Text is converted to numbers (vectors)
text1 = "AI agents use planning"
embedding1 = [0.2, 0.8, 0.1, 0.4, ...]  # 1536 dimensions

text2 = "Agents reason and decide"
embedding2 = [0.3, 0.7, 0.2, 0.5, ...]

# Similar meanings = similar vectors
similarity = cosine_similarity(embedding1, embedding2)
# High similarity = related content!
```

### Implementation: Simple Vector Memory

```python
import numpy as np
from typing import List, Dict

class LongTermMemory:
    def __init__(self):
        self.memories = []
        self.embeddings = []
        
    def store(self, text: str, metadata: Dict = None):
        """Store a memory with its embedding"""
        # In practice, use OpenAI/Cohere embeddings
        embedding = self._get_embedding(text)
        
        self.memories.append({
            "text": text,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        })
        self.embeddings.append(embedding)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find most relevant memories"""
        query_embedding = self._get_embedding(query)
        
        # Calculate similarity to all memories
        similarities = []
        for i, mem_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, mem_embedding)
            similarities.append((sim, i))
        
        # Get top k most similar
        similarities.sort(reverse=True)
        top_indices = [idx for _, idx in similarities[:top_k]]
        
        return [self.memories[i] for i in top_indices]
    
    def _get_embedding(self, text: str):
        """Convert text to vector (simplified)"""
        # In practice: use OpenAI API
        # return openai.embeddings.create(input=text, model="text-embedding-3-small")
        return np.random.rand(1536)  # Placeholder
    
    def _cosine_similarity(self, v1, v2):
        """Calculate similarity between vectors"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Usage
ltm = LongTermMemory()

# Store knowledge
ltm.store("Omar is learning about agentic AI", 
          metadata={"type": "user_info"})
ltm.store("LangGraph is good for agent orchestration",
          metadata={"type": "framework_info"})
ltm.store("Agents use tools to interact with the world",
          metadata={"type": "concept"})

# Retrieve relevant memories
relevant = ltm.retrieve("What is Omar working on?", top_k=3)
# Returns: Information about Omar learning AI agents
```

### Popular Vector Databases

**For Learning:**
- **FAISS** (Facebook) - Local, fast, simple
- **Chroma** - Easy to use, good for prototypes

**For Production:**
- **Pinecone** - Managed, scalable
- **Weaviate** - Open-source, feature-rich
- **Qdrant** - Fast, Rust-based

### Example: Using Chroma

```python
import chromadb

# Initialize
client = chromadb.Client()
collection = client.create_collection("agent_memory")

# Store memories
collection.add(
    documents=["Omar is learning AI agents", 
               "LangGraph is an agent framework"],
    ids=["mem1", "mem2"],
    metadatas=[{"type": "user"}, {"type": "knowledge"}]
)

# Query
results = collection.query(
    query_texts=["What is Omar doing?"],
    n_results=2
)

print(results['documents'])
# Output: ["Omar is learning AI agents", ...]
```

### Characteristics

✅ **Persistent** - Survives restarts  
✅ **Searchable** - Semantic similarity search  
✅ **Scalable** - Can store millions of items  
⚠️ **Slower** - Requires embedding + search  

### When to Use

- Storing facts learned across sessions
- User preferences and history
- Domain knowledge base
- Successful patterns to reuse

---

## 3️⃣ Episodic Memory (Experience Log)

### What Is It?

Episodic memory records what the agent DID and what HAPPENED. It's a historical log of actions and outcomes.

**Human analogy:** You remember going to the beach last summer - the specific event, not just the fact that beaches exist.

### What Goes In Episodic Memory?

- Complete action logs
- Task execution history
- Tool calls and results
- Errors and how they were resolved
- Timestamps and context

### Why It Matters

Episodic memory enables:
- **Learning from mistakes** - "That API call failed with this error"
- **Debugging** - Trace back what went wrong
- **Pattern recognition** - "This type of query always needs 3 search calls"
- **Analytics** - "My success rate improved over time"

### Implementation: Structured Logs

```python
import json
from datetime import datetime
from pathlib import Path

class EpisodicMemory:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_session = []
        
    def log_action(self, action_type: str, details: Dict):
        """Log a single action"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "details": details
        }
        self.current_session.append(entry)
        
        # Also write to file for persistence
        self._append_to_file(entry)
    
    def log_perception(self, context):
        self.log_action("perceive", {"context": context})
    
    def log_plan(self, plan):
        self.log_action("plan", {"plan": plan})
    
    def log_execution(self, action, result):
        self.log_action("execute", {
            "action": action,
            "result": result,
            "success": result.get("success", False)
        })
    
    def log_reflection(self, reflection):
        self.log_action("reflect", {"reflection": reflection})
    
    def get_session_history(self):
        """Get current session's history"""
        return self.current_session
    
    def search_past_sessions(self, query: str):
        """Find similar past experiences"""
        # Search through all log files
        relevant_sessions = []
        for log_file in self.log_dir.glob("*.json"):
            with open(log_file) as f:
                session = json.load(f)
                if self._is_relevant(session, query):
                    relevant_sessions.append(session)
        return relevant_sessions
    
    def _append_to_file(self, entry):
        """Persist to disk"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"session_{today}.json"
        
        if log_file.exists():
            with open(log_file) as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _is_relevant(self, session, query):
        # Simple keyword matching (could use embeddings)
        return query.lower() in str(session).lower()

# Usage
episodic = EpisodicMemory()

# Log a complete agent loop
episodic.log_perception({"goal": "research AI agents"})
episodic.log_plan({"action": "search", "query": "AI agents 2024"})
episodic.log_execution(
    action="web_search",
    result={"success": True, "found": 10}
)
episodic.log_reflection({"progress": "25%", "next": "read papers"})

# Later: Learn from past experiences
past_searches = episodic.search_past_sessions("search failed")
# Finds all times searches failed and how they were handled
```

### Example Log Format

```json
{
  "session_id": "2025-01-15-research-task",
  "goal": "Summarize AI agent papers",
  "actions": [
    {
      "timestamp": "2025-01-15T10:30:00",
      "type": "perceive",
      "context": {"tools_available": ["search", "reader"]}
    },
    {
      "timestamp": "2025-01-15T10:30:05",
      "type": "plan",
      "reasoning": "Need to find papers first",
      "action": "search"
    },
    {
      "timestamp": "2025-01-15T10:30:10",
      "type": "execute",
      "tool": "web_search",
      "parameters": {"query": "AI agents 2024"},
      "result": {"success": true, "count": 10}
    },
    {
      "timestamp": "2025-01-15T10:30:15",
      "type": "reflect",
      "evaluation": "Good results",
      "progress": 0.25,
      "next_action": "read_papers"
    }
  ]
}
```

### Characteristics

✅ **Complete history** - Full trace of actions  
✅ **Debuggable** - Easy to see what went wrong  
✅ **Analyzable** - Can extract patterns  
⚠️ **Large** - Grows over time  

### When to Use

- Tracking agent behavior
- Debugging failures
- Learning from past attempts
- Audit trails
- Performance analysis

---

## 🔄 How All Three Work Together

```python
class AgentWithMemory:
    def __init__(self, goal, tools):
        self.goal = goal
        self.tools = tools
        
        # Three memory systems
        self.short_term = ShortTermMemory(max_size=10)
        self.long_term = LongTermMemory()
        self.episodic = EpisodicMemory()
    
    def run(self):
        while not self.is_goal_achieved():
            # 1. PERCEIVE - use all memory types
            context = {
                "goal": self.goal,
                "recent_context": self.short_term.get_context(),
                "relevant_knowledge": self.long_term.retrieve(self.goal),
                "past_attempts": self.episodic.search_past_sessions(self.goal)
            }
            self.episodic.log_perception(context)
            
            # 2. PLAN - informed by memories
            plan = self.plan(context)
            self.episodic.log_plan(plan)
            
            # 3. ACT
            result = self.act(plan)
            self.episodic.log_execution(plan, result)
            
            # Update short-term memory
            self.short_term.add({
                "action": plan,
                "result": result
            })
            
            # 4. REFLECT
            reflection = self.reflect(result)
            self.episodic.log_reflection(reflection)
            
            # Store important learnings in long-term memory
            if reflection.get("important"):
                self.long_term.store(
                    text=reflection["learning"],
                    metadata={"type": "insight", "task": self.goal}
                )
```

---

## 💡 Memory Design Patterns

### Pattern 1: Hybrid Search

Combine keyword + semantic search:

```python
def smart_retrieve(query):
    # Get semantic matches
    semantic_results = vector_db.search(query, top_k=10)
    
    # Get keyword matches
    keyword_results = keyword_search(query, top_k=10)
    
    # Combine and re-rank
    combined = merge_and_rerank(semantic_results, keyword_results)
    return combined[:5]
```

### Pattern 2: Memory Consolidation

Periodically summarize old memories:

```python
def consolidate_memories():
    """Compress old memories to save space"""
    old_memories = get_memories_older_than(days=30)
    
    # Summarize into key insights
    summary = llm.summarize(old_memories)
    
    # Store summary, delete details
    store_consolidated(summary)
    delete_old_details(old_memories)
```

### Pattern 3: Relevance Filtering

Only retrieve truly relevant memories:

```python
def retrieve_with_threshold(query, threshold=0.7):
    """Only return highly relevant memories"""
    all_results = memory.retrieve(query, top_k=20)
    
    # Filter by similarity threshold
    relevant = [r for r in all_results if r.similarity > threshold]
    
    return relevant
```

---

## ⚠️ Common Memory Pitfalls

### 1. Context Overflow

**Problem:** Trying to fit too much in short-term memory

```python
# BAD: Keeping everything
for i in range(1000):
    short_term.add(result[i])  # Context explodes!
```

**Solution:** Be selective

```python
# GOOD: Keep only important items
if result.is_important():
    short_term.add(result)
```

### 2. Retrieval Irrelevance

**Problem:** Getting unrelated memories

```python
# BAD: Taking top results blindly
memories = retrieve(query, top_k=5)
# Might include completely unrelated stuff
```

**Solution:** Use similarity thresholds

```python
# GOOD: Filter by relevance
memories = retrieve_with_threshold(query, min_similarity=0.75)
```

### 3. Memory Staleness

**Problem:** Old information becomes outdated

**Solution:** Add timestamps and freshness scoring

```python
def retrieve_fresh(query, recency_weight=0.3):
    results = retrieve(query)
    
    for r in results:
        age_days = (now - r.timestamp).days
        freshness = 1 / (1 + age_days)
        
        # Combine semantic similarity with freshness
        r.score = r.similarity * (1 - recency_weight) + \
                  freshness * recency_weight
    
    return sorted(results, key=lambda x: x.score, reverse=True)
```

---

## 💡 Key Takeaways

1. **Memory enables autonomy** - Without memory, agents are stateless chatbots
2. **Three types serve different purposes:**
   - Short-term = Current context
   - Long-term = Permanent knowledge
   - Episodic = Historical actions
3. **Vector databases enable semantic search** - Find by meaning, not keywords
4. **Logs are underrated** - Episodic memory helps debugging and learning
5. **Design for relevance** - Not all memories are equally important

**Golden Rule:** The right memory at the right time is what makes agents smart.

---

## 🛠️ Hands-On Challenge

### Task: Build a Memory-Enabled Agent

Create an agent that remembers information across runs.

```python
class MemoryAgent:
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.episodic = EpisodicMemory()
    
    def chat(self, user_message):
        """Chat with memory"""
        # Add to short-term
        self.short_term.add({"role": "user", "content": user_message})
        
        # Retrieve relevant long-term memories
        relevant = self.long_term.retrieve(user_message)
        
        # Generate response with memory context
        context = {
            "recent": self.short_term.get_context(),
            "knowledge": relevant
        }
        
        response = self.generate_response(user_message, context)
        
        # Store important facts
        if self.is_worth_remembering(user_message):
            self.long_term.store(user_message)
        
        # Log interaction
        self.episodic.log_action("chat", {
            "user": user_message,
            "agent": response
        })
        
        return response

# Test it:
agent = MemoryAgent()
agent.chat("My name is Omar")
agent.chat("I'm learning about AI agents")
# Later...
agent.chat("What's my name?")  # Should remember!
```

### Your Task:

1. **Implement the memory systems** from the examples
2. **Create a simple chatbot** that remembers facts
3. **Test it** - tell it info in one session, restart, ask it questions
4. **Extend it** - Add memory search, better retrieval logic

---

## 🤔 Reflection Questions

1. **How much memory is too much?** When should you forget things?

2. **Privacy concerns?** What if agents remember sensitive information?

3. **Memory vs recomputation?** Sometimes it's faster to redo than to remember. When?

4. **How do humans manage memory?** What can we learn from neuroscience?

---

## 📚 Learn More

### Essential Reading
- **Vector Databases Explained** - [Pinecone Learning Center](https://www.pinecone.io/learn/)
- **LangChain Memory** - [Memory Types Guide](https://python.langchain.com/docs/modules/memory/)
- **Memory in LLMs** - Research on context management

### Tools to Explore
- **Chroma** - Simple vector DB for learning
- **FAISS** - Facebook's similarity search library
- **LlamaIndex** - Data framework for LLM apps with memory

### Papers
- "Memory in Language Models" - How LLMs use context
- "Retrieval Augmented Generation (RAG)" - External memory for LLMs

---

## ➡️ What's Next?

You now understand memory systems! Next, let's PUT IT ALL TOGETHER.

**Next Lesson:** [Building Your First Complete Agent](04-first-agent.md)

We'll combine everything:
- The agent loop (Lesson 2)
- Memory systems (Lesson 3)
- Into a working agent that can actually accomplish tasks!

Time to build! 🚀

---

## 📝 My Personal Notes

*Fill this in as you learn:*

**What clicked for me:**
- 

**What confused me:**
- 

**Memory ideas for my agents:**
- 

**Questions about implementation:**
- 

---

**Lesson Status:** ✅ Complete  
**Estimated Time:** 40-50 minutes  
**Next Step:** Implement the memory challenge, then move to Lesson 4!
