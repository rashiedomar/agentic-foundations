"""
Memory System for Research Assistant
Implements: Short-term, Long-term, and Episodic memory
"""

import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import deque


class ShortTermMemory:
    """Working memory for current task - Lesson 3"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.context = deque(maxlen=max_size)
    
    def add(self, item: Dict[str, Any]):
        """Add item to working memory"""
        self.context.append({
            **item,
            "timestamp": datetime.now().isoformat()
        })
    
    def get(self) -> List[Dict]:
        """Get current context"""
        return list(self.context)
    
    def clear(self):
        """Clear working memory"""
        self.context.clear()
    
    def get_latest(self, n: int = 3) -> List[Dict]:
        """Get n most recent items"""
        return list(self.context)[-n:]


class LongTermMemory:
    """Persistent knowledge storage - Lesson 3"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)
        self.knowledge_base = self._load_knowledge()
    
    def _load_knowledge(self) -> Dict:
        """Load existing knowledge base"""
        kb_file = self.storage_path / "knowledge_base.json"
        if kb_file.exists():
            with open(kb_file, 'r') as f:
                return json.load(f)
        return {"facts": [], "topics": {}}
    
    def _save_knowledge(self):
        """Persist knowledge to disk"""
        kb_file = self.storage_path / "knowledge_base.json"
        with open(kb_file, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
    
    def store(self, key: str, data: Any, metadata: Optional[Dict] = None):
        """Store information in long-term memory"""
        entry = {
            "data": data,
            "metadata": metadata or {},
            "stored_at": datetime.now().isoformat()
        }
        
        if key not in self.knowledge_base:
            self.knowledge_base[key] = []
        
        self.knowledge_base[key].append(entry)
        self._save_knowledge()
    
    def retrieve(self, key: str, limit: int = 5) -> List[Dict]:
        """Retrieve information from long-term memory"""
        if key in self.knowledge_base:
            return self.knowledge_base[key][-limit:]
        return []
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Simple keyword search in knowledge base"""
        results = []
        query_lower = query.lower()
        
        for key, entries in self.knowledge_base.items():
            if query_lower in key.lower():
                results.extend(entries[-limit:])
        
        return results[:limit]
    
    def store_fact(self, fact: str, topic: str):
        """Store a fact about a topic"""
        if "facts" not in self.knowledge_base:
            self.knowledge_base["facts"] = []
        
        self.knowledge_base["facts"].append({
            "fact": fact,
            "topic": topic,
            "timestamp": datetime.now().isoformat()
        })
        self._save_knowledge()
    
    def get_topic_facts(self, topic: str) -> List[str]:
        """Get all facts about a topic"""
        if "facts" not in self.knowledge_base:
            return []
        
        return [
            f["fact"] for f in self.knowledge_base["facts"]
            if f["topic"].lower() == topic.lower()
        ]


class EpisodicMemory:
    """Log of all actions and results - Lesson 3"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.current_session = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_action(self, action_type: str, details: Dict[str, Any]):
        """Log a single action"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "details": details
        }
        self.current_session.append(entry)
        self._persist_entry(entry)
    
    def _persist_entry(self, entry: Dict):
        """Write entry to log file"""
        log_file = self.log_dir / f"session_{self.session_id}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_session_history(self) -> List[Dict]:
        """Get current session's history"""
        return self.current_session
    
    def search_past_sessions(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Search through all past sessions"""
        results = []
        
        for log_file in sorted(self.log_dir.glob("session_*.jsonl")):
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if keyword.lower() in json.dumps(entry).lower():
                        results.append(entry)
                        if len(results) >= limit:
                            return results
        
        return results
    
    def get_success_rate(self) -> float:
        """Calculate success rate of current session"""
        if not self.current_session:
            return 1.0
        
        successes = sum(
            1 for entry in self.current_session
            if entry.get("details", {}).get("success", False)
        )
        
        return successes / len(self.current_session)
    
    def get_summary(self) -> str:
        """Get session summary"""
        if not self.current_session:
            return "No actions logged yet"
        
        action_counts = {}
        for entry in self.current_session:
            action = entry["action_type"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        summary = f"Session Summary (ID: {self.session_id})\n"
        summary += f"Total actions: {len(self.current_session)}\n"
        summary += f"Success rate: {self.get_success_rate():.1%}\n"
        summary += "\nAction breakdown:\n"
        for action, count in action_counts.items():
            summary += f"  - {action}: {count}\n"
        
        return summary


class MemoryManager:
    """Unified interface for all memory systems"""
    
    def __init__(self, memory_dir: Path, logs_dir: Path):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(memory_dir)
        self.episodic = EpisodicMemory(logs_dir)
    
    def clear_working_memory(self):
        """Clear short-term memory (new task)"""
        self.short_term.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get complete context from all memory systems"""
        return {
            "recent": self.short_term.get(),
            "working_memory": self.short_term.get_latest(3),
            "session_history": self.episodic.get_session_history()[-5:]
        }
    
    def remember_fact(self, fact: str, topic: str):
        """Store important fact in long-term memory"""
        self.long_term.store_fact(fact, topic)
        self.short_term.add({
            "type": "fact_stored",
            "fact": fact,
            "topic": topic
        })