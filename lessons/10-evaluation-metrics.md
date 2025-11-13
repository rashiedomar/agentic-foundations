# Lesson 10: Evaluation & Metrics

> **What you'll learn:** How to measure agent performance, define success criteria, track key metrics, and build evaluation frameworks that tell you if your agent is actually working well.

---

## 🎯 Learning Objectives

By the end of this lesson, you'll understand:
- What metrics matter for AI agents
- How to design evaluation frameworks
- Different types of evaluation (offline vs online)
- Building automated testing pipelines
- Human-in-the-loop evaluation
- Performance monitoring in production
- When an agent is "good enough" to deploy

---

## 📊 The Measurement Challenge

You've built an agent, but...

```
❓ Is it actually good?
❓ Is it getting better or worse?
❓ How does it compare to alternatives?
❓ When is it safe to deploy?
❓ How do users feel about it?
❓ What's the ROI?
```

**Without metrics, you're flying blind.**

---

## 🎯 Core Metrics for Agents

### 1. Task Success Metrics

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

@dataclass
class TaskMetrics:
    """Core metrics for agent task execution"""
    
    task_id: str
    success: bool
    completion_time: float  # seconds
    steps_taken: int
    tools_used: List[str]
    tokens_consumed: int
    cost: float
    error_count: int
    retry_count: int
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency (0-1)"""
        # Fewer steps and retries = more efficient
        base_score = 1.0
        base_score -= (self.steps_taken - 1) * 0.05  # Penalize extra steps
        base_score -= self.retry_count * 0.1  # Penalize retries
        base_score -= self.error_count * 0.15  # Penalize errors
        
        return max(0, min(1, base_score))
    
    @property
    def cost_per_token(self) -> float:
        """Calculate cost efficiency"""
        if self.tokens_consumed == 0:
            return 0
        return self.cost / self.tokens_consumed

class SuccessRateTracker:
    """Track agent success rates over time"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.results = []
        self.timestamps = []
    
    def record(self, success: bool):
        """Record a task result"""
        self.results.append(success)
        self.timestamps.append(datetime.now())
        
        # Keep only recent results
        if len(self.results) > self.window_size:
            self.results = self.results[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
    
    @property
    def success_rate(self) -> float:
        """Calculate current success rate"""
        if not self.results:
            return 0.0
        return sum(self.results) / len(self.results)
    
    @property
    def trend(self) -> str:
        """Determine if performance is improving"""
        if len(self.results) < 20:
            return "insufficient_data"
        
        # Compare first half to second half
        mid = len(self.results) // 2
        first_half = sum(self.results[:mid]) / mid
        second_half = sum(self.results[mid:]) / len(self.results[mid:])
        
        if second_half > first_half + 0.1:
            return "improving ↑"
        elif second_half < first_half - 0.1:
            return "degrading ↓"
        else:
            return "stable →"
    
    def get_hourly_rates(self) -> Dict[int, float]:
        """Get success rates by hour of day"""
        hourly = {}
        
        for i, (result, timestamp) in enumerate(zip(self.results, self.timestamps)):
            hour = timestamp.hour
            if hour not in hourly:
                hourly[hour] = []
            hourly[hour].append(result)
        
        return {
            hour: sum(results) / len(results)
            for hour, results in hourly.items()
        }
```

### 2. Performance Metrics

```python
import time
from functools import wraps
import psutil
import tracemalloc

class PerformanceMonitor:
    """Monitor agent performance metrics"""
    
    def __init__(self):
        self.metrics = []
        tracemalloc.start()
    
    def measure(self, func):
        """Decorator to measure function performance"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start measurements
            start_time = time.perf_counter()
            start_memory = tracemalloc.get_traced_memory()[0]
            process = psutil.Process()
            start_cpu = process.cpu_percent()
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
            
            # End measurements
            end_time = time.perf_counter()
            end_memory = tracemalloc.get_traced_memory()[0]
            end_cpu = process.cpu_percent()
            
            # Record metrics
            metrics = {
                'function': func.__name__,
                'duration': end_time - start_time,
                'memory_used': (end_memory - start_memory) / 1024 / 1024,  # MB
                'cpu_usage': (start_cpu + end_cpu) / 2,
                'success': success,
                'error': error,
                'timestamp': datetime.now()
            }
            
            self.metrics.append(metrics)
            
            if not success:
                raise Exception(error)
            
            return result
        
        return wrapper
    
    def get_summary(self) -> dict:
        """Get performance summary"""
        if not self.metrics:
            return {}
        
        durations = [m['duration'] for m in self.metrics]
        memory_usage = [m['memory_used'] for m in self.metrics]
        
        return {
            'total_calls': len(self.metrics),
            'success_rate': sum(m['success'] for m in self.metrics) / len(self.metrics),
            'avg_duration': np.mean(durations),
            'p95_duration': np.percentile(durations, 95),
            'p99_duration': np.percentile(durations, 99),
            'avg_memory_mb': np.mean(memory_usage),
            'peak_memory_mb': max(memory_usage),
            'errors': [m for m in self.metrics if not m['success']]
        }

# Usage
monitor = PerformanceMonitor()

class Agent:
    @monitor.measure
    def execute_task(self, task: str):
        # Task execution
        time.sleep(0.1)  # Simulate work
        return "result"

agent = Agent()
agent.execute_task("test task")

print(monitor.get_summary())
```

### 3. Quality Metrics

```python
from typing import Tuple
import difflib

class QualityEvaluator:
    """Evaluate quality of agent outputs"""
    
    def __init__(self):
        self.evaluations = []
    
    def evaluate_accuracy(
        self,
        predicted: str,
        expected: str,
        fuzzy: bool = True
    ) -> float:
        """Evaluate accuracy of output"""
        
        if fuzzy:
            # Fuzzy string matching
            ratio = difflib.SequenceMatcher(
                None,
                predicted.lower(),
                expected.lower()
            ).ratio()
            return ratio
        else:
            # Exact match
            return 1.0 if predicted == expected else 0.0
    
    def evaluate_completeness(
        self,
        output: dict,
        required_fields: List[str]
    ) -> float:
        """Check if all required fields are present"""
        
        present = sum(
            1 for field in required_fields
            if field in output and output[field] is not None
        )
        
        return present / len(required_fields) if required_fields else 1.0
    
    def evaluate_coherence(
        self,
        text: str,
        llm_judge = None
    ) -> float:
        """Evaluate coherence of text output"""
        
        if llm_judge:
            # Use LLM as judge
            prompt = f"""
            Rate the coherence of this text from 0-10:
            "{text}"
            
            Consider: logical flow, consistency, clarity
            Output only a number.
            """
            
            score = float(llm_judge.complete(prompt)) / 10
            return score
        else:
            # Simple heuristics
            sentences = text.split('.')
            if len(sentences) < 2:
                return 0.5
            
            # Check for basic coherence markers
            score = 1.0
            
            # Penalize very short sentences
            avg_length = np.mean([len(s.split()) for s in sentences])
            if avg_length < 5:
                score -= 0.2
            
            # Penalize repetition
            unique_words = len(set(text.lower().split()))
            total_words = len(text.split())
            diversity = unique_words / total_words if total_words > 0 else 0
            
            if diversity < 0.3:
                score -= 0.3
            
            return max(0, score)
    
    def evaluate_safety(
        self,
        output: str,
        forbidden_patterns: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """Check output for safety violations"""
        
        if forbidden_patterns is None:
            forbidden_patterns = [
                'DROP TABLE',
                'DELETE FROM',
                'sudo rm -rf',
                'format C:',
                'password',
                'api_key',
                'secret'
            ]
        
        violations = []
        for pattern in forbidden_patterns:
            if pattern.lower() in output.lower():
                violations.append(pattern)
        
        return len(violations) == 0, violations

# Composite quality score
class QualityScorer:
    """Calculate composite quality score"""
    
    def __init__(self, weights: dict = None):
        self.weights = weights or {
            'accuracy': 0.3,
            'completeness': 0.2,
            'coherence': 0.2,
            'safety': 0.3
        }
        self.evaluator = QualityEvaluator()
    
    def score(
        self,
        output: Any,
        expected: Any = None,
        required_fields: List[str] = None
    ) -> dict:
        """Calculate weighted quality score"""
        
        scores = {}
        
        # Accuracy (if expected output provided)
        if expected:
            scores['accuracy'] = self.evaluator.evaluate_accuracy(
                str(output),
                str(expected)
            )
        
        # Completeness (if structured output)
        if isinstance(output, dict) and required_fields:
            scores['completeness'] = self.evaluator.evaluate_completeness(
                output,
                required_fields
            )
        
        # Coherence (for text)
        if isinstance(output, str):
            scores['coherence'] = self.evaluator.evaluate_coherence(output)
        
        # Safety
        safe, violations = self.evaluator.evaluate_safety(str(output))
        scores['safety'] = 1.0 if safe else 0.0
        
        # Weighted average
        total_weight = 0
        weighted_sum = 0
        
        for metric, score in scores.items():
            if metric in self.weights:
                weighted_sum += score * self.weights[metric]
                total_weight += self.weights[metric]
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        return {
            'overall': final_score,
            'breakdown': scores,
            'safety_violations': violations if not safe else []
        }
```

---

## 🧪 Evaluation Frameworks

### 1. Offline Evaluation (Before Deployment)

```python
from typing import List, Callable
import json

class TestCase:
    """Single test case for agent evaluation"""
    
    def __init__(
        self,
        name: str,
        input: Any,
        expected_output: Any = None,
        expected_behavior: dict = None,
        timeout: int = 60
    ):
        self.name = name
        self.input = input
        self.expected_output = expected_output
        self.expected_behavior = expected_behavior
        self.timeout = timeout
        self.result = None

class TestSuite:
    """Collection of test cases"""
    
    def __init__(self, name: str):
        self.name = name
        self.test_cases = []
        self.results = []
    
    def add_test(self, test_case: TestCase):
        """Add test case to suite"""
        self.test_cases.append(test_case)
    
    def run(self, agent: Any, verbose: bool = True) -> dict:
        """Run all tests against agent"""
        
        passed = 0
        failed = 0
        errors = 0
        
        for test in self.test_cases:
            if verbose:
                print(f"\nRunning test: {test.name}")
            
            try:
                # Run agent
                start_time = time.time()
                output = agent.execute(test.input)
                duration = time.time() - start_time
                
                # Check timeout
                if duration > test.timeout:
                    test.result = "timeout"
                    failed += 1
                    continue
                
                # Evaluate output
                if test.expected_output:
                    evaluator = QualityEvaluator()
                    accuracy = evaluator.evaluate_accuracy(
                        str(output),
                        str(test.expected_output)
                    )
                    
                    if accuracy > 0.9:
                        test.result = "passed"
                        passed += 1
                    else:
                        test.result = "failed"
                        failed += 1
                
                # Check behavior
                if test.expected_behavior:
                    behavior_match = self._check_behavior(
                        agent,
                        test.expected_behavior
                    )
                    
                    if behavior_match:
                        test.result = "passed"
                        passed += 1
                    else:
                        test.result = "failed"
                        failed += 1
                
            except Exception as e:
                test.result = f"error: {e}"
                errors += 1
            
            if verbose:
                print(f"  Result: {test.result}")
        
        # Summary
        total = len(self.test_cases)
        pass_rate = passed / total if total > 0 else 0
        
        return {
            'suite': self.name,
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'pass_rate': pass_rate,
            'tests': [
                {
                    'name': t.name,
                    'result': t.result
                }
                for t in self.test_cases
            ]
        }
    
    def _check_behavior(self, agent: Any, expected: dict) -> bool:
        """Check if agent exhibited expected behavior"""
        
        # Check tools used
        if 'tools_used' in expected:
            if not all(tool in agent.tools_used for tool in expected['tools_used']):
                return False
        
        # Check number of steps
        if 'max_steps' in expected:
            if agent.steps_taken > expected['max_steps']:
                return False
        
        return True

# Example test suite
def create_research_agent_tests() -> TestSuite:
    """Create test suite for research agent"""
    
    suite = TestSuite("Research Agent Tests")
    
    # Test 1: Simple research
    suite.add_test(TestCase(
        name="Simple Topic Research",
        input="Research quantum computing basics",
        expected_behavior={
            'tools_used': ['web_search'],
            'max_steps': 5
        }
    ))
    
    # Test 2: Complex research
    suite.add_test(TestCase(
        name="Multi-source Research",
        input="Compare top 3 AI frameworks with pros and cons",
        expected_behavior={
            'tools_used': ['web_search', 'summarizer'],
            'max_steps': 10
        }
    ))
    
    # Test 3: Error handling
    suite.add_test(TestCase(
        name="Handle Invalid Topic",
        input="Research ajkfhaskjdhfkajsdhf",
        expected_output="Could not find information on this topic"
    ))
    
    return suite
```

### 2. Online Evaluation (In Production)

```python
from abc import ABC, abstractmethod
import random

class OnlineEvaluator:
    """Evaluate agent performance in production"""
    
    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = sample_rate
        self.metrics = {
            'latency': [],
            'success': [],
            'user_satisfaction': [],
            'costs': []
        }
    
    def should_evaluate(self) -> bool:
        """Decide if this request should be evaluated"""
        return random.random() < self.sample_rate
    
    def evaluate_request(
        self,
        request: dict,
        response: dict,
        metadata: dict
    ) -> dict:
        """Evaluate a single request/response"""
        
        evaluation = {
            'timestamp': datetime.now(),
            'request_id': metadata.get('request_id'),
            'latency': metadata.get('duration'),
            'success': response.get('success', False),
            'tokens': metadata.get('tokens_used'),
            'cost': metadata.get('cost')
        }
        
        # Record metrics
        self.metrics['latency'].append(evaluation['latency'])
        self.metrics['success'].append(evaluation['success'])
        self.metrics['costs'].append(evaluation['cost'])
        
        return evaluation
    
    def get_dashboard_metrics(self) -> dict:
        """Get metrics for dashboard"""
        
        return {
            'requests_evaluated': len(self.metrics['success']),
            'avg_latency': np.mean(self.metrics['latency']) if self.metrics['latency'] else 0,
            'p99_latency': np.percentile(self.metrics['latency'], 99) if self.metrics['latency'] else 0,
            'success_rate': np.mean(self.metrics['success']) if self.metrics['success'] else 0,
            'avg_cost': np.mean(self.metrics['costs']) if self.metrics['costs'] else 0,
            'total_cost': sum(self.metrics['costs']) if self.metrics['costs'] else 0
        }

# A/B Testing
class ABTestEvaluator:
    """Compare two agent versions"""
    
    def __init__(
        self,
        agent_a: Any,
        agent_b: Any,
        traffic_split: float = 0.5
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.traffic_split = traffic_split
        
        self.results_a = []
        self.results_b = []
    
    def route_request(self, request: dict) -> Tuple[Any, str]:
        """Route request to agent A or B"""
        
        if random.random() < self.traffic_split:
            return self.agent_a, "A"
        else:
            return self.agent_b, "B"
    
    def process_request(self, request: dict) -> dict:
        """Process request and track results"""
        
        agent, version = self.route_request(request)
        
        start_time = time.time()
        response = agent.execute(request)
        duration = time.time() - start_time
        
        result = {
            'version': version,
            'request': request,
            'response': response,
            'duration': duration,
            'success': response.get('success', False)
        }
        
        if version == "A":
            self.results_a.append(result)
        else:
            self.results_b.append(result)
        
        return response
    
    def get_comparison(self) -> dict:
        """Compare performance of A vs B"""
        
        def calculate_stats(results):
            if not results:
                return {}
            
            return {
                'count': len(results),
                'success_rate': sum(r['success'] for r in results) / len(results),
                'avg_duration': np.mean([r['duration'] for r in results]),
                'p95_duration': np.percentile([r['duration'] for r in results], 95)
            }
        
        stats_a = calculate_stats(self.results_a)
        stats_b = calculate_stats(self.results_b)
        
        # Statistical significance test
        if len(self.results_a) > 30 and len(self.results_b) > 30:
            from scipy import stats
            
            success_a = [r['success'] for r in self.results_a]
            success_b = [r['success'] for r in self.results_b]
            
            t_stat, p_value = stats.ttest_ind(success_a, success_b)
            significant = p_value < 0.05
        else:
            p_value = None
            significant = False
        
        return {
            'agent_a': stats_a,
            'agent_b': stats_b,
            'p_value': p_value,
            'significant_difference': significant,
            'winner': 'A' if stats_a.get('success_rate', 0) > stats_b.get('success_rate', 0) else 'B'
        }
```

### 3. Human-in-the-Loop Evaluation

```python
class HumanEvaluator:
    """Collect and analyze human feedback"""
    
    def __init__(self):
        self.feedback_queue = []
        self.ratings = []
        self.comments = []
    
    def request_feedback(
        self,
        task: str,
        agent_output: Any,
        user_id: str
    ) -> str:
        """Request human feedback on output"""
        
        feedback_id = f"feedback_{datetime.now().timestamp()}"
        
        self.feedback_queue.append({
            'id': feedback_id,
            'task': task,
            'output': agent_output,
            'user_id': user_id,
            'status': 'pending'
        })
        
        return feedback_id
    
    def submit_feedback(
        self,
        feedback_id: str,
        rating: int,  # 1-5 stars
        comment: str = None,
        corrections: dict = None
    ):
        """Submit human feedback"""
        
        # Find feedback request
        for item in self.feedback_queue:
            if item['id'] == feedback_id:
                item['status'] = 'completed'
                item['rating'] = rating
                item['comment'] = comment
                item['corrections'] = corrections
                
                self.ratings.append(rating)
                if comment:
                    self.comments.append(comment)
                
                break
    
    def get_human_metrics(self) -> dict:
        """Get aggregated human feedback metrics"""
        
        if not self.ratings:
            return {'message': 'No feedback yet'}
        
        return {
            'total_feedback': len(self.ratings),
            'avg_rating': np.mean(self.ratings),
            'rating_distribution': {
                i: self.ratings.count(i)
                for i in range(1, 6)
            },
            'satisfaction_rate': sum(r >= 4 for r in self.ratings) / len(self.ratings),
            'recent_comments': self.comments[-5:]
        }
    
    def learn_from_corrections(
        self,
        threshold: int = 5
    ) -> List[dict]:
        """Identify patterns from human corrections"""
        
        corrections = [
            item for item in self.feedback_queue
            if item.get('corrections') and item['status'] == 'completed'
        ]
        
        if len(corrections) < threshold:
            return []
        
        # Analyze correction patterns
        patterns = {}
        
        for item in corrections:
            for field, correction in item['corrections'].items():
                if field not in patterns:
                    patterns[field] = []
                patterns[field].append(correction)
        
        # Find common corrections
        common_patterns = []
        for field, corrections_list in patterns.items():
            if len(corrections_list) >= threshold:
                common_patterns.append({
                    'field': field,
                    'frequency': len(corrections_list),
                    'examples': corrections_list[:3]
                })
        
        return common_patterns
```

---

## 📈 Production Monitoring

### Real-Time Dashboard

```python
from typing import Deque
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class AgentDashboard:
    """Real-time monitoring dashboard"""
    
    def __init__(self, window_minutes: int = 60):
        self.window = timedelta(minutes=window_minutes)
        
        # Metrics storage (using deque for efficient rotation)
        self.latencies: Deque = deque(maxlen=1000)
        self.success_rates: Deque = deque(maxlen=1000)
        self.costs: Deque = deque(maxlen=1000)
        self.timestamps: Deque = deque(maxlen=1000)
        
        # Alerts
        self.alerts = []
        self.alert_thresholds = {
            'latency_p99': 10.0,  # seconds
            'success_rate': 0.9,   # 90%
            'cost_per_hour': 100   # dollars
        }
    
    def record_metric(
        self,
        latency: float,
        success: bool,
        cost: float
    ):
        """Record a metric point"""
        
        now = datetime.now()
        
        self.timestamps.append(now)
        self.latencies.append(latency)
        self.success_rates.append(success)
        self.costs.append(cost)
        
        # Check for alerts
        self._check_alerts()
    
    def _check_alerts(self):
        """Check if any thresholds are breached"""
        
        if len(self.latencies) < 10:
            return  # Not enough data
        
        # Check latency
        p99_latency = np.percentile(list(self.latencies), 99)
        if p99_latency > self.alert_thresholds['latency_p99']:
            self._trigger_alert(
                f"High latency: {p99_latency:.2f}s (threshold: {self.alert_thresholds['latency_p99']}s)"
            )
        
        # Check success rate
        recent_success = list(self.success_rates)[-100:]
        success_rate = sum(recent_success) / len(recent_success)
        if success_rate < self.alert_thresholds['success_rate']:
            self._trigger_alert(
                f"Low success rate: {success_rate:.2%} (threshold: {self.alert_thresholds['success_rate']:.2%})"
            )
        
        # Check cost
        recent_window = datetime.now() - timedelta(hours=1)
        recent_costs = [
            cost for cost, ts in zip(self.costs, self.timestamps)
            if ts > recent_window
        ]
        
        if recent_costs:
            hourly_cost = sum(recent_costs)
            if hourly_cost > self.alert_thresholds['cost_per_hour']:
                self._trigger_alert(
                    f"High cost: ${hourly_cost:.2f}/hour (threshold: ${self.alert_thresholds['cost_per_hour']})"
                )
    
    def _trigger_alert(self, message: str):
        """Trigger an alert"""
        
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'severity': 'high'
        }
        
        self.alerts.append(alert)
        
        # In production, would send to PagerDuty, Slack, etc.
        print(f"🚨 ALERT: {message}")
    
    def get_current_metrics(self) -> dict:
        """Get current dashboard metrics"""
        
        if not self.latencies:
            return {'status': 'no_data'}
        
        # Get recent data
        recent_window = datetime.now() - self.window
        recent_indices = [
            i for i, ts in enumerate(self.timestamps)
            if ts > recent_window
        ]
        
        if not recent_indices:
            return {'status': 'no_recent_data'}
        
        recent_latencies = [self.latencies[i] for i in recent_indices]
        recent_success = [self.success_rates[i] for i in recent_indices]
        recent_costs = [self.costs[i] for i in recent_indices]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'window_minutes': self.window.seconds // 60,
            'requests': len(recent_indices),
            'latency': {
                'mean': np.mean(recent_latencies),
                'p50': np.percentile(recent_latencies, 50),
                'p95': np.percentile(recent_latencies, 95),
                'p99': np.percentile(recent_latencies, 99)
            },
            'success_rate': sum(recent_success) / len(recent_success),
            'cost': {
                'total': sum(recent_costs),
                'per_request': np.mean(recent_costs)
            },
            'alerts': self.alerts[-5:]  # Recent alerts
        }
    
    def plot_metrics(self):
        """Generate metric plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Latency over time
        axes[0, 0].plot(self.timestamps, self.latencies)
        axes[0, 0].set_title('Latency Over Time')
        axes[0, 0].set_ylabel('Seconds')
        
        # Success rate
        window_size = 20
        success_moving_avg = [
            np.mean(list(self.success_rates)[max(0, i-window_size):i+1])
            for i in range(len(self.success_rates))
        ]
        axes[0, 1].plot(self.timestamps, success_moving_avg)
        axes[0, 1].set_title('Success Rate (Moving Average)')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_ylim([0, 1])
        
        # Cost distribution
        axes[1, 0].hist(self.costs, bins=20)
        axes[1, 0].set_title('Cost Distribution')
        axes[1, 0].set_xlabel('Cost ($)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Alerts timeline
        alert_times = [a['timestamp'] for a in self.alerts]
        axes[1, 1].scatter(alert_times, [1] * len(alert_times), c='red', s=100)
        axes[1, 1].set_title('Alerts')
        axes[1, 1].set_ylabel('Alert')
        axes[1, 1].set_ylim([0, 2])
        
        plt.tight_layout()
        plt.show()
```

---

## 🎯 Defining Success Criteria

### The "Good Enough" Framework

```python
class ReadinessEvaluator:
    """Determine if agent is ready for production"""
    
    def __init__(self):
        self.criteria = {
            'functional': {
                'success_rate': 0.95,
                'critical_failure_rate': 0.001
            },
            'performance': {
                'p95_latency': 5.0,  # seconds
                'p99_latency': 10.0
            },
            'quality': {
                'accuracy': 0.9,
                'user_satisfaction': 4.0  # out of 5
            },
            'cost': {
                'per_request': 0.10,  # dollars
                'monthly_budget': 1000
            },
            'safety': {
                'harmful_output_rate': 0.0001,
                'data_leak_rate': 0.0
            }
        }
    
    def evaluate_readiness(
        self,
        test_results: dict,
        production_preview: dict = None
    ) -> dict:
        """Comprehensive readiness evaluation"""
        
        readiness = {
            'ready': True,
            'score': 0,
            'blockers': [],
            'warnings': [],
            'passed': []
        }
        
        # Check each criterion
        checks = [
            self._check_functional(test_results),
            self._check_performance(test_results),
            self._check_quality(test_results),
            self._check_cost(test_results),
            self._check_safety(test_results)
        ]
        
        # Aggregate results
        total_score = 0
        max_score = len(checks)
        
        for check in checks:
            total_score += check['score']
            
            if check['status'] == 'pass':
                readiness['passed'].append(check['category'])
            elif check['status'] == 'warning':
                readiness['warnings'].append(check['message'])
            else:  # fail
                readiness['blockers'].append(check['message'])
                readiness['ready'] = False
        
        readiness['score'] = total_score / max_score
        
        # Production preview results (if available)
        if production_preview:
            preview_check = self._check_production_preview(production_preview)
            if preview_check['status'] == 'fail':
                readiness['ready'] = False
                readiness['blockers'].append(preview_check['message'])
        
        return readiness
    
    def _check_functional(self, results: dict) -> dict:
        """Check functional requirements"""
        
        success_rate = results.get('success_rate', 0)
        critical_failures = results.get('critical_failures', 0)
        total = results.get('total_tests', 1)
        
        critical_rate = critical_failures / total if total > 0 else 1
        
        if success_rate >= self.criteria['functional']['success_rate'] and \
           critical_rate <= self.criteria['functional']['critical_failure_rate']:
            return {
                'category': 'functional',
                'status': 'pass',
                'score': 1.0,
                'message': f'Success rate: {success_rate:.2%}'
            }
        else:
            return {
                'category': 'functional',
                'status': 'fail',
                'score': 0.0,
                'message': f'Low success rate: {success_rate:.2%} (need {self.criteria["functional"]["success_rate"]:.2%})'
            }
    
    def _check_performance(self, results: dict) -> dict:
        """Check performance requirements"""
        
        p95 = results.get('p95_latency', float('inf'))
        p99 = results.get('p99_latency', float('inf'))
        
        if p95 <= self.criteria['performance']['p95_latency'] and \
           p99 <= self.criteria['performance']['p99_latency']:
            return {
                'category': 'performance',
                'status': 'pass',
                'score': 1.0,
                'message': f'P95: {p95:.2f}s, P99: {p99:.2f}s'
            }
        elif p99 <= self.criteria['performance']['p99_latency'] * 1.5:
            return {
                'category': 'performance',
                'status': 'warning',
                'score': 0.5,
                'message': f'Borderline performance - P99: {p99:.2f}s'
            }
        else:
            return {
                'category': 'performance',
                'status': 'fail',
                'score': 0.0,
                'message': f'Too slow - P99: {p99:.2f}s (max: {self.criteria["performance"]["p99_latency"]}s)'
            }
    
    def _check_quality(self, results: dict) -> dict:
        """Check quality requirements"""
        # Implementation similar to above
        pass
    
    def _check_cost(self, results: dict) -> dict:
        """Check cost requirements"""
        # Implementation similar to above
        pass
    
    def _check_safety(self, results: dict) -> dict:
        """Check safety requirements"""
        # Implementation similar to above
        pass
```

---

## 📊 Metrics That Matter

### Business Metrics

```python
class BusinessMetrics:
    """Track business impact of agent"""
    
    def __init__(self):
        self.before_agent = {}
        self.after_agent = {}
    
    def calculate_roi(
        self,
        agent_cost: float,
        time_saved_hours: float,
        hourly_rate: float,
        error_reduction_value: float
    ) -> dict:
        """Calculate return on investment"""
        
        # Benefits
        labor_savings = time_saved_hours * hourly_rate
        quality_improvement = error_reduction_value
        total_benefits = labor_savings + quality_improvement
        
        # ROI
        roi = ((total_benefits - agent_cost) / agent_cost) * 100
        
        # Payback period (months)
        monthly_benefit = total_benefits / 12
        monthly_cost = agent_cost / 12
        
        if monthly_benefit > monthly_cost:
            payback_months = agent_cost / (monthly_benefit - monthly_cost)
        else:
            payback_months = float('inf')
        
        return {
            'roi_percentage': roi,
            'payback_months': payback_months,
            'annual_savings': total_benefits - agent_cost,
            'breakdown': {
                'labor_savings': labor_savings,
                'quality_improvement': quality_improvement,
                'total_benefits': total_benefits,
                'agent_cost': agent_cost
            }
        }
    
    def track_kpis(self) -> dict:
        """Track key performance indicators"""
        
        return {
            'efficiency': {
                'tasks_per_hour': 10,  # vs 2 for human
                'accuracy_rate': 0.95,  # vs 0.85 for human
                'availability': '24/7'   # vs 40 hours/week
            },
            'customer_impact': {
                'response_time': '< 1 minute',  # vs hours
                'satisfaction_score': 4.5,      # out of 5
                'resolution_rate': 0.88         # first contact
            },
            'operational': {
                'cost_per_task': 0.05,    # vs $5 human
                'scalability': 'unlimited', # vs limited hiring
                'consistency': 0.98         # vs 0.70 human variance
            }
        }
```

---

## 🎯 Complete Evaluation System

```python
class ComprehensiveEvaluationSystem:
    """Complete evaluation system for production agents"""
    
    def __init__(self, agent: Any):
        self.agent = agent
        
        # Components
        self.offline_evaluator = TestSuite("Comprehensive Tests")
        self.online_evaluator = OnlineEvaluator()
        self.human_evaluator = HumanEvaluator()
        self.performance_monitor = PerformanceMonitor()
        self.quality_scorer = QualityScorer()
        self.dashboard = AgentDashboard()
        self.readiness_evaluator = ReadinessEvaluator()
    
    def run_full_evaluation(self) -> dict:
        """Run complete evaluation pipeline"""
        
        print("🔬 Starting Comprehensive Evaluation...")
        
        # 1. Offline tests
        print("\n1️⃣ Running offline tests...")
        offline_results = self.offline_evaluator.run(self.agent)
        
        # 2. Performance benchmarks
        print("\n2️⃣ Running performance benchmarks...")
        perf_results = self._run_performance_benchmarks()
        
        # 3. Quality evaluation
        print("\n3️⃣ Evaluating output quality...")
        quality_results = self._evaluate_quality()
        
        # 4. Safety checks
        print("\n4️⃣ Running safety checks...")
        safety_results = self._run_safety_checks()
        
        # 5. Cost analysis
        print("\n5️⃣ Analyzing costs...")
        cost_results = self._analyze_costs()
        
        # 6. Readiness assessment
        print("\n6️⃣ Assessing production readiness...")
        readiness = self.readiness_evaluator.evaluate_readiness({
            'success_rate': offline_results['pass_rate'],
            'p95_latency': perf_results['p95_latency'],
            'p99_latency': perf_results['p99_latency'],
            'accuracy': quality_results['accuracy'],
            'cost_per_request': cost_results['per_request']
        })
        
        # Final report
        return {
            'timestamp': datetime.now().isoformat(),
            'agent_version': getattr(self.agent, 'version', 'unknown'),
            'evaluation_results': {
                'offline_tests': offline_results,
                'performance': perf_results,
                'quality': quality_results,
                'safety': safety_results,
                'cost': cost_results
            },
            'readiness': readiness,
            'recommendation': self._make_recommendation(readiness)
        }
    
    def _make_recommendation(self, readiness: dict) -> str:
        """Make deployment recommendation"""
        
        if readiness['ready'] and readiness['score'] > 0.9:
            return "✅ READY FOR PRODUCTION - All criteria met"
        elif readiness['ready'] and readiness['score'] > 0.8:
            return "⚠️ READY WITH CAUTION - Monitor closely after deployment"
        elif readiness['score'] > 0.7:
            return "🔧 ALMOST READY - Address warnings before deployment"
        else:
            return "❌ NOT READY - Critical issues must be resolved"
```

---

## 💡 Key Takeaways

1. **Measure what matters** - Not all metrics are equally important
2. **Test before deploy** - Comprehensive offline evaluation saves pain
3. **Monitor continuously** - Production behavior differs from tests
4. **Human feedback is gold** - Users know quality when they see it
5. **Safety first** - One bad output can destroy trust
6. **Cost matters** - Track and optimize for sustainability
7. **Define "good enough"** - Perfect is the enemy of deployed

**The goal isn't perfect agents - it's agents that reliably deliver value.**

---

## 🚀 Final Challenge: Build Your Evaluation Suite

Create a complete evaluation system for your agent:

```python
class YourAgentEvaluator:
    """Custom evaluator for your agent"""
    
    def __init__(self, agent):
        self.agent = agent
        # Your implementation
    
    def create_test_suite(self) -> TestSuite:
        """Define your test cases"""
        pass
    
    def define_success_criteria(self) -> dict:
        """Define what success looks like"""
        pass
    
    def run_evaluation(self) -> dict:
        """Run complete evaluation"""
        pass
    
    def generate_report(self, results: dict) -> str:
        """Generate readable report"""
        pass

# Run it!
evaluator = YourAgentEvaluator(your_agent)
results = evaluator.run_evaluation()
print(evaluator.generate_report(results))
```

---

## 🎓 What You've Accomplished

Through these 10 lessons, you've learned:

✅ **Lesson 1:** What makes agents different from models  
✅ **Lesson 2:** The core agent loop  
✅ **Lesson 3:** Memory systems for persistence  
✅ **Lesson 4:** Building your first agent  
✅ **Lesson 5:** Planning and reasoning strategies  
✅ **Lesson 6:** Tool integration  
✅ **Lesson 7:** Error handling and recovery  
✅ **Lesson 8:** Multi-agent systems  
✅ **Lesson 9:** Environment interfaces  
✅ **Lesson 10:** Evaluation and metrics  

**You now have the foundation to build production-ready AI agents!** 🎉

---

## 🚀 What's Next?

### Immediate Next Steps
1. **Build something real** - Apply these concepts to a real problem
2. **Join the community** - Share your agents and learn from others
3. **Contribute** - Open source your tools and frameworks
4. **Stay updated** - This field moves fast!

### Advanced Topics to Explore
- **Reinforcement Learning** for agent improvement
- **Constitutional AI** for safe agents
- **Mixture of Agents** architectures
- **Agent marketplaces** and ecosystems
- **Edge deployment** of agents

### Resources for Continued Learning
- **Papers:** Read the latest research on arxiv.org
- **Communities:** Join Discord/Slack groups focused on agents
- **Frameworks:** Master LangGraph, CrewAI, AutoGen
- **Build:** Create and share your own agent frameworks

---

## 🙏 Thank You!

Congratulations on completing **Agentic Foundations**! 

You've gone from understanding what agents are to knowing how to build, test, and deploy them. This is just the beginning of your journey.

**Remember:**
- Every expert was once a beginner
- The best way to learn is by building
- Share your knowledge to help others
- The agent revolution is just beginning

Now go build something amazing! 🚀

---

## 📝 Your Final Reflection

*Take a moment to reflect on your journey:*

**What I've learned:**
- 

**What surprised me:**
- 

**What I want to build next:**
- 

**My biggest challenge:**
- 

**My proudest moment:**
- 

---

**Course Status:** ✅ COMPLETE!  
**Total Time:** ~10-15 hours  
**Difficulty:** Progressed from ⭐ to ⭐⭐⭐⭐⭐  
**Achievement:** You're now ready to build production AI agents!

**Final Message:** The future needs builders like you. Go make it happen! 💪
