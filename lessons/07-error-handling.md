# Lesson 7: Error Handling & Recovery

> **What you'll learn:** How to build resilient agents that gracefully handle failures, recover from errors, and continue working toward their goals even when things go wrong.

---

## 🎯 Learning Objectives

By the end of this lesson, you'll understand:
- Why error handling is critical for autonomous agents
- Different types of errors agents encounter
- Recovery strategies (retry, fallback, replan)
- Building resilient agent architectures
- Debugging and monitoring agents in production
- Graceful degradation patterns

---

## 🚨 The Reality: Agents Operate in Chaos

Here's what WILL happen to your agent in production:

```
❌ APIs go down
❌ Rate limits hit
❌ Network timeouts
❌ LLMs hallucinate
❌ Tools return unexpected formats
❌ Disk fills up
❌ Files don't exist
❌ Permissions denied
❌ Dependencies break
❌ User inputs are wild
```

**An agent without error handling is like a car without brakes - dangerous and unreliable.**

---

## 🔍 Types of Errors

### 1. **Transient Errors** (Temporary, Can Retry)

Errors that might succeed if you try again:

```python
# Network timeouts
TimeoutError: Request timed out after 30s

# Rate limits
RateLimitError: Too many requests, retry after 60s

# Temporary service issues
ServiceUnavailable: API temporarily unavailable

# Resource conflicts
LockError: Resource is locked by another process
```

**Strategy:** Retry with exponential backoff

### 2. **Permanent Errors** (Won't Succeed, Need Alternative)

Errors that won't be fixed by retrying:

```python
# Bad parameters
ValueError: Invalid email format

# Authorization issues
PermissionError: Access denied to file

# Resource doesn't exist
FileNotFoundError: No such file

# Logic errors
TypeError: Expected string, got int
```

**Strategy:** Fallback to alternative approach or fail gracefully

### 3. **Semantic Errors** (LLM Mistakes)

When the LLM does something wrong:

```python
# Wrong tool selected
Agent chose 'delete_file' instead of 'read_file'

# Invalid parameters
Agent passed {"query": 123} instead of string

# Hallucinated data
Agent made up a URL that doesn't exist

# Infinite loops
Agent keeps searching for the same thing
```

**Strategy:** Validation, reflection, and replanning

### 4. **Resource Exhaustion**

Running out of resources:

```python
# Token limits
TokenLimitExceeded: Context window full

# Memory limits
MemoryError: Out of RAM

# Time limits
TimeoutError: Max execution time exceeded

# Cost limits
BudgetExceeded: API cost limit reached
```

**Strategy:** Resource management and budgeting

---

## 🛡️ Error Handling Patterns

### Pattern 1: Retry with Exponential Backoff

```python
import time
from typing import Callable, Any

class RetryStrategy:
    """Retry failed operations with exponential backoff"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_error = None
        
        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    print(f"✅ Succeeded on attempt {attempt + 1}")
                
                return result
                
            except (TimeoutError, ConnectionError, RateLimitError) as e:
                last_error = e
                
                if attempt < self.max_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                    print(f"🔄 Retrying in {delay:.1f}s...")
                    
                    time.sleep(delay)
                else:
                    print(f"❌ All {self.max_attempts} attempts failed")
        
        # All retries exhausted
        raise last_error

# Usage
retry = RetryStrategy(max_attempts=3)

try:
    result = retry.execute(api_call, param1="value")
except Exception as e:
    print(f"Failed after all retries: {e}")
```

### Pattern 2: Fallback Chain

```python
class FallbackChain:
    """Try multiple approaches until one succeeds"""
    
    def __init__(self, strategies: list):
        self.strategies = strategies
    
    def execute(self, task: str) -> Any:
        """Try each strategy until one works"""
        errors = []
        
        for i, strategy in enumerate(self.strategies):
            try:
                print(f"🔄 Trying strategy {i + 1}: {strategy.name}")
                result = strategy.execute(task)
                print(f"✅ Strategy {i + 1} succeeded!")
                return result
                
            except Exception as e:
                errors.append((strategy.name, str(e)))
                print(f"⚠️ Strategy {i + 1} failed: {e}")
                continue
        
        # All strategies failed
        raise RuntimeError(
            f"All strategies failed:\n" + 
            "\n".join(f"- {name}: {err}" for name, err in errors)
        )

# Example: Search with fallbacks
class SearchFallbackChain:
    def __init__(self):
        self.chain = FallbackChain([
            PrimarySearchStrategy(),      # Try best option first
            SecondarySearchStrategy(),    # Fallback to alternative
            CachedSearchStrategy(),       # Use cached results
            ManualSearchStrategy()        # Last resort: ask user
        ])
    
    def search(self, query: str):
        return self.chain.execute(query)
```

### Pattern 3: Circuit Breaker

```python
from datetime import datetime, timedelta
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Prevent cascading failures by stopping calls to failing services"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = CircuitState.HALF_OPEN
                print("🔄 Circuit breaker: Testing if service recovered")
            else:
                raise RuntimeError("Circuit breaker is OPEN - service is failing")
        
        try:
            result = func(*args, **kwargs)
            
            # Success! Reset or close circuit
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                print("✅ Circuit breaker: Service recovered, circuit CLOSED")
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                print(f"⚠️ Circuit breaker: OPEN after {self.failure_count} failures")
            
            raise e

# Usage
breaker = CircuitBreaker(failure_threshold=5, timeout=60)

def call_api():
    try:
        return breaker.call(external_api.request, endpoint="/data")
    except RuntimeError as e:
        # Circuit is open, use fallback
        return cached_data()
```

### Pattern 4: Graceful Degradation

```python
class GracefulAgent:
    """Agent that degrades gracefully when tools fail"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.available_tools = set(tools.keys())
    
    def execute_task(self, task: str):
        """Execute task with graceful degradation"""
        
        # Plan with available tools
        plan = self.create_plan(task, self.available_tools)
        
        for step in plan:
            try:
                result = self.execute_step(step)
                step.result = result
                
            except ToolUnavailableError as e:
                # Tool failed, mark as unavailable
                self.available_tools.discard(step.tool)
                print(f"⚠️ Tool {step.tool} unavailable, replanning...")
                
                # Replan with remaining tools
                alternative_plan = self.create_plan(
                    task,
                    self.available_tools
                )
                
                return self.execute_task_with_plan(alternative_plan)
            
            except CriticalError as e:
                # Can't recover, return partial results
                return {
                    "success": False,
                    "partial_results": [s.result for s in plan if s.result],
                    "error": str(e)
                }
        
        return {
            "success": True,
            "results": [s.result for s in plan]
        }
```

---

## 🔧 Building a Resilient Agent

Let's put it all together:

```python
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExecutionContext:
    """Track execution state and errors"""
    task: str
    attempt: int = 0
    errors: list = None
    partial_results: list = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.partial_results is None:
            self.partial_results = []

class ResilientAgent:
    """Agent with comprehensive error handling and recovery"""
    
    def __init__(
        self,
        llm,
        tools,
        max_retries: int = 3,
        max_replans: int = 2,
        timeout: int = 300
    ):
        self.llm = llm
        self.tools = tools
        self.max_retries = max_retries
        self.max_replans = max_replans
        self.timeout = timeout
        
        # Error handling components
        self.retry_strategy = RetryStrategy(max_attempts=max_retries)
        self.circuit_breakers = {
            tool_name: CircuitBreaker()
            for tool_name in tools.keys()
        }
        
        # Monitoring
        self.error_log = []
        self.success_rate = {}
    
    def execute(self, task: str) -> Dict[str, Any]:
        """Execute task with full error handling"""
        context = ExecutionContext(task=task)
        
        try:
            return self._execute_with_timeout(context)
        except TimeoutError:
            logger.error(f"Task timed out after {self.timeout}s")
            return self._handle_timeout(context)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return self._handle_critical_error(context, e)
    
    def _execute_with_timeout(self, context: ExecutionContext) -> Dict:
        """Execute with timeout protection"""
        
        # Create plan
        plan = self._create_plan_with_retry(context)
        
        if not plan:
            return self._handle_planning_failure(context)
        
        # Execute plan
        for step_num, step in enumerate(plan):
            try:
                result = self._execute_step_safely(step, context)
                context.partial_results.append(result)
                
            except RecoverableError as e:
                # Try to recover
                logger.warning(f"Step {step_num} failed, attempting recovery")
                recovered = self._attempt_recovery(step, context, e)
                
                if recovered:
                    context.partial_results.append(recovered)
                else:
                    # Can't recover, replan
                    return self._replan_and_continue(context, step_num)
            
            except PermanentError as e:
                # Can't continue this path
                logger.error(f"Permanent error in step {step_num}")
                return self._handle_permanent_error(context, step_num, e)
        
        return {
            "success": True,
            "results": context.partial_results,
            "errors": context.errors
        }
    
    def _execute_step_safely(self, step, context):
        """Execute a single step with all safety measures"""
        
        # 1. Validate step parameters
        if not self._validate_step(step):
            raise ValidationError(f"Invalid step parameters: {step}")
        
        # 2. Check circuit breaker
        breaker = self.circuit_breakers.get(step.tool)
        if breaker and breaker.state == CircuitState.OPEN:
            raise ToolUnavailableError(f"Circuit open for {step.tool}")
        
        # 3. Execute with retry
        try:
            result = self.retry_strategy.execute(
                self._execute_tool,
                step.tool,
                step.parameters
            )
            
            # 4. Validate result
            if not self._validate_result(result):
                raise ValidationError(f"Invalid result from {step.tool}")
            
            return result
            
        except Exception as e:
            # Log error
            self.error_log.append({
                "step": step,
                "error": str(e),
                "timestamp": datetime.now()
            })
            raise
    
    def _attempt_recovery(self, step, context, error) -> Optional[Any]:
        """Try to recover from error"""
        
        recovery_strategies = [
            self._try_alternative_parameters,
            self._try_alternative_tool,
            self._try_cached_result,
            self._try_llm_approximation
        ]
        
        for strategy in recovery_strategies:
            try:
                logger.info(f"Trying recovery strategy: {strategy.__name__}")
                result = strategy(step, context, error)
                
                if result:
                    logger.info(f"✅ Recovered using {strategy.__name__}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")
                continue
        
        return None
    
    def _replan_and_continue(self, context, failed_step_num):
        """Replan when execution fails"""
        
        if context.attempt >= self.max_replans:
            return {
                "success": False,
                "error": "Max replans exceeded",
                "partial_results": context.partial_results
            }
        
        context.attempt += 1
        logger.info(f"Replanning (attempt {context.attempt})")
        
        # Create new plan considering what we've learned
        new_plan = self._create_adaptive_plan(
            context.task,
            completed_steps=context.partial_results,
            failed_at=failed_step_num,
            errors=context.errors
        )
        
        if new_plan:
            # Continue with new plan
            return self._execute_with_timeout(context)
        else:
            return {
                "success": False,
                "error": "Could not create recovery plan",
                "partial_results": context.partial_results
            }
    
    def _create_adaptive_plan(self, task, completed_steps, failed_at, errors):
        """Create plan that learns from previous errors"""
        
        prompt = f"""
        Task: {task}
        
        Progress so far:
        - Completed steps: {len(completed_steps)}
        - Failed at step: {failed_at}
        - Errors encountered: {[str(e) for e in errors[-3:]]}
        
        Create an alternative plan that:
        1. Avoids the tools/approaches that failed
        2. Uses lessons learned from errors
        3. Takes a different approach to achieve the goal
        
        Available tools: {self._get_available_tools()}
        """
        
        return self.llm.complete(prompt)
    
    def _get_available_tools(self) -> list:
        """Get tools that aren't circuit-broken"""
        available = []
        for name, breaker in self.circuit_breakers.items():
            if breaker.state != CircuitState.OPEN:
                available.append(name)
        return available
    
    def _handle_permanent_error(self, context, step_num, error):
        """Handle errors that can't be recovered"""
        return {
            "success": False,
            "error": str(error),
            "partial_results": context.partial_results,
            "completed_steps": step_num,
            "recommendation": self._generate_user_guidance(context, error)
        }
    
    def _generate_user_guidance(self, context, error):
        """Generate helpful guidance for the user"""
        prompt = f"""
        The agent encountered an error it couldn't recover from:
        
        Task: {context.task}
        Error: {error}
        Progress: Completed {len(context.partial_results)} steps
        
        Generate helpful guidance for the user:
        1. What went wrong
        2. What was accomplished
        3. What they could try instead
        4. Any manual steps they might need to take
        """
        
        return self.llm.complete(prompt)

# Custom exception types
class RecoverableError(Exception):
    """Error that might be recoverable"""
    pass

class PermanentError(Exception):
    """Error that can't be recovered"""
    pass

class ValidationError(Exception):
    """Invalid parameters or results"""
    pass

class ToolUnavailableError(Exception):
    """Tool is not available"""
    pass
```

---

## 📊 Error Monitoring & Debugging

### Error Dashboard

```python
class ErrorMonitor:
    """Monitor and analyze agent errors"""
    
    def __init__(self):
        self.error_log = []
        self.success_log = []
    
    def log_error(self, error_info: dict):
        """Log an error occurrence"""
        self.error_log.append({
            **error_info,
            "timestamp": datetime.now()
        })
    
    def log_success(self, success_info: dict):
        """Log a successful execution"""
        self.success_log.append({
            **success_info,
            "timestamp": datetime.now()
        })
    
    def get_error_summary(self) -> dict:
        """Get summary of errors"""
        if not self.error_log:
            return {"message": "No errors logged"}
        
        error_types = {}
        for error in self.error_log:
            error_type = error.get("type", "Unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "most_common": max(error_types, key=error_types.get),
            "recent_errors": self.error_log[-5:]
        }
    
    def get_success_rate(self, time_window: int = 3600) -> float:
        """Calculate success rate in time window (seconds)"""
        cutoff = datetime.now() - timedelta(seconds=time_window)
        
        recent_errors = [
            e for e in self.error_log 
            if e["timestamp"] > cutoff
        ]
        recent_successes = [
            s for s in self.success_log 
            if s["timestamp"] > cutoff
        ]
        
        total = len(recent_errors) + len(recent_successes)
        if total == 0:
            return 1.0
        
        return len(recent_successes) / total
    
    def identify_patterns(self):
        """Identify error patterns"""
        # Group errors by type and context
        patterns = {}
        
        for error in self.error_log:
            key = (error.get("type"), error.get("tool"))
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(error)
        
        # Find repeating patterns
        concerning_patterns = {
            k: v for k, v in patterns.items()
            if len(v) >= 3  # 3+ occurrences
        }
        
        return concerning_patterns
```

### Debug Mode

```python
class DebuggableAgent(ResilientAgent):
    """Agent with enhanced debugging capabilities"""
    
    def __init__(self, *args, debug_mode=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_mode = debug_mode
        self.execution_trace = []
    
    def execute(self, task: str):
        """Execute with optional debug output"""
        
        if self.debug_mode:
            print("\n" + "="*60)
            print("🐛 DEBUG MODE ENABLED")
            print("="*60)
            self._print_context()
        
        try:
            result = super().execute(task)
            
            if self.debug_mode:
                self._print_execution_trace()
                self._print_final_state()
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                self._print_debug_info(e)
            raise
    
    def _print_context(self):
        """Print initial context"""
        print("\n📋 INITIAL STATE:")
        print(f"  Available tools: {list(self.tools.keys())}")
        print(f"  Max retries: {self.max_retries}")
        print(f"  Max replans: {self.max_replans}")
    
    def _print_execution_trace(self):
        """Print detailed execution trace"""
        print("\n📊 EXECUTION TRACE:")
        for i, trace in enumerate(self.execution_trace):
            print(f"\n  Step {i+1}: {trace['action']}")
            print(f"    Status: {trace['status']}")
            if trace.get('error'):
                print(f"    Error: {trace['error']}")
    
    def _print_debug_info(self, error):
        """Print comprehensive debug information"""
        print("\n" + "="*60)
        print("🐛 ERROR DEBUG INFORMATION")
        print("="*60)
        print(f"\n❌ Error: {error}")
        print(f"\n📍 Traceback:")
        import traceback
        traceback.print_exc()
        print(f"\n📊 Error Statistics:")
        print(self.error_monitor.get_error_summary())
```

---

## 🎯 Best Practices

### ✅ DO:

1. **Log everything** - You can't debug what you can't see
2. **Fail gracefully** - Return partial results when possible
3. **Be specific** - Clear error messages help recovery
4. **Test error paths** - Intentionally trigger errors in testing
5. **Monitor in production** - Track error rates and patterns
6. **Timeout everything** - No operation should block forever
7. **Validate inputs** - Catch bad data early
8. **Validate outputs** - Don't trust tool results blindly

### ❌ DON'T:

1. **Catch-all exceptions** - `except Exception: pass` hides problems
2. **Retry forever** - Always have max attempts
3. **Ignore errors** - Every error means something
4. **Trust LLM outputs** - Always validate before acting
5. **Forget the user** - Explain what went wrong
6. **Hard-code timeouts** - Make them configurable
7. **Fail silently** - Always communicate failures

---

## 🧪 Testing Error Scenarios

```python
import pytest

class TestResilientAgent:
    """Test agent's error handling"""
    
    @pytest.fixture
    def agent(self):
        return ResilientAgent(llm=mock_llm, tools=mock_tools)
    
    def test_retry_on_timeout(self, agent):
        """Test retry logic for timeouts"""
        mock_tool = Mock(side_effect=[
            TimeoutError(),
            TimeoutError(),
            {"success": True}
        ])
        
        result = agent.retry_strategy.execute(mock_tool)
        
        assert result["success"] == True
        assert mock_tool.call_count == 3
    
    def test_fallback_on_permanent_error(self, agent):
        """Test fallback when primary method fails"""
        # Simulate primary tool failing
        agent.tools['primary'].execute = Mock(
            side_effect=PermissionError()
        )
        
        result = agent.execute("task")
        
        # Should use fallback tool
        assert agent.tools['fallback'].execute.called
    
    def test_graceful_degradation(self, agent):
        """Test partial results on failure"""
        # Simulate failure mid-execution
        agent.tools['tool2'].execute = Mock(
            side_effect=CriticalError()
        )
        
        result = agent.execute("multi-step task")
        
        assert result["success"] == False
        assert len(result["partial_results"]) > 0
    
    def test_circuit_breaker_opens(self, agent):
        """Test circuit breaker after repeated failures"""
        breaker = agent.circuit_breakers['api_tool']
        
        # Trigger multiple failures
        for _ in range(5):
            try:
                breaker.call(lambda: raise_error())
            except:
                pass
        
        assert breaker.state == CircuitState.OPEN
    
    def test_max_replans_limit(self, agent):
        """Test that agent stops after max replans"""
        # Simulate planning always failing
        agent._create_plan = Mock(return_value=None)
        
        result = agent.execute("task")
        
        assert result["success"] == False
        assert "Max replans exceeded" in result["error"]
```

---

## 💡 Key Takeaways

1. **Errors are inevitable** - Plan for them from day one
2. **Categorize errors** - Transient vs permanent, recoverable vs not
3. **Multiple recovery strategies** - Retry, fallback, replan, degrade
4. **Circuit breakers prevent cascading failures**
5. **Always have an exit strategy** - Don't get stuck
6. **Monitoring is essential** - Track errors to improve
7. **Fail gracefully** - Return partial results, explain what happened
8. **Debug mode is invaluable** - Build observability in

**Remember:** A resilient agent isn't one that never fails - it's one that handles failure gracefully and keeps working toward the goal.

---

## 🚀 Hands-On Challenge

### Build a Fault-Tolerant Research Agent

Create an agent that:
1. Searches multiple sources
2. Retries failed searches
3. Falls back to cached data
4. Returns partial results if needed
5. Logs all errors
6. Reports what it couldn't complete

```python
class FaultTolerantResearchAgent(ResilientAgent):
    """Research agent with comprehensive error handling"""
    
    def research(self, topic: str):
        # Your implementation here
        pass

# Test it with:
# - Network failures
# - Rate limits
# - Invalid topics
# - Partial data availability
```

---

## 📚 Further Reading

### Essential Resources
- **Site Reliability Engineering Book** - Google's approach to resilience
- **Release It!** by Michael Nygard - Design patterns for production
- **The Phoenix Project** - Understanding system failures

### Papers
- "Chaos Engineering" - Netflix's approach to resilience testing
- "Fault Injection Testing" - Proactive error discovery

### Tools
- **Chaos Monkey** - Random failure injection
- **Gremlin** - Chaos engineering platform
- **Sentry** - Error tracking and monitoring

---

## ➡️ What's Next?

You now know how to build agents that don't crumble at the first error!

**Next Lesson:** [Multi-Agent Systems](08-multi-agent.md)

Learn how to coordinate multiple agents working together, each with their own error handling!

---

## 🤔 Reflection Questions

1. **What's the most critical error to handle in your use case?**
2. **When should an agent give up vs keep trying?**
3. **How do you balance retry attempts vs. responsiveness?**
4. **What information should you show users when things fail?**
5. **How would you test error scenarios systematically?**

---

## 📝 Your Notes

*Your thoughts on error handling:*

**Errors I'm most worried about:**
- 

**Recovery strategies that fit my use case:**
- 

**Monitoring approach:**
- 

**Questions:**
- 

---

**Lesson Status:** ✅ Complete  
**Estimated Time:** 60-90 minutes  
**Difficulty:** ⭐⭐⭐⭐ (Advanced)  
**Prerequisites:** Lessons 1-6  
**Next:** Multi-Agent Systems