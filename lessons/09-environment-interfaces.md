# Lesson 9: Environment Interfaces

> **What you'll learn:** How to connect agents to the real world through APIs, databases, file systems, web interfaces, and other external systems - turning theoretical agents into practical tools.

---

## 🎯 Learning Objectives

By the end of this lesson, you'll understand:
- Different types of environments agents interact with
- Building adapters and interfaces for external systems
- Authentication and security considerations
- State management across environments
- Sandboxing and safety boundaries
- Real-time vs batch processing interfaces

---

## 🌍 The Interface Challenge

Your agent is smart, but it's trapped in a box:

```
┌────────────────────┐
│   SMART AGENT      │
│  • Can plan ✅     │
│  • Can reason ✅   │
│  • Has memory ✅   │
│  • Has tools ✅    │
└────────────────────┘
         ???
    How to connect?
         ???
┌────────────────────┐
│   REAL WORLD       │
│  • Web APIs        │
│  • Databases       │
│  • File Systems    │
│  • User Interfaces │
│  • IoT Devices     │
└────────────────────┘
```

**Environment Interfaces are the bridges that connect agents to reality.**

---

## 🔌 Types of Environment Interfaces

### 1. API Interfaces

Connecting to web services and REST APIs.

```python
from typing import Dict, Any, Optional
import requests
from datetime import datetime, timedelta
import jwt

class APIInterface:
    """Base interface for external APIs"""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'
    
    def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make authenticated request"""
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise EnvironmentError(f"API timeout after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise EnvironmentError(f"API error: {e}")

# Example: GitHub Interface
class GitHubInterface(APIInterface):
    """Interface to GitHub API"""
    
    def __init__(self, token: str):
        super().__init__(
            base_url="https://api.github.com",
            api_key=token
        )
    
    def create_issue(self, repo: str, title: str, body: str) -> Dict:
        """Create GitHub issue"""
        return self.request(
            'POST',
            f'/repos/{repo}/issues',
            json={'title': title, 'body': body}
        )
    
    def get_pull_requests(self, repo: str) -> list:
        """Get open pull requests"""
        return self.request('GET', f'/repos/{repo}/pulls')
    
    def create_gist(self, files: Dict[str, str], public: bool = False) -> Dict:
        """Create a GitHub Gist"""
        gist_data = {
            'public': public,
            'files': {
                filename: {'content': content}
                for filename, content in files.items()
            }
        }
        return self.request('POST', '/gists', json=gist_data)
```

### 2. Database Interfaces

Connecting to various databases safely.

```python
import sqlite3
import psycopg2
from contextlib import contextmanager
from typing import List, Tuple, Any

class DatabaseInterface:
    """Safe database interface with connection pooling"""
    
    def __init__(self, connection_string: str, pool_size: int = 5):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.connections = []
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        conn = None
        try:
            if self.connections:
                conn = self.connections.pop()
            else:
                conn = self._create_connection()
            
            yield conn
            
        finally:
            if conn:
                if len(self.connections) < self.pool_size:
                    self.connections.append(conn)
                else:
                    conn.close()
    
    def _create_connection(self):
        """Create new database connection"""
        # Parse connection string and create appropriate connection
        if 'sqlite' in self.connection_string:
            return sqlite3.connect(self.connection_string)
        elif 'postgresql' in self.connection_string:
            return psycopg2.connect(self.connection_string)
        else:
            raise ValueError(f"Unsupported database: {self.connection_string}")
    
    def execute_query(
        self,
        query: str,
        params: Tuple = None,
        fetch_all: bool = True
    ) -> Any:
        """Execute query safely with parameterization"""
        
        # Validate query (basic SQL injection prevention)
        if not self._is_safe_query(query):
            raise SecurityError("Potentially unsafe query detected")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch_all:
                return cursor.fetchall()
            else:
                conn.commit()
                return cursor.rowcount
    
    def _is_safe_query(self, query: str) -> bool:
        """Basic query safety check"""
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper and 'WHERE' not in query_upper:
                return False
        
        return True

# Usage with agent
class DatabaseAgent:
    def __init__(self, db_interface: DatabaseInterface):
        self.db = db_interface
    
    def analyze_sales(self, start_date: str, end_date: str):
        """Agent analyzes sales data"""
        query = """
        SELECT 
            product_id,
            SUM(quantity) as total_sold,
            AVG(price) as avg_price
        FROM sales
        WHERE sale_date BETWEEN ? AND ?
        GROUP BY product_id
        ORDER BY total_sold DESC
        """
        
        results = self.db.execute_query(query, (start_date, end_date))
        return self._analyze_results(results)
```

### 3. File System Interface

Safe file system access with sandboxing.

```python
import os
from pathlib import Path
from typing import List, Optional
import shutil
import tempfile

class FileSystemInterface:
    """Sandboxed file system access"""
    
    def __init__(
        self,
        sandbox_dir: str,
        max_file_size: int = 10 * 1024 * 1024  # 10MB
    ):
        self.sandbox = Path(sandbox_dir).resolve()
        self.max_file_size = max_file_size
        
        # Create sandbox if doesn't exist
        self.sandbox.mkdir(parents=True, exist_ok=True)
    
    def _validate_path(self, path: str) -> Path:
        """Ensure path is within sandbox"""
        abs_path = (self.sandbox / path).resolve()
        
        if not abs_path.is_relative_to(self.sandbox):
            raise SecurityError(f"Path escape attempt: {path}")
        
        return abs_path
    
    def read_file(self, path: str) -> str:
        """Read file safely"""
        safe_path = self._validate_path(path)
        
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if safe_path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large: {path}")
        
        return safe_path.read_text()
    
    def write_file(self, path: str, content: str):
        """Write file safely"""
        safe_path = self._validate_path(path)
        
        # Check size
        if len(content.encode()) > self.max_file_size:
            raise ValueError("Content too large")
        
        # Create parent directories
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write atomically
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=safe_path.parent,
            delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # Atomic move
        shutil.move(tmp_path, safe_path)
    
    def list_files(self, pattern: str = "*") -> List[str]:
        """List files matching pattern"""
        files = []
        
        for file_path in self.sandbox.rglob(pattern):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.sandbox)
                files.append(str(rel_path))
        
        return files
    
    def delete_file(self, path: str):
        """Delete file safely"""
        safe_path = self._validate_path(path)
        
        if safe_path.exists():
            if safe_path.is_file():
                safe_path.unlink()
            else:
                raise ValueError(f"Not a file: {path}")

# Agent with file system access
class FileProcessingAgent:
    def __init__(self, fs_interface: FileSystemInterface):
        self.fs = fs_interface
    
    def process_documents(self, pattern: str = "*.txt"):
        """Process all matching documents"""
        files = self.fs.list_files(pattern)
        
        results = []
        for file_path in files:
            content = self.fs.read_file(file_path)
            processed = self._process_content(content)
            
            # Save processed version
            output_path = f"processed/{file_path}"
            self.fs.write_file(output_path, processed)
            
            results.append({
                "original": file_path,
                "processed": output_path,
                "size": len(processed)
            })
        
        return results
```

### 4. Web Browser Interface

Interact with web pages programmatically.

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc

class BrowserInterface:
    """Web browser automation interface"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
    
    def __enter__(self):
        """Start browser session"""
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Use undetected-chromedriver to avoid bot detection
        self.driver = uc.Chrome(options=options)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up browser session"""
        if self.driver:
            self.driver.quit()
    
    def navigate_to(self, url: str):
        """Navigate to URL"""
        self.driver.get(url)
    
    def wait_for_element(
        self,
        selector: str,
        by: By = By.CSS_SELECTOR,
        timeout: int = 10
    ):
        """Wait for element to appear"""
        wait = WebDriverWait(self.driver, timeout)
        return wait.until(
            EC.presence_of_element_located((by, selector))
        )
    
    def extract_text(self, selector: str) -> str:
        """Extract text from element"""
        element = self.driver.find_element(By.CSS_SELECTOR, selector)
        return element.text
    
    def fill_form(self, form_data: Dict[str, str]):
        """Fill out a form"""
        for selector, value in form_data.items():
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            element.clear()
            element.send_keys(value)
    
    def click(self, selector: str):
        """Click an element"""
        element = self.driver.find_element(By.CSS_SELECTOR, selector)
        element.click()
    
    def screenshot(self, filepath: str):
        """Take screenshot"""
        self.driver.save_screenshot(filepath)
    
    def execute_script(self, script: str) -> Any:
        """Execute JavaScript"""
        return self.driver.execute_script(script)

# Web scraping agent
class WebScrapingAgent:
    def __init__(self, browser: BrowserInterface):
        self.browser = browser
    
    def scrape_product_info(self, url: str) -> dict:
        """Scrape product information from webpage"""
        self.browser.navigate_to(url)
        
        # Wait for content to load
        self.browser.wait_for_element('.product-info')
        
        # Extract information
        product_info = {
            'title': self.browser.extract_text('h1.product-title'),
            'price': self.browser.extract_text('.price'),
            'description': self.browser.extract_text('.description'),
            'availability': self.browser.extract_text('.availability')
        }
        
        # Take screenshot for verification
        self.browser.screenshot('product_screenshot.png')
        
        return product_info
```

### 5. Message Queue Interface

Connect to message brokers for async communication.

```python
import pika
import json
from typing import Callable, Any
import redis
from kafka import KafkaProducer, KafkaConsumer

class MessageQueueInterface:
    """Abstract interface for message queues"""
    
    def publish(self, topic: str, message: Any):
        raise NotImplementedError
    
    def subscribe(self, topic: str, callback: Callable):
        raise NotImplementedError

class RabbitMQInterface(MessageQueueInterface):
    """RabbitMQ implementation"""
    
    def __init__(self, host: str = 'localhost', port: int = 5672):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, port=port)
        )
        self.channel = self.connection.channel()
    
    def publish(self, queue: str, message: Any):
        """Publish message to queue"""
        self.channel.queue_declare(queue=queue, durable=True)
        
        self.channel.basic_publish(
            exchange='',
            routing_key=queue,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2  # Make message persistent
            )
        )
    
    def subscribe(self, queue: str, callback: Callable):
        """Subscribe to queue messages"""
        self.channel.queue_declare(queue=queue, durable=True)
        
        def wrapper(ch, method, properties, body):
            message = json.loads(body)
            callback(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=wrapper,
            auto_ack=False
        )
        
        self.channel.start_consuming()

class RedisInterface(MessageQueueInterface):
    """Redis Pub/Sub implementation"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True
        )
        self.pubsub = self.redis_client.pubsub()
    
    def publish(self, channel: str, message: Any):
        """Publish to Redis channel"""
        self.redis_client.publish(channel, json.dumps(message))
    
    def subscribe(self, channel: str, callback: Callable):
        """Subscribe to Redis channel"""
        self.pubsub.subscribe(channel)
        
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                callback(data)

# Async task processing agent
class TaskProcessingAgent:
    def __init__(self, mq: MessageQueueInterface):
        self.mq = mq
    
    def start_processing(self):
        """Start processing tasks from queue"""
        
        def process_task(task: dict):
            print(f"Processing task: {task['id']}")
            
            # Do the work
            result = self._execute_task(task)
            
            # Publish result
            self.mq.publish('results', {
                'task_id': task['id'],
                'result': result,
                'status': 'completed'
            })
        
        # Subscribe to task queue
        self.mq.subscribe('tasks', process_task)
```

---

## 🔐 Security & Authentication

### OAuth2 Authentication

```python
from typing import Optional
import requests
from datetime import datetime, timedelta

class OAuth2Interface:
    """OAuth2 authentication handler"""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        auth_url: str,
        token_url: str
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url
        self.token_url = token_url
        self.access_token = None
        self.token_expiry = None
    
    def get_authorization_url(self, redirect_uri: str, scope: str) -> str:
        """Get OAuth authorization URL"""
        params = {
            'client_id': self.client_id,
            'redirect_uri': redirect_uri,
            'scope': scope,
            'response_type': 'code'
        }
        
        query = '&'.join(f"{k}={v}" for k, v in params.items())
        return f"{self.auth_url}?{query}"
    
    def exchange_code_for_token(self, code: str, redirect_uri: str) -> str:
        """Exchange authorization code for access token"""
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(self.token_url, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data['access_token']
        
        # Calculate expiry
        expires_in = token_data.get('expires_in', 3600)
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
        
        return self.access_token
    
    def get_valid_token(self) -> str:
        """Get valid access token (refresh if needed)"""
        
        if self.access_token and self.token_expiry:
            if datetime.now() < self.token_expiry:
                return self.access_token
        
        # Token expired or doesn't exist
        raise AuthenticationError("Token expired. Re-authentication required.")
    
    def make_authenticated_request(
        self,
        url: str,
        method: str = 'GET',
        **kwargs
    ) -> dict:
        """Make authenticated API request"""
        
        token = self.get_valid_token()
        
        headers = kwargs.pop('headers', {})
        headers['Authorization'] = f'Bearer {token}'
        
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            **kwargs
        )
        response.raise_for_status()
        
        return response.json()
```

### API Key Management

```python
import os
from cryptography.fernet import Fernet
import keyring

class SecureAPIKeyManager:
    """Secure storage and retrieval of API keys"""
    
    def __init__(self, service_name: str = "agent_system"):
        self.service_name = service_name
        self.cipher = None
        self._init_encryption()
    
    def _init_encryption(self):
        """Initialize encryption key"""
        # Get or create encryption key
        key = keyring.get_password(self.service_name, "encryption_key")
        
        if not key:
            key = Fernet.generate_key().decode()
            keyring.set_password(self.service_name, "encryption_key", key)
        
        self.cipher = Fernet(key.encode())
    
    def store_api_key(self, service: str, api_key: str):
        """Securely store API key"""
        encrypted = self.cipher.encrypt(api_key.encode())
        keyring.set_password(
            self.service_name,
            f"api_key_{service}",
            encrypted.decode()
        )
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Retrieve API key"""
        encrypted = keyring.get_password(
            self.service_name,
            f"api_key_{service}"
        )
        
        if encrypted:
            decrypted = self.cipher.decrypt(encrypted.encode())
            return decrypted.decode()
        
        # Try environment variable as fallback
        env_key = f"{service.upper()}_API_KEY"
        return os.environ.get(env_key)
    
    def delete_api_key(self, service: str):
        """Delete stored API key"""
        keyring.delete_password(
            self.service_name,
            f"api_key_{service}"
        )
```

---

## 🎮 Real-Time Interfaces

### WebSocket Interface

```python
import websocket
import json
from threading import Thread
from typing import Callable

class WebSocketInterface:
    """Real-time WebSocket communication"""
    
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.callbacks = {}
    
    def connect(self):
        """Establish WebSocket connection"""
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Run in separate thread
        wst = Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
    
    def _on_open(self, ws):
        print(f"WebSocket connected to {self.url}")
    
    def _on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            event_type = data.get('type', 'default')
            
            if event_type in self.callbacks:
                self.callbacks[event_type](data)
        except json.JSONDecodeError:
            print(f"Invalid message: {message}")
    
    def _on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_msg}")
    
    def send(self, data: dict):
        """Send data through WebSocket"""
        if self.ws:
            self.ws.send(json.dumps(data))
    
    def on(self, event_type: str, callback: Callable):
        """Register event handler"""
        self.callbacks[event_type] = callback
    
    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()

# Real-time monitoring agent
class MonitoringAgent:
    def __init__(self, ws: WebSocketInterface):
        self.ws = ws
        self.metrics = []
        
        # Register handlers
        self.ws.on('metric', self.handle_metric)
        self.ws.on('alert', self.handle_alert)
    
    def handle_metric(self, data: dict):
        """Process incoming metric"""
        self.metrics.append(data)
        
        # Check for anomalies
        if self._is_anomaly(data):
            self.trigger_alert(data)
    
    def handle_alert(self, data: dict):
        """Handle incoming alert"""
        print(f"🚨 ALERT: {data['message']}")
        
        # Take action
        self.respond_to_alert(data)
    
    def _is_anomaly(self, metric: dict) -> bool:
        """Detect anomalies in metrics"""
        # Simple threshold check
        return metric.get('value', 0) > metric.get('threshold', float('inf'))
```

---

## 🏗️ Complete Environment Manager

```python
from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EnvironmentType(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class EnvironmentManager:
    """Manages all environment interfaces for an agent"""
    
    def __init__(self, env_type: EnvironmentType = EnvironmentType.DEVELOPMENT):
        self.env_type = env_type
        self.interfaces = {}
        self.config = self._load_config()
        self._initialize_interfaces()
    
    def _load_config(self) -> dict:
        """Load environment-specific configuration"""
        configs = {
            EnvironmentType.DEVELOPMENT: {
                'sandbox_dir': '/tmp/agent_sandbox',
                'db_connection': 'sqlite:///dev.db',
                'api_timeout': 60,
                'max_file_size': 50 * 1024 * 1024  # 50MB
            },
            EnvironmentType.STAGING: {
                'sandbox_dir': '/var/agent/sandbox',
                'db_connection': 'postgresql://staging_db',
                'api_timeout': 30,
                'max_file_size': 10 * 1024 * 1024  # 10MB
            },
            EnvironmentType.PRODUCTION: {
                'sandbox_dir': '/secure/agent/sandbox',
                'db_connection': 'postgresql://prod_db',
                'api_timeout': 10,
                'max_file_size': 5 * 1024 * 1024  # 5MB
            }
        }
        
        return configs[self.env_type]
    
    def _initialize_interfaces(self):
        """Initialize all interfaces based on environment"""
        
        # File system
        self.interfaces['filesystem'] = FileSystemInterface(
            sandbox_dir=self.config['sandbox_dir'],
            max_file_size=self.config['max_file_size']
        )
        
        # Database
        self.interfaces['database'] = DatabaseInterface(
            connection_string=self.config['db_connection']
        )
        
        # API key manager
        self.interfaces['api_keys'] = SecureAPIKeyManager(
            service_name=f"agent_{self.env_type.value}"
        )
        
        logger.info(f"Initialized {len(self.interfaces)} interfaces "
                   f"for {self.env_type.value} environment")
    
    def get_interface(self, name: str) -> Any:
        """Get specific interface"""
        if name not in self.interfaces:
            raise ValueError(f"Interface '{name}' not found")
        return self.interfaces[name]
    
    def register_interface(self, name: str, interface: Any):
        """Register custom interface"""
        self.interfaces[name] = interface
        logger.info(f"Registered interface: {name}")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all interfaces"""
        health = {}
        
        for name, interface in self.interfaces.items():
            try:
                # Each interface should implement health_check
                if hasattr(interface, 'health_check'):
                    health[name] = interface.health_check()
                else:
                    # Basic connectivity test
                    health[name] = True
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health[name] = False
        
        return health
    
    def cleanup(self):
        """Clean up all interfaces"""
        for name, interface in self.interfaces.items():
            try:
                if hasattr(interface, 'cleanup'):
                    interface.cleanup()
                elif hasattr(interface, 'close'):
                    interface.close()
            except Exception as e:
                logger.error(f"Cleanup failed for {name}: {e}")

# Agent with environment awareness
class EnvironmentAwareAgent:
    """Agent that adapts to its environment"""
    
    def __init__(
        self,
        llm,
        env_manager: EnvironmentManager
    ):
        self.llm = llm
        self.env = env_manager
        
        # Get interfaces
        self.fs = env_manager.get_interface('filesystem')
        self.db = env_manager.get_interface('database')
        self.api_keys = env_manager.get_interface('api_keys')
    
    def execute_task(self, task: str) -> dict:
        """Execute task using available interfaces"""
        
        # Check environment health
        health = self.env.health_check()
        
        if not all(health.values()):
            # Some interfaces are down
            failed = [k for k, v in health.items() if not v]
            logger.warning(f"Degraded mode - failed interfaces: {failed}")
        
        # Adapt behavior based on environment
        if self.env.env_type == EnvironmentType.PRODUCTION:
            # Extra safety in production
            return self._execute_safe(task)
        else:
            # More permissive in dev
            return self._execute_normal(task)
    
    def _execute_safe(self, task: str) -> dict:
        """Production-safe execution"""
        # Add extra validation, logging, etc.
        pass
    
    def _execute_normal(self, task: str) -> dict:
        """Normal execution"""
        pass
```

---

## 🎯 Best Practices

### 1. **Always Use Abstraction Layers**

Never let agents directly access external systems:

```python
# ❌ BAD: Direct access
class BadAgent:
    def get_data(self):
        import psycopg2
        conn = psycopg2.connect("dbname=prod")  # Direct DB access!
        
# ✅ GOOD: Through interface
class GoodAgent:
    def __init__(self, db_interface: DatabaseInterface):
        self.db = db_interface
    
    def get_data(self):
        return self.db.execute_query("SELECT * FROM data")
```

### 2. **Implement Proper Timeouts**

```python
class TimeoutInterface:
    def request_with_timeout(self, func, timeout=30):
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout}s")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = func()
        finally:
            signal.alarm(0)  # Disable alarm
        
        return result
```

### 3. **Rate Limiting**

```python
from functools import wraps
import time

def rate_limit(calls: int, period: int):
    """Decorator for rate limiting"""
    min_interval = period / calls
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

class APIInterface:
    @rate_limit(calls=100, period=60)  # 100 calls per minute
    def make_request(self, endpoint: str):
        return requests.get(endpoint)
```

### 4. **Graceful Degradation**

```python
class RobustInterface:
    def get_data(self, source: str):
        try:
            # Try primary source
            return self.primary_source.get(source)
        except ConnectionError:
            try:
                # Fallback to cache
                return self.cache.get(source)
            except KeyError:
                # Last resort - return stale data with warning
                return {
                    "data": self.stale_data.get(source),
                    "warning": "Using stale data - primary source unavailable"
                }
```

---

## 🔌 Putting It All Together

```python
# Complete agent with multiple environment interfaces
class ProductionAgent:
    def __init__(self):
        # Initialize environment
        self.env = EnvironmentManager(EnvironmentType.PRODUCTION)
        
        # Set up interfaces
        self.github = GitHubInterface(
            token=self.env.get_interface('api_keys').get_api_key('github')
        )
        self.db = self.env.get_interface('database')
        self.fs = self.env.get_interface('filesystem')
        
        # Message queue for async tasks
        self.mq = RabbitMQInterface()
        
        # WebSocket for real-time updates
        self.ws = WebSocketInterface('wss://agent.example.com/live')
        self.ws.connect()
    
    def process_github_event(self, event: dict):
        """Process GitHub webhook event"""
        
        if event['type'] == 'issue':
            # Analyze issue
            analysis = self.analyze_issue(event['issue'])
            
            # Store in database
            self.db.execute_query(
                "INSERT INTO issues (id, analysis) VALUES (?, ?)",
                (event['issue']['id'], json.dumps(analysis))
            )
            
            # Publish to queue for further processing
            self.mq.publish('issue_queue', {
                'issue_id': event['issue']['id'],
                'analysis': analysis
            })
            
            # Real-time update
            self.ws.send({
                'type': 'issue_analyzed',
                'issue_id': event['issue']['id']
            })
    
    def run(self):
        """Main agent loop"""
        try:
            # Health check
            health = self.env.health_check()
            print(f"System health: {health}")
            
            # Start processing
            self.mq.subscribe('tasks', self.process_task)
            
        finally:
            # Cleanup
            self.ws.close()
            self.env.cleanup()
```

---

## 💡 Key Takeaways

1. **Interfaces are bridges** - They connect your agent to the real world
2. **Abstraction is essential** - Never let agents directly touch external systems
3. **Security first** - Always validate, sanitize, and sandbox
4. **Plan for failure** - External systems will fail; be ready
5. **Environment awareness** - Agents should adapt to dev/staging/prod
6. **Rate limits are real** - Respect them or get banned
7. **Async when possible** - Don't block on external calls

---

## 🚀 Hands-On Challenge

### Build a Data Pipeline Agent

Create an agent that:
1. Monitors a directory for new CSV files
2. Validates and cleans the data
3. Loads into a database
4. Sends notification via webhook
5. Archives processed files

Requirements:
- Use FileSystemInterface for file operations
- Use DatabaseInterface for data storage
- Implement proper error handling
- Add rate limiting for webhook calls
- Support different environments (dev/prod)

```python
class DataPipelineAgent:
    def __init__(self, env_manager: EnvironmentManager):
        self.env = env_manager
        # Your implementation here
    
    def process_pipeline(self):
        # 1. Monitor directory
        # 2. Process new files
        # 3. Load to database
        # 4. Send notifications
        # 5. Archive files
        pass

# Test it
agent = DataPipelineAgent(
    EnvironmentManager(EnvironmentType.DEVELOPMENT)
)
agent.process_pipeline()
```

---

## 📚 Further Learning

### Essential Topics
- **API Design Patterns** - RESTful, GraphQL, gRPC
- **Database Patterns** - Connection pooling, transactions
- **Message Queues** - RabbitMQ, Kafka, Redis
- **WebSockets** - Real-time communication
- **Security** - OAuth2, JWT, API keys

### Tools to Explore
- **Requests** - HTTP library
- **SQLAlchemy** - Database ORM
- **Celery** - Distributed task queue
- **FastAPI** - Modern web framework
- **Selenium** - Web automation

---

## ➡️ What's Next?

Almost there! One more lesson to complete your journey.

**Next Lesson:** [Evaluation & Metrics](10-evaluation.md)

Learn how to measure agent performance, track metrics, and know when your agent is actually working well!

---

## 🤔 Reflection Questions

1. **What interfaces does your agent need for your use case?**
2. **How would you handle authentication across multiple services?**
3. **What happens when an external API changes its format?**
4. **How do you test agents that depend on external systems?**
5. **When should an agent use sync vs async interfaces?**

---

## 📝 Your Notes

*Your thoughts on environment interfaces:*

**Interfaces I need to build:**
- 

**Security concerns:**
- 

**External systems to integrate:**
- 

**Questions:**
- 

---

**Lesson Status:** ✅ Complete  
**Estimated Time:** 90 minutes  
**Difficulty:** ⭐⭐⭐⭐ (Advanced)  
**Prerequisites:** Lessons 1-8  
**Next:** Evaluation & Metrics!
