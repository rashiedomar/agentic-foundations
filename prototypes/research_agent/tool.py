"""
Tools for Research Assistant
Implements: Wikipedia search, News API, File operations
With comprehensive error handling (Lesson 7)
"""

import json
import requests
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ToolResult:
    """Standardized tool result - Lesson 6"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp
        }


class BaseTool(ABC):
    """Base class for all tools - Lesson 6"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.execution_count = 0
        self.failure_count = 0
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        pass
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute with error handling - Lesson 7"""
        self.execution_count += 1
        
        try:
            logger.info(f"Executing {self.name} with params: {kwargs}")
            result = self._execute(**kwargs)
            logger.info(f"{self.name} completed successfully")
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"{self.name} failed: {e}")
            
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}"
            )
    
    def get_stats(self) -> Dict:
        """Get tool execution statistics"""
        return {
            "name": self.name,
            "executions": self.execution_count,
            "failures": self.failure_count,
            "success_rate": (
                (self.execution_count - self.failure_count) / self.execution_count
                if self.execution_count > 0 else 1.0
            )
        }


class WikipediaSearchTool(BaseTool):
    """Search Wikipedia - completely free!"""
    
    @property
    def description(self) -> str:
        return """
        Search Wikipedia for information on any topic.
        Returns: Article summaries and links.
        
        Parameters:
        - query (str): The search query
        - limit (int): Max number of results (default: 3)
        """
    
    def _execute(self, query: str, limit: int = 3) -> Dict:
        """Search Wikipedia API"""
        url = "https://en.wikipedia.org/w/api.php"
        
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("query", {}).get("search", []):
            # Get article content
            content = self._get_article_summary(item["title"])
            
            results.append({
                "title": item["title"],
                "snippet": item["snippet"],
                "summary": content,
                "url": f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}"
            })
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
    
    def _get_article_summary(self, title: str) -> str:
        """Get article summary"""
        url = "https://en.wikipedia.org/w/api.php"
        
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                return page.get("extract", "")[:500]  # First 500 chars
                
        except Exception as e:
            logger.warning(f"Failed to get summary for {title}: {e}")
            return ""


class NewsSearchTool(BaseTool):
    """Search news - 100 requests/day free with NewsAPI"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key
    
    @property
    def description(self) -> str:
        return """
        Search recent news articles.
        Returns: Recent news articles on the topic.
        
        Parameters:
        - query (str): The search query
        - limit (int): Max number of articles (default: 5)
        """
    
    def _execute(self, query: str, limit: int = 5) -> Dict:
        """Search news API"""
        if not self.api_key:
            # Fallback to RSS/free sources
            return self._fallback_search(query, limit)
        
        url = "https://newsapi.org/v2/everything"
        
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": limit,
            "apiKey": self.api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        articles = []
        
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title"),
                "description": article.get("description"),
                "source": article.get("source", {}).get("name"),
                "url": article.get("url"),
                "published": article.get("publishedAt")
            })
        
        return {
            "query": query,
            "articles_count": len(articles),
            "articles": articles
        }
    
    def _fallback_search(self, query: str, limit: int) -> Dict:
        """Fallback when no API key - use Wikipedia news"""
        logger.info("NewsAPI key not found, using Wikipedia fallback")
        
        wiki_tool = WikipediaSearchTool()
        wiki_result = wiki_tool.execute(query=f"{query} news", limit=limit)
        
        if wiki_result.success:
            return {
                "query": query,
                "articles_count": len(wiki_result.data["results"]),
                "articles": [
                    {
                        "title": r["title"],
                        "description": r["snippet"],
                        "source": "Wikipedia",
                        "url": r["url"]
                    }
                    for r in wiki_result.data["results"]
                ],
                "fallback_used": True
            }
        
        raise Exception("Both NewsAPI and Wikipedia fallback failed")


class FileWriterTool(BaseTool):
    """Write research reports to files"""
    
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    @property
    def description(self) -> str:
        return """
        Write content to a file.
        
        Parameters:
        - filename (str): Name of the file
        - content (str): Content to write
        - format (str): File format (txt, md, json)
        """
    
    def _execute(self, filename: str, content: str, format: str = "md") -> Dict:
        """Write to file"""
        # Add extension if not present
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "filename": filename,
            "filepath": str(filepath),
            "size": len(content),
            "format": format
        }


class WebSearchTool(BaseTool):
    """Simple web search using DuckDuckGo (free, no API key)"""
    
    @property
    def description(self) -> str:
        return """
        Search the web for information.
        
        Parameters:
        - query (str): The search query
        - limit (int): Max results (default: 5)
        """
    
    def _execute(self, query: str, limit: int = 5) -> Dict:
        """Search using DuckDuckGo instant answers API"""
        url = "https://api.duckduckgo.com/"
        
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        
        # Get abstract
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", query),
                "snippet": data.get("Abstract"),
                "url": data.get("AbstractURL", "")
            })
        
        # Get related topics
        for topic in data.get("RelatedTopics", [])[:limit]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", "")
                })
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results[:limit]
        }


class ToolRegistry:
    """Manage and access all tools - Lesson 6"""
    
    def __init__(self, output_dir, news_api_key: Optional[str] = None):
        self.tools = {
            "wikipedia": WikipediaSearchTool(),
            "news": NewsSearchTool(api_key=news_api_key),
            "web_search": WebSearchTool(),
            "file_writer": FileWriterTool(output_dir)
        }
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List available tools"""
        return list(self.tools.keys())
    
    def get_descriptions(self) -> str:
        """Get all tool descriptions for LLM"""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"Tool: {name}\n{tool.description}")
        return "\n\n".join(descriptions)
    
    def get_stats(self) -> Dict:
        """Get statistics for all tools"""
        return {
            name: tool.get_stats()
            for name, tool in self.tools.items()
        }