# 🚀 QUICK START GUIDE - Research Assistant

Get your AI research agent running in 5 minutes!

## ✅ Step 1: Get Your Free Gemini API Key (2 minutes)

1. Visit: **https://makersuite.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy your key (looks like: `AIzaSyD...`)

**No credit card required!** Free tier includes:
- 60 requests per minute
- Plenty for research tasks

## ✅ Step 2: Install Dependencies (1 minute)

```bash
# Navigate to the project
cd research-assistant

# Install requirements
pip install -r requirements.txt
```

That's it! Only 3 packages needed (all free):
- `google-generativeai` - For Gemini
- `requests` - For API calls
- `python-dateutil` - For dates

## ✅ Step 3: Set Your API Key (30 seconds)

**Option A: Environment Variable (recommended)**
```bash
export GEMINI_API_KEY='your-api-key-here'
```

**Option B: Pass it directly**
```bash
python main.py "topic" --api-key "your-api-key-here"
```

**Option C: Edit config.py**
```python
# In config.py, replace:
GEMINI_API_KEY = "your-api-key-here"
```

## ✅ Step 4: Run Your First Research! (30 seconds)

```bash
# Simple research
python main.py "artificial intelligence"

# With options
python main.py "quantum computing" --depth deep --max-iterations 8
```

## 🎉 That's It!

You should see:
```
==============================================================
🤖 RESEARCH ASSISTANT AGENT
==============================================================
Topic: artificial intelligence
Depth: medium
Max iterations: 5
==============================================================

🔧 Initializing components...
  ✅ LLM initialized (Gemini)
  ✅ Memory systems ready
  ✅ Tools ready: wikipedia, news, web_search, file_writer
  ✅ Agent initialized

============================================================
🔍 RESEARCH TOPIC: artificial intelligence
📊 DEPTH: medium
============================================================

--- ITERATION 1 ---
[PERCEIVE] Gathering context...
[PLAN] Creating research plan...
[ACT] Executing plan...
...
```

## 📖 Example Commands

### Quick Research (1-2 minutes)
```bash
python main.py "Python programming" --depth quick
```

### Balanced Research (2-4 minutes) - **Recommended**
```bash
python main.py "machine learning trends 2024" --depth medium
```

### Deep Research (5-8 minutes)
```bash
python main.py "climate change solutions" --depth deep --max-iterations 10
```

### Current News Topics
```bash
python main.py "AI news 2024" --depth medium
```

## 📁 Where Are My Reports?

After research completes, find your reports in:
```
research-assistant/
└── data/
    └── reports/
        └── research_your_topic_TIMESTAMP.md
```

Example output:
```
📄 Report saved to: data/reports/research_artificial_intelligence_20241023_120530.md
```

## 🐛 Troubleshooting

### Problem: "API key required"
```
❌ Error: Gemini API key required!
```

**Fix:**
```bash
# Check if key is set
echo $GEMINI_API_KEY

# If empty, set it:
export GEMINI_API_KEY='your-key-here'

# Or pass directly:
python main.py "topic" --api-key "your-key-here"
```

### Problem: "Module not found"
```
ModuleNotFoundError: No module named 'google.generativeai'
```

**Fix:**
```bash
pip install -r requirements.txt
```

### Problem: Rate limit
```
⚠️ RateLimitError
```

**Fix:** Wait 1 minute (free tier: 60 req/min). Agent auto-retries!

### Problem: No results
```
⚠️ Step failed: No results found
```

**Fix:**
- Try different search terms
- Use `--depth deep` for more sources
- Check your internet connection

## 🎯 Tips for Better Results

1. **Be specific with topics**
   ```bash
   # ❌ Too broad
   python main.py "technology"
   
   # ✅ More specific
   python main.py "AI in healthcare 2024"
   ```

2. **Use appropriate depth**
   - `quick`: Overview only (2-3 sources)
   - `medium`: Balanced (4-6 sources) ← **Best for most cases**
   - `deep`: Comprehensive (8+ sources)

3. **Check the logs**
   ```bash
   # View latest log
   tail -f data/logs/research_*.log
   ```

4. **Browse your memory**
   ```bash
   # See what the agent learned
   cat data/memory/knowledge_base.json
   ```

## 🎓 What Just Happened?

Your agent:
1. ✅ Searched Wikipedia
2. ✅ Searched the web (DuckDuckGo)
3. ✅ Optionally searched news
4. ✅ Synthesized information using Gemini
5. ✅ Created a structured report
6. ✅ Stored knowledge for future use

All **autonomously**! No manual steps required.

## 🚀 Next Steps

### Test Different Topics
```bash
python main.py "renewable energy"
python main.py "space exploration"
python main.py "cryptocurrency trends"
```

### Run the Demo
```bash
python demo.py
```
Shows all components working (no API key needed!)

### Customize
Edit `config.py` to change:
- Number of sources
- Memory size
- Timeout settings
- Output format

### Build More!
Check out:
- **Multi-Agent Debate System** (coming next!)
- Add your own tools
- Integrate more APIs
- Deploy as web service

## 📚 Learn More

- **README.md** - Full documentation
- **Lessons 1-7** - Understanding the concepts
- **agent.py** - See the agent loop in action
- **tools.py** - Add your own tools

## 💬 Need Help?

1. Run the demo: `python demo.py`
2. Check logs: `data/logs/`
3. Review memory: `data/memory/`
4. Read the code - it's well commented!

---

## 🎉 You're Ready!

You now have an autonomous AI research assistant powered by **100% free APIs**!

**Happy researching! 🔍**