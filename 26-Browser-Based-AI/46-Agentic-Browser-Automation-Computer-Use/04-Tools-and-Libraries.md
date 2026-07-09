# Tools and Libraries for Browser Automation Agents

> **Description:** Comprehensive guide to the tooling ecosystem for browser automation agents. Covers browser automation frameworks, AI-powered tools, headless browsers, proxy services, CAPTCHA solving, fingerprint management, and the complete developer toolkit for building production browser agents.

---

## Table of Contents

1. [Tooling Landscape Overview](#tooling-landscape-overview)
2. [Browser Automation Frameworks](#browser-automation-frameworks)
3. [AI-Powered Browser Agents](#ai-powered-browser-agents)
4. [Headless Browser Solutions](#headless-browser-solutions)
5. [Browser Pool and Infrastructure](#browser-pool-and-infrastructure)
6. [Proxy and Network Tools](#proxy-and-network-tools)
7. [Fingerprint Management](#fingerprint-management)
8. [CAPTCHA Solving](#captcha-solving)
9. [Data Extraction Tools](#data-extraction-tools)
10. [Testing and Debugging](#testing-and-debugging)
11. [Monitoring and Observability](#monitoring-and-observability)
12. [Security Tools](#security-tools)
13. [Comparison Matrix](#comparison-matrix)
14. [Integration Patterns](#integration-patterns)
15. [Best Practices](#best-practices)
16. [Cross-References](#cross-references)

---

## Tooling Landscape Overview

### Ecosystem Map

```
┌─────────────────────────────────────────────────────────────┐
│                    AI/LLM Layer                              │
│  GPT-4o │ Claude │ Gemini │ Qwen-VL │ LLaVA │ Pixtral      │
├─────────────────────────────────────────────────────────────┤
│                  Agent Framework Layer                       │
│  Browser Use │ Stagehand │ AgentQL │ AutoGPT │ Mind2Web     │
├─────────────────────────────────────────────────────────────┤
│                Browser Automation Layer                      │
│  Playwright │ Puppeteer │ Selenium │ WebDriver │ CDP        │
├─────────────────────────────────────────────────────────────┤
│                  Browser Runtime Layer                       │
│  Chrome │ Firefox │ WebKit │ Chromium │ Edge               │
├─────────────────────────────────────────────────────────────┤
│                 Infrastructure Layer                         │
│  Docker │ Kubernetes │ Browserbase │ BrightData │ ScraperAPI│
├─────────────────────────────────────────────────────────────┤
│                  Utility Layer                               │
│  CAPTCHA Solvers │ Proxy Managers │ Fingerprint Managers    │
│  Data Extractors │ Screenshot Optimizers │ Rate Limiters    │
└─────────────────────────────────────────────────────────────┘
```

### Tool Selection Criteria

| Criterion | Weight | Description |
|---|---|---|
| Reliability | 30% | How consistent is the tool in production? |
| Performance | 25% | Speed and resource efficiency |
| Cost | 20% | Licensing, API costs, infrastructure |
| Community | 15% | Ecosystem, documentation, support |
| Flexibility | 10% | Customization and extensibility |

---

## Browser Automation Frameworks

### Playwright

**The modern standard for browser automation**

```python
from playwright.async_api import async_playwright

async def playwright_example():
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 ...',
            locale='en-US'
        )
        page = await context.new_page()
        
        # Navigate
        await page.goto('https://example.com')
        
        # Interact
        await page.click('button#submit')
        await page.fill('input[name="email"]', 'user@example.com')
        
        # Wait for response
        await page.wait_for_selector('.result')
        
        # Extract data
        data = await page.evaluate('() => document.body.innerText')
        
        await browser.close()
```

**Key features:**
- Multi-browser support (Chromium, Firefox, WebKit)
- Auto-waiting and retry mechanisms
- Network interception
- Mobile device emulation
- Video recording and tracing

**Python API:**
```python
# Install: pip install playwright
# Setup: playwright install

# Key classes:
# Browser - represents browser instance
# BrowserContext - isolated browser session
# Page - single tab
# Frame - iframe within page
# ElementHandle - DOM element reference
```

### Puppeteer

**Node.js-first browser automation**

```javascript
const puppeteer = require('puppeteer');

async function puppeteerExample() {
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 720 });
    
    await page.goto('https://example.com');
    
    // CDP session for advanced features
    const client = await page.target().createCDPSession();
    
    // Screenshot
    await page.screenshot({ path: 'screenshot.png', fullPage: true });
    
    // PDF generation
    await page.pdf({ path: 'page.pdf', format: 'A4' });
    
    await browser.close();
}
```

**Key features:**
- Chrome DevTools Protocol (CDP) native access
- PDF generation
- Performance metrics
- Network conditions emulation

### Selenium

**Legacy but still widely used**

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def selenium_example():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(options=options)
    
    driver.get('https://example.com')
    
    # Explicit wait
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "submit"))
    )
    element.click()
    
    # Extract
    data = driver.find_element(By.CLASS_NAME, 'result').text
    
    driver.quit()
```

**When to use:**
- Legacy codebase integration
- Specific browser compatibility needs
- WebDriver BiDi protocol requirements

### Comparison: Playwright vs Puppeteer vs Selenium

| Feature | Playwright | Puppeteer | Selenium |
|---|---|---|---|
| **Language Support** | Python, JS, Java, C# | JavaScript | Java, Python, C#, Ruby, JS |
| **Browser Support** | Chromium, Firefox, WebKit | Chromium only | All major browsers |
| **Speed** | Fast | Fast | Medium |
| **Auto-waiting** | Yes | Partial | No |
| **Network Interception** | Yes | Yes | Limited |
| **Mobile Emulation** | Yes | Yes | Limited |
| **CDP Access** | Yes | Native | Limited |
| **Parallel Execution** | Yes | Yes | Yes |
| **Community Size** | Growing rapidly | Large | Very large |
| **Recommended For** | New projects | Node.js projects | Legacy integration |

---

## AI-Powered Browser Agents

### Browser Use

**Most popular AI browser agent framework**

```python
from browser_use import Agent
from langchain_openai import ChatOpenAI

# Simple usage
agent = Agent(
    task="Find flights from NYC to London for next week",
    llm=ChatOpenAI(model="gpt-4o"),
    max_actions_per_step=3,
)

result = await agent.run()
print(result.final_result())
```

**Advanced configuration:**
```python
from browser_use import Agent, Browser, BrowserConfig

browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=True,  # For testing
        extra_chromium_args=[
            '--window-size=1920,1080',
            '--disable-gpu'
        ]
    )
)

agent = Agent(
    task="Complex multi-step task",
    llm=ChatOpenAI(model="gpt-4o"),
    browser=browser,
    max_actions_per_step=5,
    planner_llm=ChatOpenAI(model="gpt-4o-mini"),  # Separate planner
    use_vision=True,
    save_interactions_path="./interactions",
)

# Custom actions
from browser_use.controller.service import Controller

controller = Controller()

@controller.action("Save product to wishlist")
def save_to_wishlist(product_name: str):
    # Custom action implementation
    return f"Saved {product_name} to wishlist"

agent = Agent(
    task="Find and save products",
    llm=ChatOpenAI(model="gpt-4o"),
    controller=controller,
)
```

**Key features:**
- Vision support (screenshots)
- Custom action registration
- Multi-tab support
- State persistence
- Cost tracking
- Replay and debugging

### Stagehand

**Anthropic-backed browser agent**

```python
from stagehand import Stagehand

async with Stagehand() as stagehand:
    page = await stagehand.page.goto("https://example.com")
    
    # Natural language actions
    await stagehand.act("click the login button")
    await stagehand.act("type 'user@example.com' in the email field")
    await stagehand.act("click submit")
    
    # Extract structured data
    data = await stagehand.extract(
        "Extract the product name, price, and description"
    )
    
    # Query page state
    is_logged_in = await stagehand.observe(
        "Is the user logged in?"
    )
```

**Key features:**
- CDP-native browser control
- DOM-aware + vision hybrid
- Built-in caching
- Structured data extraction

### AgentQL

**Semantic web interaction**

```python
import agentql

# Use natural language to interact with web elements
page = agentql.goto("https://example.com")

# Find elements semantically
search_input = page.query("the search input field")
search_input.type("query")

submit_button = page.query("the search submit button")
submit_button.click()

# Extract structured data
results = page.query_all("search result items")
for result in results:
    print(result.name, result.price, result.url)
```

### AutoGPT / AgentGPT

**General-purpose agents with browser capabilities**

```python
# AutoGPT with browser tool
from autogpt import Agent

agent = Agent(
    name="WebResearcher",
    goals=["Research and summarize AI trends from multiple sources"],
    tools=[
        "BrowseWebsite",
        "GoogleSearch",
        "ReadArticle",
        "SummarizeText"
    ]
)

agent.run()
```

---

## Headless Browser Solutions

### Docker-Based Solutions

```dockerfile
# Browserless - Chrome as a service
FROM browserless/chrome

# Configure
ENV CONNECTION_TIMEOUT=60000
ENV MAX_CONCURRENT_SESSIONS=10
ENV MAXQUEUED=100
```

```yaml
# docker-compose.yml
services:
  browserless:
    image: browserless/chrome
    ports:
      - "3000:3000"
    environment:
      - TOKEN=your-token
      - MAX_CONCURRENT_SESSIONS=10
    
  playwright-service:
    image: mcr.microsoft.com/playwright/python:v1.45.0
    volumes:
      - ./scripts:/scripts
    command: python /scripts/agent.py
```

### Browser Pool Services

| Service | Type | Features | Pricing |
|---|---|---|---|
| **Browserbase** | Cloud | Serverless, session replay | $0.01/min |
| **Bright Data** | Proxy+Browser | Residential proxies, browsers | Pay per GB |
| **ScraperAPI** | Proxy | Rotating proxies, rendering | $49/mo+ |
| **ZenRows** | Proxy+Browser | Anti-detection, JS rendering | $49/mo+ |
| **ScrapingBee** | Proxy+Browser | Screenshots, PDF generation | $49/mo+ |
| **Browserless** | Self-hosted | Docker-based, scalable | Free (self-hosted) |

### Cloud Browser Integration

```python
from browserbase import Browserbase

# Browserbase - serverless browser
bb = Browserbase(api_key="your-key")

async def cloud_browse(task: str):
    # Create browser context
    context = await bb.contexts.create(
        project_id="your-project",
        proxy={"type": "RESIDENTIAL"}
    )
    
    # Get page
    page = await context.get_page()
    
    # Use like regular Playwright
    await page.goto("https://example.com")
    # ... your automation logic
    
    # Session is automatically managed
```

---

## Browser Pool and Infrastructure

### Connection Pool Management

```python
class BrowserConnectionPool:
    """Manage browser connections efficiently"""
    
    def __init__(self, max_connections: int = 5, browser_type: str = "chromium"):
        self.max = max_connections
        self.browser_type = browser_type
        self.pool = asyncio.Queue()
        self.active = 0
        self.playwright = None
        self.browser = None
    
    async def initialize(self):
        self.playwright = await async_playwright().start()
        self.browser = await getattr(self.playwright, self.browser_type).launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--single-process'
            ]
        )
        
        # Pre-create connections
        for _ in range(self.max):
            context = await self.browser.new_context()
            await self.pool.put(context)
    
    async def acquire(self):
        context = await self.pool.get()
        self.active += 1
        return context
    
    async def release(self, context):
        await self.pool.put(context)
        self.active -= 1
    
    async def cleanup(self):
        await self.browser.close()
        await self.playwright.stop()
```

### Resource Management

```python
class ResourceManager:
    """Manage browser resources and limits"""
    
    def __init__(self):
        self.limits = {
            'max_pages_per_context': 5,
            'max_contexts': 10,
            'max_memory_mb': 4096,
            'max_cpu_percent': 80,
            'page_timeout_ms': 30000,
            'navigation_timeout_ms': 60000,
        }
        self.usage = {}
    
    async def check_resources(self):
        """Check if resources are within limits"""
        import psutil
        
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        return {
            'memory_ok': memory.percent < self.limits['max_cpu_percent'],
            'cpu_ok': cpu < self.limits['max_cpu_percent'],
            'memory_percent': memory.percent,
            'cpu_percent': cpu
        }
    
    def get_page_config(self):
        """Get page configuration with resource limits"""
        return {
            'viewport': {'width': 1280, 'height': 720},
            'timeouts': {
                'navigation': self.limits['navigation_timeout_ms'],
                'page': self.limits['page_timeout_ms']
            }
        }
```

---

## Proxy and Network Tools

### Proxy Management

```python
class ProxyManager:
    """Manage proxy rotation and configuration"""
    
    def __init__(self, proxies: list):
        self.proxies = proxies
        self.current = 0
        self.failed = set()
    
    def get_proxy(self) -> dict:
        """Get next available proxy"""
        while self.current < len(self.proxies):
            proxy = self.proxies[self.current]
            self.current = (self.current + 1) % len(self.proxies)
            
            if proxy['id'] not in self.failed:
                return proxy
        
        raise Exception("No available proxies")
    
    def mark_failed(self, proxy_id: str):
        """Mark proxy as failed"""
        self.failed.add(proxy_id)
    
    def get_browser_args(self, proxy: dict) -> list:
        """Get browser args for proxy"""
        return [
            f'--proxy-server={proxy["server"]}',
            f'--proxy-bypass-list=<-loopback>'
        ]
    
    def get_context_args(self, proxy: dict) -> dict:
        """Get Playwright context args for proxy"""
        return {
            'proxy': {
                'server': proxy['server'],
                'username': proxy.get('username'),
                'password': proxy.get('password')
            }
        }

# Usage
proxies = [
    {"id": "1", "server": "http://proxy1:8080", "username": "user", "password": "pass"},
    {"id": "2", "server": "http://proxy2:8080", "username": "user", "password": "pass"},
]

manager = ProxyManager(proxies)
proxy = manager.get_proxy()
```

### Network Condition Emulation

```python
class NetworkEmulator:
    """Emulate various network conditions"""
    
    PROFILES = {
        'wifi': {'download': 30 * 1024 * 1024, 'upload': 15 * 1024 * 1024, 'latency': 2},
        '4g': {'download': 4 * 1024 * 1024, 'upload': 3 * 1024 * 1024, 'latency': 20},
        '3g': {'download': 750 * 1024, 'upload': 250 * 1024, 'latency': 100},
        'slow': {'download': 100 * 1024, 'upload': 50 * 1024, 'latency': 500},
    }
    
    async def emulate(self, page, profile: str = 'wifi'):
        conditions = self.PROFILES[profile]
        await page.route("**/*", lambda route: route.continue_())
        
        # Use CDP for network throttling
        client = await page.context.new_cdp_session(page)
        await client.send('Network.emulateNetworkConditions', {
            'offline': False,
            'downloadThroughput': conditions['download'],
            'uploadThroughput': conditions['upload'],
            'latency': conditions['latency']
        })
```

---

## Fingerprint Management

### Browser Fingerprint Randomization

```python
import random

class FingerprintManager:
    """Manage browser fingerprints for anti-detection"""
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    ]
    
    VIEWPORTS = [
        {'width': 1920, 'height': 1080},
        {'width': 1366, 'height': 768},
        {'width': 1440, 'height': 900},
        {'width': 1536, 'height': 864},
    ]
    
    LOCALES = ['en-US', 'en-GB', 'en-CA', 'en-AU']
    
    def generate_fingerprint(self) -> dict:
        return {
            'user_agent': random.choice(self.USER_AGENTS),
            'viewport': random.choice(self.VIEWPORTS),
            'locale': random.choice(self.LOCALES),
            'timezone': random.choice(['America/New_York', 'America/Chicago', 'America/Los_Angeles']),
            'color_scheme': random.choice(['light', 'dark']),
            'has_touch': random.choice([True, False]),
        }
    
    async def apply_fingerprint(self, context, fingerprint: dict):
        """Apply fingerprint to browser context"""
        await context.set_extra_http_headers({
            'Accept-Language': f"{fingerprint['locale']},en;q=0.9"
        })
        
        # Viewport is set during context creation
        # Additional fingerprinting via JavaScript injection
        await context.add_init_script(f"""
            Object.defineProperty(navigator, 'languages', {{
                get: () => ['{fingerprint['locale']}', 'en']
            }});
            
            Object.defineProperty(navigator, 'plugins', {{
                get: () => [1, 2, 3, 4, 5]
            }});
            
            Object.defineProperty(navigator, 'hardwareConcurrency', {{
                get: () => {random.choice([4, 8, 16])}
            }});
        """)
```

---

## CAPTCHA Solving

### CAPTCHA Detection

```python
class CAPTCHADetector:
    """Detect CAPTCHAs on pages"""
    
    INDICATORS = [
        # reCAPTCHA
        'recaptcha',
        'g-recaptcha',
        'iframe[src*="recaptcha"]',
        
        # hCaptcha
        'hcaptcha',
        'h-captcha',
        'iframe[src*="hcaptcha"]',
        
        # Cloudflare
        'cf-challenge',
        'challenge-platform',
        
        # Turnstile
        'cf-turnstile',
        'turnstile',
    ]
    
    async def detect(self, page) -> dict:
        """Detect if page has CAPTCHA"""
        for indicator in self.INDICATORS:
            element = await page.query_selector(indicator)
            if element:
                return {
                    'detected': True,
                    'type': self.classify_type(indicator),
                    'selector': indicator
                }
        
        return {'detected': False}
    
    def classify_type(self, indicator: str) -> str:
        if 'recaptcha' in indicator:
            return 'recaptcha'
        elif 'hcaptcha' in indicator:
            return 'hcaptcha'
        elif 'cf-' in indicator:
            return 'cloudflare'
        elif 'turnstile' in indicator:
            return 'turnstile'
        return 'unknown'
```

### CAPTCHA Solving Services

| Service | Types Supported | Price | Accuracy |
|---|---|---|---|
| **2Captcha** | reCAPTCHA, hCaptcha, Turnstile | $2.99/1000 | 95%+ |
| **Anti-Captcha** | reCAPTCHA, hCaptcha | $1/1000 | 95%+ |
| **CapSolver** | All major types | $1/1000 | 98%+ |
| **FunCaptcha** | FunCaptcha specific | $2/1000 | 90%+ |

```python
class CAPTCHASolver:
    """Solve CAPTCHAs using third-party services"""
    
    def __init__(self, service: str, api_key: str):
        self.service = service
        self.api_key = api_key
    
    async def solve_recaptcha(self, site_key: str, page_url: str) -> str:
        """Solve reCAPTCHA and return token"""
        
        if self.service == 'capsolver':
            return await self.solve_capsolver_recaptcha(site_key, page_url)
        elif self.service == '2captcha':
            return await self.solve_2captcha_recaptcha(site_key, page_url)
    
    async def solve_capsolver_recaptcha(self, site_key: str, page_url: str) -> str:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Create task
            async with session.post('https://api.capsolver.com/createTask', json={
                'clientKey': self.api_key,
                'task': {
                    'type': 'ReCaptchaV2TaskProxyLess',
                    'websiteURL': page_url,
                    'websiteKey': site_key,
                }
            }) as resp:
                data = await resp.json()
                task_id = data['taskId']
            
            # Poll for result
            while True:
                async with session.post('https://api.capsolver.com/getTaskResult', json={
                    'clientKey': self.api_key,
                    'taskId': task_id
                }) as resp:
                    data = await resp.json()
                    if data['status'] == 'ready':
                        return data['solution']['gRecaptchaResponse']
                    await asyncio.sleep(2)
```

---

## Data Extraction Tools

### Structured Data Extraction

```python
class DataExtractor:
    """Extract structured data from web pages"""
    
    async def extract_jsonld(self, page) -> dict:
        """Extract JSON-LD structured data"""
        return await page.evaluate("""
            () => {
                const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                return Array.from(scripts).map(s => JSON.parse(s.textContent));
            }
        """)
    
    async def extract_microdata(self, page) -> list:
        """Extract microdata"""
        return await page.evaluate("""
            () => {
                const items = document.querySelectorAll('[itemscope]');
                return Array.from(items).map(item => ({
                    type: item.getAttribute('itemtype'),
                    properties: Array.from(item.querySelectorAll('[itemprop]')).map(prop => ({
                        name: prop.getAttribute('itemprop'),
                        content: prop.getAttribute('content') || prop.textContent
                    }))
                }));
            }
        """)
    
    async def extract_opengraph(self, page) -> dict:
        """Extract OpenGraph metadata"""
        return await page.evaluate("""
            () => {
                const meta = document.querySelectorAll('meta[property^="og:"]');
                const data = {};
                meta.forEach(m => {
                    const key = m.getAttribute('property').replace('og:', '');
                    data[key] = m.getAttribute('content');
                });
                return data;
            }
        """)
    
    async def extract_table(self, page, table_selector: str) -> list:
        """Extract HTML table data"""
        return await page.evaluate(f"""
            () => {{
                const table = document.querySelector('{table_selector}');
                if (!table) return [];
                
                const headers = Array.from(table.querySelectorAll('th'))
                    .map(th => th.textContent.trim());
                
                const rows = Array.from(table.querySelectorAll('tbody tr'))
                    .map(row => {{
                        const cells = Array.from(row.querySelectorAll('td'))
                            .map(td => td.textContent.trim());
                        return headers.reduce((obj, header, i) => {{
                            obj[header] = cells[i];
                            return obj;
                        }}, {{}});
                    }});
                
                return rows;
            }}
        """)
```

### Image and Media Extraction

```python
class MediaExtractor:
    """Extract images and media from pages"""
    
    async def extract_images(self, page, min_size: tuple = (100, 100)) -> list:
        """Extract images meeting size criteria"""
        return await page.evaluate(f"""
            () => {{
                const images = Array.from(document.querySelectorAll('img'));
                return images
                    .filter(img => img.naturalWidth >= {min_size[0]} && 
                                   img.naturalHeight >= {min_size[1]})
                    .map(img => ({{
                        src: img.src,
                        alt: img.alt,
                        width: img.naturalWidth,
                        height: img.naturalHeight,
                        format: img.src.split('.').pop().split('?')[0]
                    }}));
            }}
        """)
    
    async def extract_videos(self, page) -> list:
        """Extract video sources"""
        return await page.evaluate("""
            () => {
                const videos = document.querySelectorAll('video');
                return Array.from(videos).map(v => ({
                    src: v.src,
                    sources: Array.from(v.querySelectorAll('source')).map(s => ({
                        src: s.src,
                        type: s.type
                    })),
                    poster: v.poster,
                    duration: v.duration
                }));
            }
        """)
```

---

## Testing and Debugging

### Screenshot Comparison

```python
class ScreenshotComparator:
    """Compare screenshots for visual regression"""
    
    async def compare(self, before_path: str, after_path: str, threshold: float = 0.95):
        from PIL import Image
        import imagehash
        
        before = Image.open(before_path)
        after = Image.open(after_path)
        
        # Perceptual hash comparison
        hash_before = imagehash.phash(before)
        hash_after = imagehash.phash(after)
        
        similarity = 1 - (hash_before - hash_after) / 64
        
        return {
            'similar': similarity >= threshold,
            'similarity': similarity,
            'threshold': threshold
        }
```

### Recording and Replay

```python
class SessionRecorder:
    """Record browser sessions for debugging"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.events = []
    
    async def setup_recording(self, page):
        """Setup recording hooks on page"""
        page.on('request', self.record_request)
        page.on('response', self.record_response)
        page.on('console', self.record_console)
    
    def record_request(self, request):
        self.events.append({
            'type': 'request',
            'timestamp': time.time(),
            'url': request.url,
            'method': request.method
        })
    
    def record_response(self, response):
        self.events.append({
            'type': 'response',
            'timestamp': time.time(),
            'url': response.url,
            'status': response.status
        })
    
    def record_console(self, msg):
        self.events.append({
            'type': 'console',
            'timestamp': time.time(),
            'level': msg.type,
            'text': msg.text
        })
    
    async def save_recording(self, session_id: str):
        path = os.path.join(self.output_dir, f"{session_id}.json")
        with open(path, 'w') as f:
            json.dump(self.events, f, indent=2)
        return path
```

---

## Monitoring and Observability

### Agent Metrics

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class AgentMetrics:
    """Track agent performance metrics"""
    
    task_start: float = 0
    task_end: float = 0
    steps: int = 0
    screenshots_taken: int = 0
    api_calls: int = 0
    tokens_used: int = 0
    errors: int = 0
    retries: int = 0
    
    def start_task(self):
        self.task_start = time.time()
    
    def end_task(self):
        self.task_end = time.time()
    
    @property
    def duration(self) -> float:
        return self.task_end - self.task_start
    
    @property
    def steps_per_second(self) -> float:
        return self.steps / max(self.duration, 0.001)
    
    @property
    def success_rate(self) -> float:
        return 1 - (self.errors / max(self.steps, 1))
    
    def to_dict(self) -> dict:
        return {
            'duration': self.duration,
            'steps': self.steps,
            'screenshots': self.screenshots_taken,
            'api_calls': self.api_calls,
            'tokens': self.tokens_used,
            'errors': self.errors,
            'retries': self.retries,
            'steps_per_second': self.steps_per_second,
            'success_rate': self.success_rate,
        }

class MetricsCollector:
    """Collect and report agent metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def start_task(self, task_id: str):
        self.metrics[task_id] = AgentMetrics()
        self.metrics[task_id].start_task()
    
    def record_step(self, task_id: str, tokens: int = 0):
        if task_id in self.metrics:
            m = self.metrics[task_id]
            m.steps += 1
            m.screenshots_taken += 1
            m.api_calls += 1
            m.tokens_used += tokens
    
    def record_error(self, task_id: str):
        if task_id in self.metrics:
            self.metrics[task_id].errors += 1
    
    def end_task(self, task_id: str) -> dict:
        if task_id in self.metrics:
            self.metrics[task_id].end_task()
            result = self.metrics[task_id].to_dict()
            self.history.append({'task_id': task_id, **result})
            return result
        return {}
```

### Logging and Tracing

```python
import logging
from opentelemetry import trace

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('browser_agent')

# OpenTelemetry tracing
tracer = trace.get_tracer("browser_agent")

class AgentTracer:
    """Trace agent actions for debugging"""
    
    def trace_action(self, action_type: str, params: dict):
        with tracer.start_as_current_span(f"action.{action_type}") as span:
            span.set_attribute("action.type", action_type)
            for k, v in params.items():
                span.set_attribute(f"action.{k}", str(v))
            
            logger.info(f"Action: {action_type} | Params: {params}")
            yield span
    
    def trace_screenshot(self, screenshot_data: bytes):
        with tracer.start_as_current_span("screenshot") as span:
            span.set_attribute("screenshot.size", len(screenshot_data))
            span.set_attribute("screenshot.timestamp", time.time())
    
    def trace_llm_call(self, prompt: str, response: str, tokens: int):
        with tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("llm.prompt_length", len(prompt))
            span.set_attribute("llm.response_length", len(response))
            span.set_attribute("llm.tokens", tokens)
```

---

## Security Tools

### Credential Management

```python
class CredentialManager:
    """Securely manage credentials for browser agents"""
    
    def __init__(self):
        self.vault = {}
    
    async def get_credential(self, service: str, key: str) -> str:
        """Get credential from secure storage"""
        # In production, use HashiCorp Vault, AWS Secrets Manager, etc.
        return self.vault.get(f"{service}:{key}")
    
    async def store_credential(self, service: str, key: str, value: str):
        """Store credential securely"""
        self.vault[f"{service}:{key}"] = value
    
    async def fill_login_form(self, page, service: str, selectors: dict):
        """Fill login form with stored credentials"""
        username = await self.get_credential(service, 'username')
        password = await self.get_credential(service, 'password')
        
        await page.fill(selectors['username'], username)
        await page.fill(selectors['password'], password)
        await page.click(selectors['submit'])
```

### Action Validation

```python
class ActionValidator:
    """Validate and sanitize agent actions"""
    
    ALLOWED_ACTIONS = {'click', 'type', 'scroll', 'wait', 'screenshot', 'extract'}
    BLOCKED_PATTERNS = [
        'eval(',
        'document.cookie',
        'localStorage.clear',
        'window.location =',
    ]
    
    def validate_action(self, action: dict) -> tuple:
        """Validate action is safe to execute"""
        action_type = action.get('type')
        
        if action_type not in self.ALLOWED_ACTIONS:
            return False, f"Action type '{action_type}' not allowed"
        
        # Check for dangerous patterns in parameters
        params_str = str(action.get('params', ''))
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in params_str:
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, "OK"
    
    def validate_javascript(self, script: str) -> tuple:
        """Validate JavaScript before execution"""
        dangerous_patterns = [
            'document.cookie',
            'localStorage',
            'sessionStorage',
            'XMLHttpRequest',
            'fetch(',
            'eval(',
            'Function(',
        ]
        
        for pattern in dangerous_patterns:
            if pattern in script:
                return False, f"JavaScript contains blocked pattern: {pattern}"
        
        return True, "OK"
```

---

## Comparison Matrix

### Browser Automation Frameworks

| Framework | Language | Speed | Ease of Use | Community | Price | Best For |
|---|---|---|---|---|---|---|
| **Playwright** | Python/JS/Java/C# | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | New projects |
| **Puppeteer** | JavaScript | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | Node.js projects |
| **Selenium** | Multi-lang | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Free | Legacy systems |

### AI Agent Frameworks

| Framework | Model Support | Vision | Custom Actions | Cost | Maturity |
|---|---|---|---|---|---|
| **Browser Use** | Any | Yes | Yes | Free | Production |
| **Stagehand** | Any | Yes | Yes | Free | Early |
| **AgentQL** | Any | Partial | Yes | Free | Beta |
| **AutoGPT** | Any | No | Limited | Free | Production |

### Infrastructure Services

| Service | Type | Global Coverage | Price | Reliability |
|---|---|---|---|---|
| **Browserbase** | Cloud browser | Yes | $$ | ⭐⭐⭐⭐ |
| **Bright Data** | Proxy+Browser | 195 countries | $$$ | ⭐⭐⭐⭐⭐ |
| **Browserless** | Self-hosted | Your infra | Free | ⭐⭐⭐⭐ |
| **ScraperAPI** | Proxy+Render | Yes | $$ | ⭐⭐⭐⭐ |

---

## Integration Patterns

### Plugin Architecture

```python
from abc import ABC, abstractmethod

class BrowserPlugin(ABC):
    """Base class for browser agent plugins"""
    
    @abstractmethod
    async def on_page_load(self, page):
        pass
    
    @abstractmethod
    async def before_action(self, action: dict) -> dict:
        pass
    
    async def after_action(self, action: dict, result: dict):
        pass
    
    async def on_error(self, error: Exception, page):
        pass

class AutoScrollPlugin(BrowserPlugin):
    """Plugin to auto-scroll through content"""
    
    async def on_page_load(self, page):
        await page.evaluate("""
            () => {
                window.scrollTo(0, document.body.scrollHeight / 2);
            }
        """)
    
    async def before_action(self, action):
        return action

class CookieConsentPlugin(BrowserPlugin):
    """Plugin to handle cookie consent dialogs"""
    
    async def on_page_load(self, page):
        try:
            await page.click('[class*="accept"]', timeout=2000)
        except:
            pass
    
    async def before_action(self, action):
        return action

class AgentWithPlugins:
    """Agent with plugin support"""
    
    def __init__(self):
        self.plugins: list[BrowserPlugin] = []
    
    def add_plugin(self, plugin: BrowserPlugin):
        self.plugins.append(plugin)
    
    async def execute_action(self, page, action: dict):
        # Run before_action hooks
        for plugin in self.plugins:
            action = await plugin.before_action(action)
        
        # Execute action
        try:
            result = await self._do_action(page, action)
        except Exception as e:
            for plugin in self.plugins:
                await plugin.on_error(e, page)
            raise
        
        # Run after_action hooks
        for plugin in self.plugins:
            await plugin.after_action(action, result)
        
        return result
```

---

## Best Practices

### 1. Choose the Right Tool for the Job

- **Simple web scraping:** Playwright + BeautifulSoup
- **Complex multi-step tasks:** Browser Use or Stagehand
- **High-volume scraping:** ScraperAPI + rotating proxies
- **Desktop automation:** Anthropic Computer Use

### 2. Implement Robust Error Handling

- Always have retry logic with exponential backoff
- Handle CAPTCHAs gracefully (detect and escalate)
- Implement timeout mechanisms for all operations
- Log all errors for debugging

### 3. Optimize for Cost

- Use headless mode when possible
- Minimize screenshot frequency
- Cache page states
- Use cheaper models for simple actions

### 4. Security First

- Run in sandboxed environments
- Validate all actions before execution
- Don't store credentials in code
- Use secure credential management

### 5. Monitor and Observe

- Track key metrics (success rate, duration, cost)
- Record sessions for debugging
- Set up alerts for failure patterns
- Use distributed tracing

### 6. Design for Scale

- Use browser pools for concurrent operations
- Implement connection pooling
- Queue tasks for ordered processing
- Use cloud infrastructure for burst capacity

---

## Cross-References

- **[01-Overview.md](01-Overview.md)** — Introduction to agentic browser automation
- **[02-Computer-Use-Frameworks.md](02-Computer-Use-Frameworks.md)** — Computer use frameworks and desktop automation
- **[03-Browser-Agent-Architectures.md](03-Browser-Agent-Architectures.md)** — Browser agent architectural patterns
- **[05-Production-Deployment-and-Security.md](05-Production-Deployment-and-Security.md)** — Production deployment patterns
- **[03-Agents/03-Agentic-Frameworks.md](../03-Agents/03-Agentic-Frameworks.md)** — General agentic framework patterns
- **[03-Agents/05-Tool-Implementations.md](../03-Agents/05-Tool-Implementations.md)** — Tool implementation patterns
- **[20-Agent-Infrastructure/03-Agent-Tracing-and-Observability.md](../20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md)** — Observability patterns
- **[20-Agent-Infrastructure/04-Agent-Evaluation-and-Testing.md](../20-Agent-Infrastructure-and-Observability/04-Agent-Evaluation-and-Testing.md)** — Testing patterns

---

*Last updated: July 2026*
*See also: Playwright docs, Puppeteer docs, Browser Use GitHub, Stagehand docs*
