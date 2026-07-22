# Browser Agent Architectures: Design Patterns for Web Automation

> **Description:** Deep dive into the architectural patterns, protocols, and design decisions behind browser-based AI agents. Covers DOM-aware agents, vision-only agents, hybrid approaches, multi-tab coordination, session management, and state machine patterns for reliable web automation.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [DOM-Architecture Pattern](#dom-architecture-pattern)
3. [Vision-Only Architecture](#vision-only-architecture)
4. [Hybrid Architecture](#hybrid-architecture)
5. [Session and State Management](#session-and-state-management)
6. [Multi-Tab Coordination](#multi-tab-coordination)
7. [Page Object Abstraction](#page-object-abstraction)
8. [Action Planning and Execution](#action-planning-and-execution)
9. [Network Interception](#network-interception)
10. [JavaScript Execution Layer](#javascript-execution-layer)
11. [Error Handling and Recovery](#error-handling-and-recovery)
12. [Performance Patterns](#performance-patterns)
13. [Scaling Browser Agents](#scaling-browser-agents)
14. [Code Examples](#code-examples)
15. [Architecture Decision Records](#architecture-decision-records)
16. [Cross-References](#cross-references)

---

## Architecture Overview

Browser agents operate in a fundamentally different environment than API-based agents. They must navigate complex, dynamic web applications with varying frameworks, security policies, and interaction patterns.

### Core Architecture Layers

```
┌──────────────────────────────────────────────────┐
│                   LLM / Planning Layer            │
│   (GPT-4o, Claude, Gemini - decides what to do)   │
├──────────────────────────────────────────────────┤
│                 Abstraction Layer                 │
│   (Page Objects, DOM Models, Action Primitives)    │
├──────────────────────────────────────────────────┤
│              Browser Control Layer                │
│   (CDP, WebDriver, Playwright, Puppeteer)          │
├──────────────────────────────────────────────────┤
│               Browser Runtime                    │
│   (Chrome, Firefox, WebKit - renders pages)       │
├──────────────────────────────────────────────────┤
│              Infrastructure Layer                 │
│   (Docker, VMs, Browser Pools, Storage)           │
└──────────────────────────────────────────────────┘
```

### Three Main Architecture Families

| Architecture | Input | Interaction | Strengths | Weaknesses |
|---|---|---|---|---|
| **DOM-Aware** | HTML/accessibility tree | CSS selectors, DOM events | Fast, reliable, token-efficient | Breaks with complex JS |
| **Vision-Only** | Screenshots | Pixel coordinates | Works with any UI | Slow, expensive, fragile |
| **Hybrid** | DOM + Screenshots | Combined | Best of both worlds | Complex to implement |

---

## DOM-Architecture Pattern

### How DOM-Aware Agents Work

DOM-aware agents use the browser's Document Object Model to understand page structure:

```python
class DOMAgent:
    def __init__(self, page):
        self.page = page
    
    async def get_page_state(self):
        """Extract structured page state from DOM"""
        state = await self.page.evaluate("""
            () => {
                const elements = [];
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_ELEMENT
                );
                
                let node;
                while (node = walker.nextNode()) {
                    const rect = node.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    
                    elements.push({
                        tag: node.tagName.toLowerCase(),
                        text: node.textContent?.trim().slice(0, 100),
                        role: node.getAttribute('role'),
                        ariaLabel: node.getAttribute('aria-label'),
                        placeholder: node.getAttribute('placeholder'),
                        type: node.getAttribute('type'),
                        href: node.getAttribute('href'),
                        bounds: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        },
                        visible: rect.top < window.innerHeight,
                        interactive: isInteractive(node)
                    });
                }
                return elements;
            }
        """)
        return state
```

### Accessibility Tree Agents

The accessibility tree provides semantic understanding of page structure:

```python
class AccessibilityTreeAgent:
    def __init__(self, page):
        self.page = page
    
    async def get_accessibility_tree(self):
        """Get the accessibility tree snapshot"""
        snapshot = await self.page.accessibility.snapshot()
        return self.flatten_tree(snapshot)
    
    def flatten_tree(self, node, depth=0):
        """Flatten accessibility tree into actionable list"""
        result = []
        
        if node.get('role'):
            result.append({
                'role': node['role'],
                'name': node.get('name', ''),
                'value': node.get('value'),
                'description': node.get('description'),
                'focused': node.get('focused', False),
                'depth': depth,
                'child_count': len(node.get('children', []))
            })
        
        for child in node.get('children', []):
            result.extend(self.flatten_tree(child, depth + 1))
        
        return result
    
    async def find_element(self, description: str):
        """Find element by natural language description"""
        tree = await self.get_accessibility_tree()
        
        # Use LLM to match description to tree elements
        prompt = f"""
        Accessibility tree:
        {json.dumps(tree[:50], indent=2)}
        
        Find the element that matches: "{description}"
        Return the index of the matching element.
        """
        
        response = await self.llm.chat(prompt)
        element_index = int(response.strip())
        return tree[element_index]
```

### Selector-Based Interaction

```python
class SelectorAgent:
    """Agent that uses CSS/XPath selectors for interaction"""
    
    async def click_element(self, selector: str):
        """Click element by selector"""
        await self.page.click(selector)
    
    async def fill_input(self, selector: str, value: str):
        """Fill input by selector"""
        await self.page.fill(selector, value)
    
    async def select_option(self, selector: str, value: str):
        """Select dropdown option by value"""
        await self.page.select_option(selector, value)
    
    async def extract_text(self, selector: str) -> str:
        """Extract text content from element"""
        return await self.page.text_content(selector)
    
    async def wait_for_element(self, selector: str, timeout: int = 5000):
        """Wait for element to appear"""
        await self.page.wait_for_selector(selector, timeout=timeout)
    
    # LLM-guided selector generation
    async def generate_selector(self, description: str, context: str) -> str:
        """Generate CSS selector from natural language description"""
        prompt = f"""
        Given this HTML context:
        {context}
        
        Generate a CSS selector that uniquely identifies the element described as:
        "{description}"
        
        Return ONLY the CSS selector string.
        """
        return await self.llm.chat(prompt)
```

---

## Vision-Only Architecture

### Pure Vision Pipeline

```
Screenshot → Preprocessing → Vision Encoder → Action Decoder → Execute
     │            │               │                │              │
  Capture      Resize/       Process with      Generate       Perform
  screen       crop/         vision model      action         mouse/
               annotate                        command        keyboard
```

### Screenshot Analysis

```python
class VisionAgent:
    def __init__(self, vision_model, action_model):
        self.vision = vision_model
        self.planner = action_model
    
    async def analyze_screenshot(self, screenshot: bytes, task: str):
        """Analyze screenshot and determine next action"""
        
        # Step 1: Describe what's on screen
        description = await self.vision.chat(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this screenshot in detail, focusing on interactive elements, text content, and current state."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode(screenshot)}"}}
                ]
            }]
        )
        
        # Step 2: Plan action based on description and task
        action = await self.planner.chat(
            messages=[{
                "role": "system",
                "content": """You are a browser automation agent. Given a screenshot description 
                and a task, determine the next action to take.
                
                Available actions:
                - click(x, y): Click at coordinates
                - type(text): Type text
                - scroll(direction): Scroll up/down
                - wait(seconds): Wait for page update
                - done(result): Task complete"""
            },
            {
                "role": "user", 
                "content": f"""Task: {task}
                
                Current screen state:
                {description}
                
                What action should I take next?"""
            }]
        )
        
        return self.parse_action(action)
```

### Visual Element Detection

```python
class VisualElementDetector:
    """Detect interactive elements from screenshots using vision models"""
    
    async def detect_elements(self, screenshot: bytes):
        prompt = """
        Analyze this screenshot and identify all interactive elements:
        
        For each element, provide:
        1. Type (button, input, link, dropdown, etc.)
        2. Label/text content
        3. Bounding box coordinates [x, y, width, height]
        4. State (enabled, disabled, focused, etc.)
        
        Return as JSON array.
        """
        
        response = await self.vision.analyze(screenshot, prompt)
        return json.loads(response)
    
    async def find_element(self, screenshot: bytes, description: str):
        """Find a specific element by description"""
        elements = await self.detect_elements(screenshot)
        
        prompt = f"""
        Elements found:
        {json.dumps(elements, indent=2)}
        
        Which element matches: "{description}"?
        Return the element index.
        """
        
        index = await self.llm.chat(prompt)
        return elements[int(index.strip())]
```

---

## Hybrid Architecture

### Combining DOM and Vision

```python
class HybridAgent:
    """Agent that uses both DOM and vision for maximum reliability"""
    
    def __init__(self, page, vision_model, llm):
        self.page = page
        self.vision = vision_model
        self.llm = llm
    
    async def execute_task(self, task: str):
        """Execute task using hybrid DOM + vision approach"""
        
        for step in range(self.max_steps):
            # Try DOM first (faster, more reliable)
            dom_state = await self.get_dom_state()
            dom_action = await self.plan_with_dom(task, dom_state)
            
            if dom_action and dom_action.confidence > 0.8:
                await self.execute_dom_action(dom_action)
                continue
            
            # Fall back to vision (more flexible)
            screenshot = await self.page.screenshot()
            vision_action = await self.plan_with_vision(task, screenshot)
            
            if vision_action:
                await self.execute_vision_action(vision_action)
                continue
            
            # Neither worked - ask LLM to analyze
            combined = await self.analyze_combined(dom_state, screenshot, task)
            await self.execute_combined_action(combined)
    
    async def get_dom_state(self):
        """Get DOM representation with fallback to vision"""
        try:
            return await self.extract_accessibility_tree()
        except Exception:
            return None
    
    async def plan_with_dom(self, task: str, dom_state: dict):
        """Plan action using DOM state"""
        prompt = f"""
        Task: {task}
        
        DOM State:
        {json.dumps(dom_state, indent=2)[:2000]}
        
        What action should I take? Return JSON with type, selector, value.
        """
        response = await self.llm.chat(prompt)
        return self.parse_action(response)
    
    async def plan_with_vision(self, task: str, screenshot: bytes):
        """Plan action using screenshot"""
        response = await self.vision.analyze(
            screenshot,
            f"Task: {task}. What action should I take? Return x,y coordinates and action type."
        )
        return self.parse_action(response)
```

### Confidence-Based Routing

```python
class ConfidenceRouter:
    """Route actions based on confidence scores"""
    
    def __init__(self, dom_threshold=0.8, vision_threshold=0.6):
        self.dom_threshold = dom_threshold
        self.vision_threshold = vision_threshold
    
    async def get_action(self, task: str, page) -> Action:
        # DOM analysis
        dom_result = await self.analyze_dom(task, page)
        if dom_result.confidence >= self.dom_threshold:
            return dom_result
        
        # Vision analysis
        screenshot = await page.screenshot()
        vision_result = await self.analyze_vision(task, screenshot)
        if vision_result.confidence >= self.vision_threshold:
            return vision_result
        
        # Combined analysis
        combined = await self.analyze_combined(dom_result, vision_result)
        return combined
```

---

## Session and State Management

### Browser Session Lifecycle

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐
│  Create  │───▶│ Configure │───▶│ Navigate │───▶│  Active  │
└─────────┘    └──────────┘    └─────────┘    └────┬────┘
                                                    │
                                               ┌────▼────┐
                                               │ Paused  │
                                               └────┬────┘
                                                    │
                                        ┌───────────▼───────────┐
                                        │                       │
                                   ┌────▼────┐            ┌────▼────┐
                                   │Complete │            │  Error  │
                                   └─────────┘            └────┬────┘
                                                               │
                                                          ┌────▼────┐
                                                          │ Recovery│
                                                          └─────────┘
```

### Session State Management

```python
class BrowserSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = {}
        self.history = []
        self.cookies = {}
        self.local_storage = {}
    
    async def save_state(self, page):
        """Save complete browser state"""
        self.state = {
            'url': page.url,
            'title': await page.title(),
            'cookies': await page.context.cookies(),
            'local_storage': await page.evaluate('() => JSON.stringify(localStorage)'),
            'screenshot': await page.screenshot(),
            'timestamp': time.time()
        }
        self.history.append(self.state.copy())
    
    async def restore_state(self, context):
        """Restore browser state"""
        # Restore cookies
        await context.add_cookies(self.state['cookies'])
        
        # Restore local storage
        page = await context.new_page()
        await page.goto(self.state['url'])
        await page.evaluate(f"""
            () => {{
                const data = JSON.parse('{self.state['local_storage']}');
                Object.entries(data).forEach(([k, v]) => localStorage.setItem(k, v));
            }}
        """)
        return page
    
    def get_navigation_history(self):
        """Get list of visited URLs"""
        return [s['url'] for s in self.history]
    
    def get_action_history(self):
        """Get list of actions taken"""
        return [s.get('action') for s in self.history if s.get('action')]
```

---

## Multi-Tab Coordination

### Tab Management

```python
class MultiTabManager:
    def __init__(self, context):
        self.context = context
        self.tabs = {}
        self.active_tab = None
    
    async def open_tab(self, url: str, name: str = None):
        """Open new tab with optional name"""
        page = await self.context.new_page()
        await page.goto(url)
        
        tab_id = name or f"tab_{len(self.tabs)}"
        self.tabs[tab_id] = {
            'page': page,
            'url': url,
            'created_at': time.time()
        }
        return tab_id
    
    async def switch_tab(self, tab_id: str):
        """Switch to named tab"""
        if tab_id not in self.tabs:
            raise ValueError(f"Tab {tab_id} not found")
        self.active_tab = tab_id
        return self.tabs[tab_id]['page']
    
    async def close_tab(self, tab_id: str):
        """Close named tab"""
        if tab_id in self.tabs:
            await self.tabs[tab_id]['page'].close()
            del self.tabs[tab_id]
    
    async def get_tab_screenshots(self):
        """Capture screenshots of all tabs"""
        screenshots = {}
        for tab_id, tab in self.tabs.items():
            screenshots[tab_id] = {
                'url': tab['url'],
                'screenshot': await tab['page'].screenshot(),
                'title': await tab['page'].title()
            }
        return screenshots
    
    async def coordinate_tabs(self, task: str):
        """Coordinate actions across multiple tabs"""
        # Analyze which tab to work on
        screenshots = await self.get_tab_screenshots()
        
        prompt = f"""
        Task: {task}
        
        Available tabs:
        {json.dumps({k: {'url': v['url'], 'title': v['title']} for k, v in screenshots.items()}, indent=2)}
        
        Which tab should I work on next? Return tab_id.
        """
        
        tab_id = await self.llm.chat(prompt)
        page = await self.switch_tab(tab_id.strip())
        return page
```

### Cross-Tab Data Transfer

```python
class CrossTabCoordinator:
    """Share data between tabs"""
    
    def __init__(self):
        self.shared_state = {}
    
    async def extract_from_tab(self, page, extraction_prompt: str):
        """Extract data from a page"""
        screenshot = await page.screenshot()
        
        response = await self.vision.analyze(
            screenshot,
            f"Extract the following information: {extraction_prompt}"
        )
        return json.loads(response)
    
    async def use_in_tab(self, page, data: dict, task: str):
        """Use extracted data in another tab"""
        prompt = f"""
        Data available:
        {json.dumps(data, indent=2)}
        
        Task: {task}
        
        What actions should I take on this page?
        """
        
        action = await self.llm.chat(prompt)
        await self.execute_action(page, action)
```

---

## Page Object Abstraction

### Page Object Pattern

```python
class PageObject:
    """Abstract base class for page objects"""
    
    def __init__(self, page):
        self.page = page
    
    async def wait_for_load(self):
        await self.page.wait_for_load_state('networkidle')
    
    async def is_loaded(self) -> bool:
        try:
            await self.page.wait_for_load_state('domcontentloaded', timeout=3000)
            return True
        except:
            return False

class LoginPage(PageObject):
    URL = "https://example.com/login"
    
    # Locators
    USERNAME_INPUT = "#username"
    PASSWORD_INPUT = "#password"
    SUBMIT_BUTTON = "button[type=submit]"
    ERROR_MESSAGE = ".error-message"
    
    async def navigate(self):
        await self.page.goto(self.URL)
        await self.wait_for_load()
    
    async def login(self, username: str, password: str):
        await self.page.fill(self.USERNAME_INPUT, username)
        await self.page.fill(self.PASSWORD_INPUT, password)
        await self.page.click(self.SUBMIT_BUTTON)
        await self.page.wait_for_url("**/dashboard**")
    
    async def get_error_message(self) -> str:
        return await self.page.text_content(self.ERROR_MESSAGE)

class DashboardPage(PageObject):
    URL = "https://example.com/dashboard"
    
    # Locators
    USER_MENU = ".user-menu"
    SEARCH_INPUT = "#search"
    RESULTS_CONTAINER = ".results"
    
    async def search(self, query: str):
        await self.page.fill(self.SEARCH_INPUT, query)
        await self.page.press(self.SEARCH_INPUT, "Enter")
        await self.wait_for_load()
    
    async def get_results(self) -> list:
        return await self.page.query_selector_all(f"{self.RESULTS_CONTAINER} .result-item")
```

### Dynamic Page Object Generation

```python
class DynamicPageFactory:
    """Generate page objects from page analysis"""
    
    async def create_page_object(self, page, page_type: str):
        """Dynamically create page object by analyzing page structure"""
        
        # Get page structure
        structure = await page.evaluate("""
            () => {
                const inputs = Array.from(document.querySelectorAll('input, select, textarea'));
                const buttons = Array.from(document.querySelectorAll('button, [role=button]'));
                const links = Array.from(document.querySelectorAll('a'));
                
                return {
                    inputs: inputs.map(i => ({
                        id: i.id,
                        name: i.name,
                        type: i.type,
                        placeholder: i.placeholder,
                        label: i.labels?.[0]?.textContent
                    })),
                    buttons: buttons.map(b => ({
                        id: b.id,
                        text: b.textContent.trim(),
                        type: b.type
                    })),
                    links: links.slice(0, 20).map(l => ({
                        text: l.textContent.trim(),
                        href: l.href
                    }))
                };
            }
        """)
        
        # Generate page object code
        prompt = f"""
        Generate a Python page object class for this page structure:
        {json.dumps(structure, indent=2)}
        
        Page type: {page_type}
        """
        
        code = await self.llm.chat(prompt)
        return self.compile_page_object(code)
```

---

## Action Planning and Execution

### ReAct-Style Planning

```python
class ReactPlanner:
    """ReAct-style planning for browser agents"""
    
    async def plan_and_execute(self, task: str, page):
        thought_history = []
        
        for step in range(self.max_steps):
            # Get current state
            screenshot = await page.screenshot()
            url = page.url
            
            # Think
            thought = await self.think(task, screenshot, url, thought_history)
            thought_history.append(thought)
            
            # Decide action
            action = await self.decide_action(thought)
            
            # Execute
            if action.type == "done":
                return {"status": "success", "result": action.result}
            
            result = await self.execute_action(page, action)
            
            # Observe
            await asyncio.sleep(1)
            new_screenshot = await page.screenshot()
            observation = await self.observe(new_screenshot, action, result)
            thought_history.append(observation)
        
        return {"status": "failed", "max_steps_reached": True}
    
    async def think(self, task, screenshot, url, history):
        prompt = f"""
        Task: {task}
        Current URL: {url}
        History: {json.dumps(history[-5:])}
        
        Analyze the current situation and think about the next step.
        What do you see? What should you do next?
        """
        return await self.llm.chat(prompt)
    
    async def decide_action(self, thought):
        prompt = f"""
        Based on this thought: {thought}
        
        Decide the next action. Return JSON:
        {{"type": "click|type|scroll|wait|done", "params": {{...}}}}
        """
        response = await self.llm.chat(prompt)
        return json.loads(response)
    
    async def observe(self, screenshot, action, result):
        prompt = f"""
        After executing action {action}, observe the result:
        
        What changed on screen? Did the action have the expected effect?
        """
        return await self.vision.analyze(screenshot, prompt)
```

### Chain-of-Thought for Browser Tasks

```python
class CoTBrowserAgent:
    """Chain-of-thought reasoning for browser automation"""
    
    async def execute_with_reasoning(self, task: str, page):
        # Break task into subgoals
        subgoals = await self.decompose_task(task)
        
        results = []
        for subgoal in subgoals:
            # Reason about subgoal
            reasoning = await self.reason(subgoal, page)
            
            # Execute
            result = await self.execute_subgoal(subgoal, reasoning, page)
            results.append(result)
            
            # Verify
            verified = await self.verify(result, subgoal, page)
            if not verified:
                # Re-plan
                result = await self.recover(subgoal, result, page)
        
        return results
    
    async def decompose_task(self, task: str) -> list:
        prompt = f"""
        Task: {task}
        
        Decompose this into browser-specific subgoals.
        Each subgoal should be a single, atomic action.
        """
        response = await self.llm.chat(prompt)
        return self.parse_subgoals(response)
```

---

## Network Interception

### Request/Response Interception

```python
class NetworkInterceptor:
    """Intercept and modify network requests"""
    
    def __init__(self, page):
        self.page = page
        self.captured_requests = []
        self.captured_responses = []
    
    async def setup_interception(self, patterns=None):
        """Setup request interception"""
        await self.page.route("**/*", self.handle_route)
    
    async def handle_route(self, route):
        """Handle intercepted request"""
        request = route.request
        
        # Log request
        self.captured_requests.append({
            'url': request.url,
            'method': request.method,
            'headers': request.headers,
            'post_data': request.post_data
        })
        
        # Continue request
        await route.continue_()
    
    async def intercept_api_calls(self):
        """Capture API responses for data extraction"""
        responses = []
        
        self.page.on('response', lambda resp: responses.append({
            'url': resp.url,
            'status': resp.status,
            'body': resp.body()
        }))
        
        return responses
    
    def get_api_data(self, url_pattern: str):
        """Get captured API data matching pattern"""
        import re
        return [
            r for r in self.captured_responses
            if re.match(url_pattern, r['url'])
        ]
```

### API Response Extraction

```python
class APIExtractor:
    """Extract data from intercepted API responses"""
    
    def __init__(self, page):
        self.page = page
        self.interceptor = NetworkInterceptor(page)
    
    async def extract_with_api(self, extraction_task: str):
        """Extract data using both UI and API interception"""
        
        # Setup API capture
        api_data = []
        self.page.on('response', lambda r: api_data.append(r))
        
        # Navigate to page
        await self.page.goto("https://example.com/data-page")
        await self.page.wait_for_load_state('networkidle')
        
        # Find API responses that contain the data
        for response in api_data:
            if response.url.endswith('/api/data'):
                data = await response.json()
                return data
        
        # Fall back to DOM extraction
        return await self.extract_from_dom()
```

---

## JavaScript Execution Layer

### In-Page Agent Execution

```python
class InPageAgent:
    """Execute agent logic inside the browser page"""
    
    async def inject_agent(self, page):
        """Inject agent helper functions into page"""
        await page.add_init_script("""
            window.__agent = {
                // Find element by text
                findByText: (text) => {
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT
                    );
                    while (walker.nextNode()) {
                        if (walker.currentNode.textContent.includes(text)) {
                            return walker.currentNode.parentElement;
                        }
                    }
                    return null;
                },
                
                // Get element at coordinates
                getElementAt: (x, y) => {
                    return document.elementFromPoint(x, y);
                },
                
                // Click element by text
                clickByText: (text) => {
                    const el = this.findByText(text);
                    if (el) {
                        el.click();
                        return true;
                    }
                    return false;
                },
                
                // Fill form field
                fillField: (selector, value) => {
                    const el = document.querySelector(selector);
                    if (el) {
                        el.value = value;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        return true;
                    }
                    return false;
                }
            };
        """)
    
    async def execute_in_page(self, page, script: str):
        """Execute custom script in page context"""
        return await page.evaluate(script)
```

### Shadow DOM Handling

```python
class ShadowDOMHandler:
    """Handle Shadow DOM elements"""
    
    async def find_in_shadow(self, page, selector: str):
        """Find element in Shadow DOM"""
        return await page.evaluate(f"""
            () => {{
                function findInShadow(root, selector) {{
                    // Check shadow roots
                    const elements = root.querySelectorAll('*');
                    for (const el of elements) {{
                        if (el.shadowRoot) {{
                            const found = el.shadowRoot.querySelector(selector);
                            if (found) return found;
                            
                            // Recurse into shadow root
                            const deep = findInShadow(el.shadowRoot, selector);
                            if (deep) return deep;
                        }}
                    }}
                    return root.querySelector(selector);
                }}
                
                return findInShadow(document, '{selector}');
            }}
        """)
```

---

## Error Handling and Recovery

### Error Classification

```python
class BrowserError(Enum):
    PAGE_NOT_LOADED = "page_not_loaded"
    ELEMENT_NOT_FOUND = "element_not_found"
    STALE_ELEMENT = "stale_element"
    NAVIGATION_TIMEOUT = "navigation_timeout"
    CLICK_INTERCEPTED = "click_intercepted"
    CONSENT_DIALOG = "consent_dialog"
    CAPTCHA_DETECTED = "captcha_detected"
    LOGIN_REQUIRED = "login_required"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"

class ErrorHandler:
    """Classify and handle browser errors"""
    
    async def classify_error(self, error, page) -> BrowserError:
        error_str = str(error).lower()
        
        if "not found" in error_str or "no node" in error_str:
            return BrowserError.ELEMENT_NOT_FOUND
        elif "stale" in error_str:
            return BrowserError.STALE_ELEMENT
        elif "timeout" in error_str:
            return BrowserError.NAVIGATION_TIMEOUT
        elif "intercept" in error_str:
            return BrowserError.CLICK_INTERCEPTED
        elif "captcha" in error_str or "verify" in error_str:
            return BrowserError.CAPTCHA_DETECTED
        else:
            return BrowserError.UNKNOWN
    
    async def recover(self, error_type: BrowserError, page):
        recovery_strategies = {
            BrowserError.ELEMENT_NOT_FOUND: self.retry_with_wait,
            BrowserError.STALE_ELEMENT: self.refresh_and_retry,
            BrowserError.NAVIGATION_TIMEOUT: self.retry_navigation,
            BrowserError.CAPTCHA_DETECTED: self.alert_human,
            BrowserError.CONSENT_DIALOG: self.handle_consent,
        }
        
        strategy = recovery_strategies.get(error_type, self.generic_recovery)
        return await strategy(page)
```

### Consent Dialog Handling

```python
class ConsentHandler:
    """Handle cookie consent and similar dialogs"""
    
    COMMON_SELECTORS = [
        "button[id*='accept']",
        "button[class*='accept']",
        "button:has-text('Accept')",
        "button:has-text('Accept All')",
        "button:has-text('OK')",
        "button:has-text('Got it')",
        "#onetrust-accept-btn-handler",
        ".cc-accept",
        "[data-testid='cookie-accept']"
    ]
    
    async def dismiss_consent(self, page):
        """Try to dismiss common consent dialogs"""
        for selector in self.COMMON_SELECTORS:
            try:
                button = await page.query_selector(selector)
                if button and await button.is_visible():
                    await button.click()
                    await asyncio.sleep(0.5)
                    return True
            except:
                continue
        return False
```

---

## Performance Patterns

### Parallel Page Loading

```python
class ParallelLoader:
    """Load multiple pages in parallel"""
    
    async def preload_pages(self, context, urls: list):
        """Preload multiple pages simultaneously"""
        pages = []
        for url in urls:
            page = await context.new_page()
            pages.append(page)
            await page.goto(url)  # Don't wait for full load
        
        # Wait for all to be ready
        for page in pages:
            await page.wait_for_load_state('domcontentloaded')
        
        return pages
```

### Caching and Memoization

```python
class PageCache:
    """Cache page state to avoid redundant operations"""
    
    def __init__(self):
        self.cache = {}
        self.ttl = 300  # 5 minutes
    
    def get_page_hash(self, url: str, dom_hash: str) -> str:
        return hashlib.md5(f"{url}:{dom_hash}".encode()).hexdigest()
    
    def cache_state(self, key: str, state: dict):
        self.cache[key] = {
            'state': state,
            'timestamp': time.time()
        }
    
    def get_cached_state(self, key: str) -> dict:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['state']
        return None
```

---

## Scaling Browser Agents

### Browser Pool Management

```python
class BrowserPool:
    """Manage a pool of browser instances"""
    
    def __init__(self, max_browsers: int = 5):
        self.max = max_browsers
        self.available = []
        self.in_use = {}
        self.launcher = None
    
    async def initialize(self):
        """Launch browser pool"""
        self.launcher = await async_playwright().start()
        for i in range(self.max):
            browser = await self.launcher.chromium.launch()
            self.available.append(browser)
    
    async def acquire(self):
        """Get a browser from the pool"""
        if not self.available:
            await self.wait_for_available()
        
        browser = self.available.pop()
        browser_id = id(browser)
        self.in_use[browser_id] = browser
        return browser_id, browser
    
    async def release(self, browser_id: str):
        """Return browser to pool"""
        browser = self.in_use.pop(browser_id)
        await browser.close()
        # Launch replacement
        new_browser = await self.launcher.chromium.launch()
        self.available.append(new_browser)
```

### Task Queue System

```python
import asyncio
from dataclasses import dataclass
from typing import Callable

@dataclass
class BrowserTask:
    id: str
    url: str
    action: Callable
    priority: int = 0
    retries: int = 3

class TaskQueue:
    """Queue for browser automation tasks"""
    
    def __init__(self, pool: BrowserPool, workers: int = 3):
        self.pool = pool
        self.queue = asyncio.PriorityQueue()
        self.workers = workers
    
    async def add_task(self, task: BrowserTask):
        await self.queue.put((-task.priority, task))
    
    async def process(self):
        """Process tasks with worker pool"""
        tasks = [
            asyncio.create_task(self.worker(f"worker_{i}"))
            for i in range(self.workers)
        ]
        await asyncio.gather(*tasks)
    
    async def worker(self, name: str):
        while True:
            try:
                _, task = await self.queue.get()
                
                browser_id, browser = await self.pool.acquire()
                try:
                    page = await browser.new_page()
                    await page.goto(task.url)
                    result = await task.action(page)
                    return result
                finally:
                    await self.pool.release(browser_id)
                    
            except Exception as e:
                if task.retries > 0:
                    task.retries -= 1
                    await self.queue.put((-task.priority, task))
```

---

## Code Examples

### Example 1: E-Commerce Automation Agent

```python
class ECommerceAgent:
    """Automate e-commerce workflows"""
    
    async def find_cheapest(self, product: str, sites: list):
        results = []
        
        for site in sites:
            page = await self.browser.new_page()
            await page.goto(site['url'])
            
            # Search for product
            await page.fill(site['search_selector'], product)
            await page.press(site['search_selector'], 'Enter')
            await page.wait_for_load_state('networkidle')
            
            # Extract prices
            prices = await page.evaluate("""
                () => {
                    const items = document.querySelectorAll('.product-item');
                    return Array.from(items).map(item => ({
                        name: item.querySelector('.product-name')?.textContent,
                        price: item.querySelector('.price')?.textContent,
                        url: item.querySelector('a')?.href
                    }));
                }
            """)
            
            results.extend(prices)
            await page.close()
        
        # Sort by price and return cheapest
        return sorted(results, key=lambda x: float(x['price'].replace('$', '')))[0]
```

### Example 2: Form Automation Pipeline

```python
class FormPipeline:
    """Multi-step form automation"""
    
    async def fill_application(self, page, form_data: dict):
        steps = [
            ("Personal Info", self.fill_personal_info),
            ("Address", self.fill_address),
            ("Employment", self.fill_employment),
            ("Review", self.review_and_submit)
        ]
        
        for step_name, handler in steps:
            print(f"Executing step: {step_name}")
            
            try:
                await handler(page, form_data)
                
                # Verify step completed
                screenshot = await page.screenshot()
                verified = await self.verify_step(screenshot, step_name)
                
                if not verified:
                    await self.retry_step(page, handler, form_data)
                
                # Navigate to next step
                await page.click('button.next')
                await page.wait_for_load_state('networkidle')
                
            except Exception as e:
                await self.handle_step_error(e, page, step_name)
        
        return await self.get_confirmation(page)
```

---

## Architecture Decision Records

### ADR-001: DOM vs Vision First

**Decision:** Use DOM-first with vision fallback

**Rationale:**
- DOM interaction is 5-10x faster than vision
- DOM is more reliable for known page structures
- Vision handles edge cases DOM cannot
- Combined approach gives maximum coverage

**Consequences:**
- Need to maintain both DOM and vision pipelines
- Higher initial implementation complexity
- Better overall reliability and performance

### ADR-002: Session Persistence

**Decision:** Persist session state across agent restarts

**Rationale:**
- Long-running tasks may need restarts
- Session cookies are expensive to re-obtain
- User authentication state is valuable
- Checkpoint capability improves reliability

**Consequences:**
- Need secure storage for session data
- Privacy considerations for stored data
- Additional complexity in session management

---

## Cross-References

- **[01-Overview.md](01-Overview.md)** — Introduction to agentic browser automation
- **[02-Computer-Use-Frameworks.md](02-Computer-Use-Frameworks.md)** — Computer use frameworks and desktop automation
- **[04-Tools-and-Libraries.md](04-Tools-and-Libraries.md)** — Complete tooling ecosystem
- **[05-Production-Deployment-and-Security.md](05-Production-Deployment-and-Security.md)** — Production deployment patterns
- **[03-Agents/04-Protocols-MCP-ACP.md](../03-Agents/04-Protocols-MCP-ACP.md)** — Communication protocols for agents
- **[04-RAG/02-Advanced-RAG.md](../04-RAG/02-Advanced-RAG.md)** — RAG patterns for extracting web data
- **[20-Agent-Infrastructure/04-Agent-Evaluation-and-Testing.md](../20-Agent-Infrastructure-and-Observability/04-Agent-Evaluation-and-Testing.md)** — Testing patterns for browser agents

---

*Last updated: July 2026*
*See also: Playwright documentation, Puppeteer docs, CDP specification*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
