# Architecture Analysis: Custom Skill-Based Pydantic AI Agent

A comprehensive analysis of this library covering agentic functionality, code architecture, component details, and Python patterns.

---

## Phase 1 -- High-Level Overview (Agentic Functionality POV)

### What This Library Does

This library implements a **skill-based AI agent** using the Pydantic AI framework. It solves a fundamental problem in building AI agents: **context window waste**. Most AI agents stuff their entire knowledge base into the system prompt at startup, regardless of whether the user needs it. This library instead loads knowledge *on-demand* -- the agent learns what it *could* know, then loads full details only when actually needed.

Think of it like a doctor who has a directory of all medical specialties (small), pulls out the full textbook for cardiology only when a patient has heart symptoms (medium), and then opens the specific chapter on arrhythmias only when needed (large).

### The Core Problem It Solves

LLM context windows are finite and expensive. If you have 50 skills, each with 5,000 tokens of instructions, that's 250,000 tokens jammed into every single request -- even if the user just wants the weather. This library solves that with **progressive disclosure**, a three-level information loading hierarchy:

| Level | What Loads | When | Token Cost |
|-------|-----------|------|------------|
| **Level 1 -- Metadata** | Skill name + one-line description | Always (system prompt) | ~100 tokens/skill |
| **Level 2 -- Instructions** | Full SKILL.md body | Agent decides skill is relevant | ~500-2000 tokens |
| **Level 3 -- Resources** | Reference docs, scripts, guides | Instructions reference them | Variable |

With 50 skills, Level 1 costs only ~5,000 tokens instead of 250,000. The agent uses Level 1 metadata to *decide* which skill to activate, then loads deeper levels via tool calls.

### Agentic Patterns Used

**1. Tool-Use Pattern (ReAct-style)**
The agent reasons about what it needs, calls tools to get information, then reasons again. Specifically: the agent reads skill metadata in its system prompt, decides a skill is relevant, calls `load_skill_tool` to get instructions, follows those instructions (which may direct it to call `read_skill_file_tool` for resources), and finally responds.

- *Pro*: The agent controls what knowledge to load and when, making it adaptive.
- *Con*: Extra LLM round-trips for each tool call add latency and cost per interaction. If the agent fails to recognize that a skill is relevant, it skips loading it entirely.

**2. Dynamic System Prompt Pattern**
The system prompt is not static -- it's generated at runtime by querying the `SkillLoader` for current metadata. This means adding a new skill is as simple as dropping a folder into `skills/`. No code changes needed.

- *Pro*: Zero-code extensibility. New skills are discovered automatically.
- *Con*: The system prompt changes with every new skill, which can subtly alter agent behavior.

**3. Dependency Injection Pattern**
The agent receives its dependencies (`SkillLoader`, settings, user preferences) through a typed context object (`AgentDependencies`) rather than using global state. Pydantic AI's `RunContext` carries these through every tool call.

- *Pro*: Makes every tool function testable by injecting mock dependencies. Clean separation of concerns.
- *Con*: Requires understanding the DI pattern. Adds a layer of indirection.

**4. Toolset Composition Pattern**
Skill tools are grouped into a reusable `FunctionToolset` (`skill_toolset.py`) that can be attached to any agent. This separates tool definitions from agent configuration.

- *Pro*: Portability -- the same toolset can be reused across different agents.
- *Con*: Slight indirection; tools are defined in `skill_tools.py`, wrapped in `skill_toolset.py`, and registered on the agent in `agent.py`.

### Main Features

1. **Automatic skill discovery**: Scans a directory, parses YAML frontmatter, builds a registry
2. **Three progressive disclosure tools**: `load_skill_tool`, `read_skill_file_tool`, `list_skill_files_tool`
3. **HTTP tools**: `http_get_tool` and `http_post_tool` with retry logic for skills that call APIs
4. **Multi-provider LLM support**: OpenRouter, OpenAI, and Ollama (local)
5. **Rich streaming CLI**: Real-time token streaming with tool call visibility
6. **Evaluation framework**: YAML-defined test cases with custom evaluators using `pydantic-evals`
7. **Security**: Path traversal protection on all file-access tools

---

## Phase 2 -- Architecture Analysis

### Module Dependency Graph

```
                       ┌─────────┐
                       │  cli.py │  (Entry point)
                       └────┬────┘
                            │
                       ┌────▼────┐
                       │ agent.py│  (Agent definition + tool registration)
                       └────┬────┘
                            │
            ┌───────────────┼───────────────────┐
            │               │                   │
   ┌────────▼───────┐ ┌────▼──────────┐ ┌──────▼──────────┐
   │skill_toolset.py│ │ http_tools.py │ │   prompts.py    │
   │ (FunctionToolset│ │ (GET/POST)    │ │ (System prompt  │
   │  wrapper)       │ └───────────────┘ │  template)      │
   └────────┬───────┘                    └─────────────────┘
            │
   ┌────────▼───────┐
   │skill_tools.py  │  (Core tool logic: load_skill, read_skill_file, list_skill_files)
   └────────┬───────┘
            │
   ┌────────▼──────────┐
   │dependencies.py    │  (AgentDependencies dataclass)
   └────────┬──────────┘
            │
   ┌────────▼──────────┐     ┌──────────────┐
   │skill_loader.py    │     │ providers.py  │
   │ (Discovery +      │     │ (LLM model    │
   │  metadata parsing)│     │  factory)     │
   └────────┬──────────┘     └──────┬───────┘
            │                       │
            │                ┌──────▼───────┐
            │                │ settings.py   │
            │                │ (Pydantic     │
            │                │  Settings)    │
            └────────────────┴──────────────┘
                            │
                     ┌──────▼──────┐
                     │  .env file  │
                     └─────────────┘
```

### Architectural Patterns

**1. Layered Architecture**

The code follows a clear layered structure:
- **Presentation Layer**: `cli.py` -- handles user I/O, Rich formatting, streaming display
- **Agent Layer**: `agent.py` -- defines the Pydantic AI agent, registers tools, wires dynamic system prompt
- **Tool Layer**: `skill_tools.py`, `skill_toolset.py`, `http_tools.py` -- implements tool logic
- **Service Layer**: `skill_loader.py`, `dependencies.py` -- business logic for skill discovery and dependency management
- **Configuration Layer**: `settings.py`, `providers.py` -- environment-based configuration

Each layer depends only on layers below it.

**2. Plugin Architecture (Skills)**

Skills are a plugin system. Each skill is a self-contained directory under `skills/`:

```
skills/weather/
├── SKILL.md              # Metadata (YAML frontmatter) + instructions (body)
├── references/           # Level 3 resources
│   └── api_reference.md
└── scripts/              # Optional helper scripts
```

Adding a new skill requires zero code changes -- just create a directory with a properly formatted `SKILL.md`. The `SkillLoader` discovers it automatically at startup.

**3. Factory Pattern (Providers)**

`providers.py` uses a factory pattern to create the right LLM model based on configuration:

```python
def get_llm_model() -> Union[OpenAIChatModel, OpenRouterModel]:
    if provider == 'openrouter':
        return _create_openrouter_model(settings)
    elif provider == 'openai':
        return _create_openai_model(settings)
    elif provider == 'ollama':
        return _create_ollama_model(settings)
```

This insulates the rest of the application from provider-specific details.

---

## Phase 3 -- Detailed Component Analysis

### 3.1 `settings.py` -- Configuration Foundation

The `Settings` class uses Pydantic Settings to load configuration from environment variables and `.env` files with type validation.

```python
class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    skills_dir: Path = Field(default=Path("skills"))
    llm_provider: Literal["openrouter", "openai", "ollama"] = Field(default="openrouter")
    llm_api_key: str = Field(..., description="API key for the LLM provider")
    llm_model: str = Field(default="anthropic/claude-sonnet-4.5")
```

Key design decisions:
- `Literal["openrouter", "openai", "ollama"]` constrains the provider to known values at the type level -- invalid values fail at startup, not at runtime
- `extra="ignore"` allows unknown environment variables without crashing
- `llm_api_key: str = Field(...)` -- the `...` means this field is **required**; the app won't start without it

The `load_settings()` function wraps construction with helpful error messages:

```python
def load_settings() -> Settings:
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "llm_api_key" in str(e).lower():
            error_msg += "\nMake sure to set LLM_API_KEY in your .env file"
        raise ValueError(error_msg) from e
```

### 3.2 `skill_loader.py` -- The Discovery Engine

This is the heart of the progressive disclosure system. It has two components:

**`SkillMetadata` (Pydantic Model)**

```python
class SkillMetadata(BaseModel):
    name: str = Field(..., description="Unique skill identifier")
    description: str = Field(..., description="Brief description for agent discovery")
    version: str = Field(default="1.0.0", description="Skill version")
    author: str = Field(default="", description="Skill author")
    skill_path: Path = Field(..., description="Path to skill directory")
```

This model ensures every discovered skill has at minimum a `name`, `description`, and `skill_path`. Pydantic validates this at construction time.

**`SkillLoader` (Discovery Class)**

The `discover_skills()` method performs a filesystem scan:

```python
def discover_skills(self) -> List[SkillMetadata]:
    for skill_dir in self.skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        metadata = self._parse_skill_metadata(skill_md, skill_dir)
        if metadata:
            self.skills[metadata.name] = metadata
            discovered.append(metadata)
```

The `_parse_skill_metadata()` method implements YAML frontmatter parsing. It splits the file on `---` delimiters, extracts the YAML between the first two delimiters, and validates required fields:

```python
def _parse_skill_metadata(self, skill_md: Path, skill_dir: Path) -> Optional[SkillMetadata]:
    content = skill_md.read_text(encoding="utf-8")

    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    frontmatter_yaml = parts[1].strip()
    frontmatter = yaml.safe_load(frontmatter_yaml)
    # ... validate and return SkillMetadata
```

The `get_skill_metadata_prompt()` method generates the Level 1 content -- a compact markdown list injected into the system prompt:

```python
def get_skill_metadata_prompt(self) -> str:
    if not self.skills:
        return "No skills currently available."

    lines: List[str] = []
    for skill in self.skills.values():
        lines.append(f"- **{skill.name}**: {skill.description}")

    return "\n".join(lines)
```

For 5 skills, this produces roughly 500 tokens -- a fraction of what full instructions would consume.

### 3.3 `skill_tools.py` -- The Progressive Disclosure Implementation

This module contains three async functions that implement the three tool actions. Each follows the same pattern:

1. Validate the `skill_loader` is initialized
2. Validate the requested skill exists
3. Perform the operation with security checks
4. Return content or a descriptive error string

**`load_skill()` (Level 2)**

Reads `SKILL.md`, strips the YAML frontmatter, returns only the instruction body:

```python
async def load_skill(ctx: RunContext["AgentDependencies"], skill_name: str) -> str:
    content = skill_md.read_text(encoding="utf-8")

    # Strip YAML frontmatter - return only body
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].strip()
            return body
```

**`read_skill_file()` (Level 3)**

Reads any file within a skill's directory, with path traversal protection:

```python
# Security: Ensure file is within skill directory (prevent directory traversal)
resolved_target = target_file.resolve()
resolved_skill = skill.skill_path.resolve()

if not resolved_target.is_relative_to(resolved_skill):
    return f"Error: Access denied - file must be within skill directory"
```

This is critical security code. Without it, an LLM could be prompted to read `../../.env` or any other file on the system. The check resolves both paths to absolute form and verifies containment.

**`list_skill_files()` (Discovery)**

Lists files in a skill directory using `rglob("*")` for recursive listing. Also includes the same path traversal protection for the `directory` parameter.

### 3.4 `skill_toolset.py` -- Toolset Wrapper

This module wraps the core functions from `skill_tools.py` into a Pydantic AI `FunctionToolset`:

```python
skill_tools = FunctionToolset()

@skill_tools.tool
async def load_skill_tool(ctx: RunContext[AgentDependencies], skill_name: str) -> str:
    """Load the full instructions for a skill (Level 2 progressive disclosure)."""
    return await load_skill(ctx, skill_name)
```

Why this separation? The `FunctionToolset` is Pydantic AI's mechanism for grouping related tools into a reusable unit. The toolset can be attached to any agent via `toolsets=[skill_tools]` without modifying the agent's code. The docstrings on the wrapper functions are what the LLM sees -- they're written to guide the LLM on *when* and *how* to use each tool.

### 3.5 `agent.py` -- Wiring It All Together

This is the orchestration module. Let's trace the key pieces:

**Agent Creation**

```python
skill_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt="",  # Will be set dynamically via decorator
    toolsets=[skill_tools],  # Register skill toolset here
)
```

The agent is created with:
- A model from the factory (`get_llm_model()`)
- A typed dependency class (`AgentDependencies`)
- An empty system prompt (overridden by the decorator below)
- The skill toolset registered

**Dynamic System Prompt (Decorator)**

```python
@skill_agent.system_prompt
async def get_system_prompt(ctx: RunContext[AgentDependencies]) -> str:
    await ctx.deps.initialize()
    skill_metadata = ""
    if ctx.deps.skill_loader:
        skill_metadata = ctx.deps.skill_loader.get_skill_metadata_prompt()
    return MAIN_SYSTEM_PROMPT.format(skill_metadata=skill_metadata)
```

The `@skill_agent.system_prompt` decorator tells Pydantic AI to call this function before every agent run to generate the system prompt. It initializes dependencies (which triggers skill discovery) and injects metadata into the prompt template.

**HTTP Tool Registration**

```python
@skill_agent.tool
async def http_get_tool(ctx: RunContext[AgentDependencies], url: str) -> str:
    """Make an HTTP GET request to fetch data from a URL."""
    return await http_get(ctx, url)
```

HTTP tools are registered directly on the agent (not via the toolset) using `@skill_agent.tool`. These give the agent the ability to call external APIs -- necessary for skills like weather that fetch live data.

**Logfire Instrumentation (Observability)**

```python
if _settings.logfire_token:
    import logfire
    logfire.configure(
        token=_settings.logfire_token,
        send_to_logfire='if-token-present',
        service_name=_settings.logfire_service_name,
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)
```

When a Logfire token is configured, the agent gains full observability: every LLM call, tool invocation, and HTTP request is traced. The `instrument_pydantic_ai()` call automatically wraps agent operations, and `instrument_httpx()` traces outbound HTTP calls. This is optional -- if no token is set, the code skips instrumentation entirely.

### 3.6 `dependencies.py` -- Dependency Container

```python
@dataclass
class AgentDependencies:
    skill_loader: Optional[SkillLoader] = None
    session_id: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    settings: Optional[Any] = None

    async def initialize(self) -> None:
        if not self.settings:
            self.settings = load_settings()
        if not self.skill_loader:
            self.skill_loader = SkillLoader(...)
            self.skill_loader.discover_skills()
```

This is a lazy-initialization container. Fields start as `None` and are populated on first `initialize()` call. The `initialize()` method is **idempotent** -- calling it multiple times is safe because each section checks `if not self.X` before initializing.

The `user_preferences` dict and `session_id` field exist for future extensibility (storing per-session state like preferred units or language).

### 3.7 `http_tools.py` -- External API Communication

This module provides `http_get()` and `http_post()` functions with production-quality concerns:

**Connection Pooling via Singleton Client**

```python
_http_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client
```

A single `httpx.AsyncClient` is shared across all requests, reusing TCP connections for performance.

**Retry with Exponential Backoff (GET only)**

```python
for attempt in range(MAX_RETRIES):
    if response.status_code == 429:
        delay = RETRY_BASE_DELAY * (2 ** attempt)  # Exponential backoff
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(delay)
            continue
        else:
            return f"Error: Rate limited (HTTP 429) after {MAX_RETRIES} retries."
```

Rate limiting (HTTP 429) is retried with exponential backoff (1s, 2s, 4s). Other errors are not retried for GET, and POST has no retry logic at all (since POST requests may not be idempotent).

**Response Formatting**

JSON responses are automatically pretty-printed. Very long responses (>50,000 chars) are truncated to avoid consuming excessive context window tokens.

### 3.8 `prompts.py` -- The System Prompt Template

The system prompt is a carefully designed template with a `{skill_metadata}` placeholder:

```python
MAIN_SYSTEM_PROMPT = """You are a helpful AI assistant with access to specialized skills...

## Available Skills

{skill_metadata}

## CRITICAL: You MUST Use Skills

**MANDATORY WORKFLOW:**
When a user's request relates to ANY skill listed above, you MUST:
1. **FIRST**: Call `load_skill_tool(skill_name)` ...
```

The prompt is emphatic ("CRITICAL", "MANDATORY", "MUST") because LLMs tend to take shortcuts -- they'll try to answer from training data rather than loading the skill. The prompt includes both positive examples ("CORRECT") and negative examples ("WRONG") to reinforce the expected behavior.

### 3.9 `cli.py` -- The User Interface

The CLI uses Rich for terminal UI and implements Pydantic AI's streaming protocol:

**Streaming Architecture**

```python
async with agent.iter(user_input, message_history=message_history, deps=deps) as run:
    async for node in run:
        if Agent.is_model_request_node(node):
            async with node.stream(run.ctx) as request_stream:
                async for event in request_stream:
                    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                        console.print(delta_text, end="")
```

This is a double-nested async iteration pattern:
1. Outer loop (`async for node in run`) iterates over agent execution nodes (user prompt, model request, tool call, end)
2. Inner loop (`async for event in request_stream`) iterates over streaming tokens within a model request

The CLI displays tokens in real-time as they arrive and shows tool calls with their arguments.

**Conversation History**

```python
message_history = []
# ... after each interaction:
new_messages = run.result.new_messages()
message_history.extend(new_messages)
```

Message history is maintained as a list of Pydantic AI message objects and passed to each subsequent `agent.iter()` call, enabling multi-turn conversation.

### 3.10 `tests/evals/` -- Evaluation Framework

The evaluation system uses `pydantic-evals` with YAML-defined test cases:

```yaml
cases:
  - name: weather_basic_request
    inputs: "What's the weather in New York?"
    evaluators:
      - SkillWasLoaded:
          skill_name: weather
      - ToolWasCalled:
          tool_name: load_skill_tool
      - ResponseNotEmpty:
          min_length: 50
```

Custom evaluators in `evaluators.py` verify agentic behavior:

- `SkillWasLoaded`: Checks that the agent called `load_skill_tool` with the correct skill name -- verifying the progressive disclosure pattern is working
- `ToolWasCalled`: Checks that a specific tool was invoked
- `ResponseContains`: Checks for expected content in the response
- `ResponseNotEmpty`: Checks minimum response length

The `run_agent_task()` function in `run_evals.py` extracts tool call information from the agent's message history, making it available to evaluators:

```python
async def run_agent_task(inputs: str) -> dict[str, Any]:
    deps = AgentDependencies()
    await deps.initialize()
    result = await skill_agent.run(inputs, deps=deps)
    tool_calls = extract_tool_calls(result)
    return {
        'response': result.output,
        'tools_called': tool_calls,
    }
```

### 3.11 Data Flow: Complete Request Lifecycle

Here's the full journey of a user asking "What's the weather in New York?":

```
1. CLI receives input
   └─> cli.py: main() -> stream_agent_interaction()

2. Agent starts execution
   └─> agent.py: skill_agent.iter(user_input, deps=deps)

3. System prompt generated (Level 1)
   └─> agent.py: get_system_prompt() called
   └─> dependencies.py: deps.initialize() -> SkillLoader.discover_skills()
   └─> skill_loader.py: Scans skills/ directory, parses YAML frontmatter
   └─> System prompt now contains: "- **weather**: Get weather information..."

4. LLM receives prompt + user input, decides weather skill is relevant
   └─> LLM generates tool call: load_skill_tool(skill_name="weather")

5. Tool executes (Level 2)
   └─> skill_toolset.py: load_skill_tool() -> skill_tools.py: load_skill()
   └─> Reads skills/weather/SKILL.md, strips frontmatter, returns body
   └─> LLM now has full weather instructions with API details

6. LLM follows instructions, reads API reference (Level 3)
   └─> LLM generates: read_skill_file_tool("weather", "references/api_reference.md")
   └─> skill_tools.py: read_skill_file() -> validates path -> reads file

7. LLM calls weather API
   └─> LLM generates: http_get_tool(url="https://api.open-meteo.com/v1/forecast?...")
   └─> http_tools.py: http_get() -> httpx request -> returns JSON

8. LLM formats response
   └─> Streams tokens back through CLI

9. CLI displays streaming response
   └─> cli.py: Prints tokens in real-time, shows tool calls
```

### 3.12 Security Analysis

**Strengths:**

1. **Path traversal protection** in both `read_skill_file()` and `list_skill_files()` using `Path.resolve()` + `is_relative_to()`. Tested explicitly in the test suite.
2. **YAML safe loading** via `yaml.safe_load()` prevents arbitrary Python object deserialization.
3. **No secrets in skill files** -- API keys stay in `.env`, not in skill directories.
4. **Input validation** via Pydantic models for all data structures.

**Concerns:**

1. **Global mutable HTTP client** (`_http_client` in `http_tools.py`) -- the module-level `global` variable isn't thread-safe. In an async context this is generally fine (single event loop), but if the agent were used from multiple threads, this could cause issues. A proper async singleton or context-managed client would be safer.

2. **Unrestricted HTTP requests** -- the `http_get_tool` and `http_post_tool` accept *any* URL. A prompt injection attack could trick the agent into making requests to internal services (`http://localhost:8080/admin`, `http://169.254.169.254/` for cloud metadata). Consider implementing a URL allowlist or at minimum blocking private/internal IP ranges.

3. **No file size limits on skill resources** -- `read_skill_file()` reads the entire file into memory with `read_text()`. A maliciously large file placed in a skill directory could consume excessive memory. A size check before reading would be prudent.

4. **Skill directory as attack surface** -- since skills are loaded from the filesystem with no integrity verification, anyone with write access to the `skills/` directory can inject arbitrary instructions that the agent will follow. In a multi-tenant environment, this would need access controls or content signing.

5. **`settings` field typed as `Optional[Any]`** in `AgentDependencies` -- this loses type safety. It should be `Optional[Settings]` to maintain the type contract throughout the codebase.

---

## Phase 4 -- Python Pattern Explanations

### 4.1 Pydantic `BaseModel` and `BaseSettings` (Data Validation)

**What it is**: Pydantic models are Python classes that validate data at instantiation. `BaseModel` validates arbitrary data; `BaseSettings` extends this to load from environment variables.

**How it's used**:

```python
class SkillMetadata(BaseModel):
    name: str = Field(..., description="Unique skill identifier")
    description: str = Field(..., description="Brief description")
    version: str = Field(default="1.0.0")
    skill_path: Path = Field(...)
```

The `Field(...)` with ellipsis means the field is **required** -- Pydantic raises `ValidationError` if it's missing. `Field(default="1.0.0")` provides a default. Type annotations (`str`, `Path`) are enforced at runtime:

```python
# This would raise ValidationError at construction time:
SkillMetadata(name=123, description="test", skill_path="/tmp")
# Because name must be str, not int (though Pydantic will coerce 123 -> "123" by default)
```

For `BaseSettings`, the magic is that field names map to environment variables:

```python
class Settings(BaseSettings):
    llm_api_key: str = Field(...)  # reads from LLM_API_KEY env var
    llm_model: str = Field(default="anthropic/claude-sonnet-4.5")
```

Pydantic Settings reads `LLM_API_KEY` from the environment (case-insensitive due to `case_sensitive=False`), validates it's a string, and fails fast if it's missing.

### 4.2 `@dataclass` with Default Factories

**What it is**: Python's `@dataclass` decorator auto-generates `__init__`, `__repr__`, and other boilerplate. `field(default_factory=...)` provides mutable defaults safely.

**Why `default_factory` matters**:

```python
@dataclass
class AgentDependencies:
    user_preferences: Dict[str, Any] = field(default_factory=dict)
```

Without `default_factory`, writing `user_preferences: dict = {}` would make ALL instances share the same dict object (a classic Python gotcha). `default_factory=dict` creates a fresh `dict()` for each instance.

### 4.3 `async`/`await` and Async Context Managers

**What it is**: Python's async programming model for concurrent I/O operations.

**How it's used throughout the codebase**:

```python
# Async function definition
async def load_skill(ctx: RunContext["AgentDependencies"], skill_name: str) -> str:
    ...

# Async context manager (double nesting in CLI)
async with agent.iter(user_input, ...) as run:
    async for node in run:
        if Agent.is_model_request_node(node):
            async with node.stream(run.ctx) as request_stream:
                async for event in request_stream:
                    ...
```

The `async with` statement is an **async context manager** -- it calls `__aenter__` when entering and `__aexit__` when leaving. This pattern ensures resources (network connections, streaming sessions) are properly cleaned up even if an exception occurs.

The `async for` statement is an **async iterator** -- each iteration may involve waiting for I/O (like waiting for the next token from the LLM). This is what enables real-time streaming: the CLI prints each token as it arrives rather than waiting for the complete response.

### 4.4 Decorator-Based Registration (`@agent.system_prompt`, `@agent.tool`, `@toolset.tool`)

**What it is**: Decorators are functions that wrap other functions. Pydantic AI uses them as a registration mechanism.

**How `@skill_agent.system_prompt` works**:

```python
@skill_agent.system_prompt
async def get_system_prompt(ctx: RunContext[AgentDependencies]) -> str:
    ...
```

This decorator does NOT modify `get_system_prompt`. Instead, it registers it internally so Pydantic AI calls it before each agent run to generate the system prompt. Think of it like an event subscription: "when the agent needs a system prompt, call this function."

**How `@skill_tools.tool` works**:

```python
skill_tools = FunctionToolset()

@skill_tools.tool
async def load_skill_tool(ctx: RunContext[AgentDependencies], skill_name: str) -> str:
    """Load full instructions for a skill."""
    return await load_skill(ctx, skill_name)
```

The `@skill_tools.tool` decorator registers the function in the toolset's internal registry. The function's **docstring** is crucial -- Pydantic AI extracts it and sends it to the LLM as the tool's description. The LLM uses this description to decide when to call the tool. The **type annotations** on parameters (`skill_name: str`) are also extracted and sent to the LLM as the tool's parameter schema.

### 4.5 `TYPE_CHECKING` Guard for Import Cycles

**What it is**: A pattern to avoid circular imports while maintaining type hints.

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dependencies import AgentDependencies
```

`TYPE_CHECKING` is `False` at runtime but `True` when type checkers (mypy, pyright) analyze the code. This means:
- At runtime: `AgentDependencies` is never actually imported (avoiding circular import)
- For type checking: The import is visible, so type hints work

The corresponding type hint uses a **string literal** (forward reference):

```python
async def load_skill(ctx: RunContext["AgentDependencies"], ...) -> str:
```

The quotes around `"AgentDependencies"` tell Python "don't evaluate this now -- it's a forward reference." The type checker resolves it using the `TYPE_CHECKING` import.

### 4.6 `Optional` Types and Lazy Initialization

**What it is**: Using `Optional[X]` (equivalent to `X | None`) to represent values that may not exist yet.

```python
@dataclass
class AgentDependencies:
    skill_loader: Optional[SkillLoader] = None
    settings: Optional[Any] = None

    async def initialize(self) -> None:
        if not self.settings:
            self.settings = load_settings()
        if not self.skill_loader:
            self.skill_loader = SkillLoader(...)
```

This is the **lazy initialization** pattern. Fields start as `None` and are populated on first use. The `if not self.X` guards make `initialize()` idempotent (safe to call multiple times).

### 4.7 `Union` Return Types and Factory Functions

```python
from typing import Union

def get_llm_model() -> Union[OpenAIChatModel, OpenRouterModel]:
```

`Union[A, B]` tells the type checker this function can return either type. The factory pattern branches on configuration:

```python
if provider == 'openrouter':
    return _create_openrouter_model(settings)
elif provider == 'openai':
    return _create_openai_model(settings)
```

Functions prefixed with `_` (like `_create_openrouter_model`) are **private by convention** -- they signal "don't call this from outside this module."

### 4.8 `pathlib.Path` for Filesystem Operations

**What it is**: Python's object-oriented filesystem API, replacing string-based `os.path` operations.

```python
skill_md = skill_dir / "SKILL.md"               # Path joining with /
content = skill_md.read_text(encoding="utf-8")   # File reading
target_file.resolve()                            # Absolute path resolution
resolved_target.is_relative_to(resolved_skill)   # Containment check
target_dir.rglob("*")                            # Recursive file listing
```

The `/` operator is overloaded on `Path` objects to join path components. This is cleaner and more readable than `os.path.join()`.

The `.resolve()` method is critical for security -- it resolves symlinks and `..` components to get the true absolute path, which is then checked with `is_relative_to()` to prevent directory traversal.

### 4.9 Generator Expression with `rglob`

```python
files = []
for item in target_dir.rglob("*"):
    if item.is_file():
        rel_path = item.relative_to(skill.skill_path)
        files.append(str(rel_path))
```

`rglob("*")` is a **recursive glob** -- it yields every file and directory in the tree. This is a generator (lazy iterator), meaning it doesn't load the entire directory tree into memory at once.

### 4.10 `Literal` Types for Constrained Values

```python
llm_provider: Literal["openrouter", "openai", "ollama"] = Field(...)
```

`Literal` restricts a field to specific string values. Pydantic validates this at construction time. If someone sets `LLM_PROVIDER=azure` in their `.env`, they get a validation error immediately rather than a confusing runtime failure later. This is a form of **making illegal states unrepresentable**.

### 4.11 Module-Level Initialization (Top-Level Side Effects)

```python
# In agent.py
_settings = load_settings()

if _settings.logfire_token:
    import logfire
    logfire.configure(...)
    logfire.instrument_pydantic_ai()
```

This code runs when the module is first imported. It's a deliberate design choice -- instrumentation must happen before the agent processes any requests. The conditional import (`import logfire` inside the `if` block) means the `logfire` dependency is only needed if the feature is enabled.

**Caveat**: This makes the module import have side effects (configuring observability, reading environment variables). This can complicate testing -- importing `src.agent` in a test immediately triggers settings loading and potentially Logfire configuration. The tests work around this by using mock objects rather than importing and running the real agent.

### 4.12 The `isinstance` Type Narrowing Pattern

```python
if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
    initial_text = event.part.content
elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
    delta_text = event.delta.content_delta
```

This pattern handles polymorphic event streams. The streaming API yields different event types (`PartStartEvent`, `PartDeltaEvent`, etc.), and `isinstance` checks determine which type we're dealing with. After the `isinstance` check, the type checker knows the specific type and allows access to type-specific attributes (`.content`, `.content_delta`).

### 4.13 Exception Chaining with `raise ... from`

```python
raise ValueError(error_msg) from e
```

The `from e` clause chains the new `ValueError` to the original exception. When the error is displayed, Python shows both exceptions:

```
ValueError: Failed to load settings: ...
The above exception was the direct cause of the following exception:
pydantic.ValidationError: ...
```

This preserves the full error context for debugging while presenting a cleaner user-facing message.

---

This analysis covers the complete codebase -- from the high-level agentic design philosophy through the architecture, into every component's internals, and down to the Python-level patterns that make it work. The progressive disclosure pattern at the heart of this library is a genuinely useful architectural innovation for building scalable AI agents, and the implementation demonstrates clean separation of concerns with Pydantic AI's dependency injection and tooling system.

---

## Phase 5 -- Architectural Alternatives: Skill System Comparison

This section compares the Pydantic + YAML/Markdown approach used in this project against other architectural options for building skill/plugin systems in AI agents.

### 5.1 Option 1: Pydantic Models + YAML/Markdown Frontmatter (This Project)

**How it works**: Skills are filesystem directories with structured markdown files. YAML frontmatter provides typed metadata, parsed into Pydantic models at runtime.

```python
class SkillMetadata(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    version: str = Field(default="1.0.0")
    skill_path: Path = Field(...)
```

**Used by**: Claude Skills (Anthropic), this project, many documentation-first systems.

**Pros**:
- Human-readable/editable skill definitions
- Git-friendly (markdown diffs are clean)
- Type validation at load time catches errors early
- No code required to add a skill -- just markdown
- Pydantic provides excellent error messages when validation fails

**Cons**:
- Runtime parsing overhead on startup
- YAML frontmatter parsing is fragile (split on `---`)
- Two-language system (YAML + Python) can confuse contributors
- Skills can't have complex initialization logic

**Debugging/Evals**:
- **Debugging**: Excellent. Pydantic's `ValidationError` tells you exactly which field failed and why. Structured logging (as in `skill_loader.py`) traces discovery.
- **Evals**: Easy to create synthetic skills in temp directories for testing (see `tests/test_skill_loader.py`).

---

### 5.2 Option 2: Python Decorators + Class-Based Skills

**How it works**: Skills are Python classes or functions decorated with registration metadata. The decorator registers them in a global or injected registry.

```python
@skill(name="weather", description="Get weather information")
class WeatherSkill:
    def execute(self, location: str) -> str:
        ...
```

**Used by**: LangChain Tools, Semantic Kernel Skills, CrewAI Tools, many Python agent frameworks.

**Pros**:
- Full Python power -- skills can have complex initialization, dependencies, async setup
- Type hints on methods become tool schemas automatically
- IDE autocomplete and static analysis work fully
- Single language (Python only)
- Can leverage Python's import system for discovery

**Cons**:
- Adding a skill requires writing Python code (higher barrier)
- Skills are code, not content -- harder for non-developers to contribute
- Import-time side effects can cause issues (circular imports, slow startup)
- Harder to version skill "instructions" separately from skill "code"

**Debugging/Evals**:
- **Debugging**: Excellent. Standard Python debugging (breakpoints, stack traces) works. Static type checkers catch errors before runtime.
- **Evals**: Can unit test skills directly as Python classes. However, testing the "instruction following" aspect requires mocking the LLM, which is harder than testing against static markdown.

---

### 5.3 Option 3: JSON Schema / OpenAPI Specification

**How it works**: Skills are defined as JSON Schema documents describing their interface. Often used when skills are actually external APIs.

```json
{
  "name": "weather",
  "description": "Get weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"}
    },
    "required": ["location"]
  }
}
```

**Used by**: OpenAI Function Calling, many API-first agent systems, AutoGPT plugins.

**Pros**:
- Language-agnostic (skills can be implemented in any language)
- Direct mapping to LLM function calling schemas
- Can validate inputs/outputs against schema
- Well-understood standard with tooling ecosystem

**Cons**:
- Verbose compared to Python type hints
- No place for rich instructions (just schema, not guidance)
- Two sources of truth if you also have Python implementation
- JSON Schema is powerful but complex (allOf, oneOf, etc.)

**Debugging/Evals**:
- **Debugging**: Moderate. Schema validation errors can be cryptic. No line numbers in JSON errors.
- **Evals**: Easy to programmatically generate test cases from schema. Can use property-based testing (generate random valid inputs from schema).

---

### 5.4 Option 4: DSL (Domain-Specific Language) Based

**How it works**: A custom language or configuration format specifically for defining agent skills.

```
skill weather {
  description: "Get weather information"
  input location: string
  output: WeatherReport
  
  instructions {
    1. Look up coordinates for {location}
    2. Call Open-Meteo API
    3. Format response
  }
}
```

**Used by**: Some enterprise agent platforms, internal tools at large companies.

**Pros**:
- Tailored exactly to the domain
- Can enforce constraints impossible in general-purpose languages
- Can generate multiple outputs (docs, schemas, code) from one source

**Cons**:
- Requires building/maintaining a parser
- Learning curve for contributors
- Limited tooling (no IDE support unless you build it)
- Hard to extend when requirements change

**Debugging/Evals**:
- **Debugging**: Depends entirely on DSL implementation quality. Often poor error messages.
- **Evals**: Requires custom tooling to parse and validate DSL.

---

### 5.5 Option 5: Database/Registry-Based

**How it works**: Skills are stored in a database with structured fields. Admin UI or API for management.

```sql
CREATE TABLE skills (
  id UUID PRIMARY KEY,
  name VARCHAR(100),
  description TEXT,
  instructions TEXT,
  version INTEGER,
  created_at TIMESTAMP
);
```

**Used by**: Enterprise agent platforms, multi-tenant SaaS products, systems needing access control.

**Pros**:
- Dynamic updates without deployment
- Built-in versioning, audit trails
- Access control per skill
- Can query/filter skills (e.g., "all skills tagged 'finance'")

**Cons**:
- Requires database infrastructure
- Skills aren't in version control (or require sync mechanism)
- Harder to develop locally
- Migration complexity when schema changes

**Debugging/Evals**:
- **Debugging**: Requires database query tools. Harder to trace "why did this skill load this way?"
- **Evals**: Can snapshot database state for reproducible tests, but adds infrastructure complexity.

---

### 5.6 Option 6: Hybrid (Code + Content Separation)

**How it works**: Skill *logic* is Python code, but skill *instructions* are separate content files. The code defines the interface; the content defines how the LLM should use it.

```python
# weather_skill.py
@skill
class WeatherSkill:
    """Instructions loaded from SKILL.md"""
    
    async def get_weather(self, location: str) -> WeatherData:
        # Implementation
        ...
```

```markdown
# SKILL.md
Instructions for using WeatherSkill...
```

**Used by**: This project is partially this pattern (instructions in markdown, tools in Python). Some LangChain implementations.

**Pros**:
- Best of both worlds: type-safe code + editable instructions
- Instructions can be updated without code changes
- Clear separation of "what can the skill do" vs "how should the LLM use it"

**Cons**:
- Two files to keep in sync
- More complex mental model
- Need conventions for linking code to content

**Debugging/Evals**:
- **Debugging**: Good. Code errors are Python errors. Content errors are validation errors.
- **Evals**: Can test code logic separately from instruction following. Can A/B test different instructions with same code.

---

### 5.7 Comparison Matrix: Debugging & Evals Focus

| Approach | Error Messages | Stack Traces | Static Analysis | Unit Testing | Integration Testing | Eval Reproducibility |
|----------|---------------|--------------|-----------------|--------------|--------------------|--------------------|
| **Pydantic + YAML** | Excellent | Good (Python) | Partial (YAML not checked) | Easy (temp dirs) | Easy | Excellent (files are deterministic) |
| **Python Decorators** | Excellent | Excellent | Excellent | Excellent | Medium (mock LLM) | Good |
| **JSON Schema** | Moderate | N/A | External tools | Medium | Medium | Excellent |
| **DSL** | Varies | Varies | Requires custom | Requires custom | Requires custom | Varies |
| **Database** | Moderate | Good | None | Requires fixtures | Complex | Requires snapshots |
| **Hybrid** | Excellent | Excellent | Good | Excellent | Good | Excellent |

---

### 5.8 Why This Project Uses Pydantic + YAML/Markdown

Based on the codebase, the design priorities are:

1. **Zero-code skill creation**: Anyone can add a skill by creating a folder with `SKILL.md`. This is the Claude Skills pattern -- accessible to non-programmers.

2. **Progressive disclosure of instructions**: The content is meant to be read by the LLM, not executed as code. Markdown is a natural fit for LLM-consumable content.

3. **Type safety at the boundary**: Pydantic validates the *metadata* (name, description, version) but not the *instructions*. This catches structural errors while allowing free-form content.

4. **Testability**: The test suite creates temporary skill directories, writes markdown files, and verifies the loader works correctly. This is much simpler than mocking a database or parsing DSL.

---

### 5.9 Recommendations Based on Use Case

| If You Need... | Best Approach |
|----------------|---------------|
| Non-developers creating skills | Pydantic + YAML/Markdown |
| Maximum type safety | Python Decorators |
| Skills as external services | JSON Schema / OpenAPI |
| Enterprise governance | Database with audit trail |
| Rapid iteration on instructions | Hybrid (code + content) |
| Simple agent with few skills | Python Decorators (least overhead) |

For **debugging**, Python-native approaches (decorators, Pydantic) win because you get real stack traces, IDE support, and static analysis. 

For **evals**, file-based approaches (this project's style) win because you can create deterministic test fixtures without mocking infrastructure, and you can version-control your eval datasets alongside skills.
