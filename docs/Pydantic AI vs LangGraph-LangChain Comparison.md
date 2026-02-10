# Pydantic AI vs LangGraph/LangChain for Complex Skills Development

A comprehensive comparison for planning complex skills development that requires robust evaluation, debugging, and production deployment.

---

## Executive Summary

| Dimension | Pydantic AI | LangGraph/LangChain |
|-----------|-------------|---------------------|
| **Philosophy** | Type-safe, minimal, Python-native | Feature-rich, abstraction-heavy, ecosystem-first |
| **Learning Curve** | Low (if you know Pydantic) | Medium-High (many concepts) |
| **Debugging** | Excellent (Python-native) | Moderate (abstraction layers) |
| **Evals** | Native `pydantic-evals` library | LangSmith (paid) or custom |
| **Traceability** | Logfire integration | LangSmith integration |
| **Deployment** | Standard Python deployment | LangServe or standard Python |
| **Best For** | Type-safe agents, clean architecture | Complex workflows, multi-agent orchestration |

---

## 1. Development Experience

### 1.1 Pydantic AI

**Philosophy**: "Python-first with type safety." The library is designed to feel like writing normal Python with Pydantic models.

**Tool Definition**:
```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

class WeatherResult(BaseModel):
    temperature: float
    conditions: str
    location: str

agent = Agent(model, deps_type=AgentDependencies, result_type=WeatherResult)

@agent.tool
async def get_weather(ctx: RunContext[AgentDependencies], location: str) -> str:
    """Get weather for a location."""
    # Implementation
    return weather_data
```

**Strengths**:
- **Type inference**: Return types and parameter types are automatically extracted and sent to the LLM as tool schemas
- **Dependency injection**: `RunContext[T]` carries typed dependencies through every tool call -- testable and explicit
- **Minimal abstraction**: What you write is close to what executes -- no hidden layers
- **Pydantic validation**: Input/output validation is automatic via Pydantic models
- **Result types**: Can enforce structured output with `result_type=YourModel`

**Weaknesses**:
- **Fewer built-in integrations**: No pre-built tools for common services (you write them)
- **Less mature ecosystem**: Smaller community, fewer examples
- **No visual workflow builder**: Code-only development

**Code Patterns in This Project**:
```python
# Toolset composition (skill_toolset.py)
skill_tools = FunctionToolset()

@skill_tools.tool
async def load_skill_tool(ctx: RunContext[AgentDependencies], skill_name: str) -> str:
    """Load skill instructions."""
    return await load_skill(ctx, skill_name)

# Register toolset on agent
agent = Agent(model, deps_type=AgentDependencies, toolsets=[skill_tools])
```

---

### 1.2 LangGraph/LangChain

**Philosophy**: "Batteries included with composable abstractions." LangChain provides pre-built components; LangGraph adds stateful graph-based orchestration.

**Tool Definition**:
```python
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

class WeatherInput(BaseModel):
    location: str = Field(description="The city to get weather for")

@tool(args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """Get weather for a location."""
    # Implementation
    return weather_data
```

**LangGraph Workflow**:
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    current_skill: str | None

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", execute_tools)
workflow.add_edge("agent", "tools")
workflow.add_conditional_edges("tools", should_continue, {"continue": "agent", "end": END})

app = workflow.compile()
```

**Strengths**:
- **Rich ecosystem**: Pre-built tools for databases, APIs, document loaders, vector stores
- **LangGraph for complex flows**: Explicit state machines, cycles, parallel execution, human-in-the-loop
- **Community and examples**: Large community, extensive documentation, many tutorials
- **Visual debugging**: LangSmith shows execution graphs visually
- **Checkpointing**: LangGraph supports persistent state and resumption

**Weaknesses**:
- **Abstraction complexity**: Multiple abstraction layers can obscure what's happening
- **Version churn**: Frequent breaking changes between versions
- **Two Pydantic versions**: LangChain uses `pydantic_v1` internally, causing conflicts with Pydantic v2
- **Verbose for simple cases**: Simple agents require significant boilerplate

---

## 2. Debugging

### 2.1 Pydantic AI Debugging

**Stack Traces**: Clean Python stack traces. When something fails, you see exactly where in your code it happened.

```
Traceback (most recent call last):
  File "src/skill_tools.py", line 45, in load_skill
    content = skill_md.read_text(encoding="utf-8")
FileNotFoundError: [Errno 2] No such file or directory: 'skills/weather/SKILL.md'
```

**Type Errors**: Caught at development time by mypy/pyright, or at runtime by Pydantic.

```python
# This fails at runtime with clear error:
SkillMetadata(name=123, description="test", skill_path="/tmp")
# ValidationError: name - Input should be a valid string
```

**Logging**: Standard Python logging. This project uses structured logging:
```python
logger.info(f"skill_discovered: name={metadata.name}, version={metadata.version}")
```

**Breakpoints**: Standard Python debugger (pdb, IDE debuggers) works perfectly. You can step through tool execution, inspect `RunContext`, examine dependencies.

**Logfire Integration**: Optional observability with Pydantic's Logfire:
```python
import logfire
logfire.instrument_pydantic_ai()  # Auto-traces all agent operations
logfire.instrument_httpx()         # Traces HTTP calls
```

---

### 2.2 LangGraph/LangChain Debugging

**Stack Traces**: Often obscured by abstraction layers. A simple tool error may show 10+ frames of LangChain internals before your code.

```
Traceback (most recent call last):
  File "langchain_core/runnables/base.py", line 1234, in invoke
    ...
  File "langchain_core/runnables/config.py", line 567, in run_in_executor
    ...
  File "langchain/agents/agent.py", line 890, in _take_next_step
    ...
  File "your_code.py", line 45, in get_weather
    raise ValueError("API error")
ValueError: API error
```

**Type Errors**: Mixed. LangChain uses Pydantic v1 internally, which has different error messages and behaviors than v2. Type checking is less strict.

**Verbose Mode**: LangChain has a verbose flag:
```python
from langchain.globals import set_verbose
set_verbose(True)  # Prints all chain steps
```

**LangSmith**: The primary debugging tool. Paid service that provides:
- Visual execution traces
- Token usage per step
- Latency breakdowns
- Error highlighting

**Callbacks**: LangChain's callback system for custom logging:
```python
from langchain.callbacks import StdOutCallbackHandler
agent.invoke({"input": "..."}, callbacks=[StdOutCallbackHandler()])
```

---

### 2.3 Debugging Comparison

| Aspect | Pydantic AI | LangGraph/LangChain |
|--------|-------------|---------------------|
| Stack trace clarity | Excellent | Poor (many abstraction layers) |
| Breakpoint debugging | Excellent | Moderate (hard to step through chains) |
| Type error messages | Excellent (Pydantic v2) | Moderate (Pydantic v1) |
| Built-in logging | Standard Python | Callbacks + verbose mode |
| Observability | Logfire (optional) | LangSmith (recommended, paid) |
| IDE support | Excellent | Moderate (dynamic types) |

---

## 3. Evaluation (Evals)

### 3.1 Pydantic AI Evals

**Native Library**: `pydantic-evals` provides a YAML-based evaluation framework.

**Eval Definition** (from this project):
```yaml
# tests/evals/skill_loading.yaml
name: skill_loading_verification

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

**Custom Evaluators**:
```python
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

@dataclass
class SkillWasLoaded(Evaluator):
    skill_name: str

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        if isinstance(output, dict) and 'tools_called' in output:
            for tool_name, args in output['tools_called']:
                if tool_name == 'load_skill_tool' and args.get('skill_name') == self.skill_name:
                    return EvaluationReason(value=True, reason=f'Loaded "{self.skill_name}"')
        return EvaluationReason(value=False, reason='Skill not loaded')
```

**Running Evals**:
```python
from pydantic_evals import Dataset

dataset = Dataset.from_file('tests/evals/skill_loading.yaml', custom_evaluator_types=[SkillWasLoaded])
report = await dataset.evaluate(run_agent_task)
```

**Strengths**:
- YAML-based test cases (version controllable, human-readable)
- Custom evaluators are Python classes (type-safe, testable)
- No external service required
- Deterministic and reproducible

**Weaknesses**:
- Less mature than LangSmith
- No visual dashboard (CLI output only)
- Manual setup for complex evaluators

---

### 3.2 LangGraph/LangChain Evals

**LangSmith Evals**: Cloud-based evaluation platform (paid for production use).

**Eval Definition**:
```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Create dataset
dataset = client.create_dataset("weather_skill_tests")
client.create_examples(
    inputs=[{"input": "What's the weather in NYC?"}],
    outputs=[{"expected_skill": "weather"}],
    dataset_id=dataset.id
)

# Define evaluator
def skill_loaded_evaluator(run, example):
    # Check if weather skill was loaded
    for step in run.child_runs:
        if step.name == "load_skill" and "weather" in str(step.inputs):
            return {"score": 1, "key": "skill_loaded"}
    return {"score": 0, "key": "skill_loaded"}

# Run evaluation
results = evaluate(
    agent.invoke,
    data=dataset,
    evaluators=[skill_loaded_evaluator]
)
```

**LangSmith Features**:
- Visual dashboard for eval results
- Comparison across runs
- Annotation queue for human evaluation
- A/B testing of prompts
- Regression detection

**Open-Source Alternatives**:
- **Ragas**: Eval framework for RAG systems
- **DeepEval**: Open-source LLM evaluation
- **Custom pytest**: Roll your own with mocking

**Strengths**:
- Professional dashboard and analytics
- Team collaboration features
- Historical tracking and regression detection
- Human-in-the-loop annotation

**Weaknesses**:
- Paid service for production use
- Vendor lock-in to LangSmith
- Requires network connectivity
- Less control over eval logic

---

### 3.3 Evals Comparison

| Aspect | Pydantic AI (`pydantic-evals`) | LangSmith |
|--------|-------------------------------|-----------|
| Cost | Free (open source) | Free tier limited, paid for production |
| Setup complexity | Low (YAML + Python) | Medium (API keys, SDK setup) |
| Offline/CI use | Excellent | Requires network |
| Visual dashboard | No (CLI only) | Yes (web UI) |
| Custom evaluators | Python classes | Python functions |
| Team collaboration | Git-based | Built-in UI |
| Version control | Native (files in repo) | Separate from code |
| Determinism | High | Depends on LLM consistency |

---

## 4. Traceability and Observability

### 4.1 Pydantic AI Traceability

**Logfire**: Pydantic's observability platform (optional, has free tier).

```python
import logfire

logfire.configure(token="...", service_name="skill-agent")
logfire.instrument_pydantic_ai()  # Traces agent runs
logfire.instrument_httpx()         # Traces HTTP calls
```

**What's Traced**:
- Every LLM call (input, output, tokens, latency)
- Tool invocations (name, arguments, return value)
- HTTP requests (URL, status, timing)
- Custom spans you add

**Structured Logging**: This project uses structured logging for traceability:
```python
logger.info(f"load_skill_success: skill_name={skill_name}, body_length={len(body)}")
logger.warning(f"load_skill_not_found: skill_name={skill_name}, available={available}")
```

**Manual Spans**:
```python
with logfire.span("custom_operation"):
    # Your code here
    logfire.info("step_completed", step="validation")
```

---

### 4.2 LangGraph/LangChain Traceability

**LangSmith**: The default observability solution.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "..."
os.environ["LANGCHAIN_PROJECT"] = "skill-agent"

# All LangChain operations are now traced automatically
```

**What's Traced**:
- Chain/graph execution flow
- Each node execution (input, output, timing)
- Token usage and costs
- Error locations
- Nested runs (parent-child relationships)

**Visual Graph**: LangSmith shows LangGraph execution as a visual graph with:
- Node colors (success/failure)
- Edge traversal order
- State at each step

**Callbacks for Custom Tracing**:
```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomTracer(BaseCallbackHandler):
    def on_tool_start(self, tool, input_str, **kwargs):
        print(f"Tool started: {tool.name}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"Tool finished: {output}")
```

---

### 4.3 Traceability Comparison

| Aspect | Pydantic AI (Logfire) | LangChain (LangSmith) |
|--------|----------------------|----------------------|
| Auto-instrumentation | Yes (one line) | Yes (env vars) |
| Visual traces | Yes (web UI) | Yes (web UI) |
| Graph visualization | No | Yes (LangGraph) |
| Cost attribution | Yes (token counts) | Yes (with cost tracking) |
| Self-hosted option | No (cloud only) | Enterprise only |
| Free tier | Yes (generous) | Yes (limited) |
| OpenTelemetry export | Yes | Yes |

---

## 5. Deployment

### 5.1 Pydantic AI Deployment

**Standard Python Deployment**: No special requirements. Deploy as any Python application.

**Options**:
- **Docker**: Standard Python container
- **Cloud Functions**: AWS Lambda, GCP Cloud Functions, Azure Functions
- **Kubernetes**: Standard deployment
- **Serverless**: Works with any ASGI framework (FastAPI, Starlette)

**Example FastAPI Integration**:
```python
from fastapi import FastAPI
from pydantic_ai import Agent

app = FastAPI()
agent = Agent(model, deps_type=AgentDependencies)

@app.post("/chat")
async def chat(message: str):
    deps = AgentDependencies()
    await deps.initialize()
    result = await agent.run(message, deps=deps)
    return {"response": result.output}
```

**Scaling Considerations**:
- Agents are stateless (dependencies created per request)
- HTTP client pooling handled in `http_tools.py`
- No special infrastructure required

---

### 5.2 LangGraph/LangChain Deployment

**LangServe**: Official deployment framework for LangChain.

```python
from langserve import add_routes
from fastapi import FastAPI

app = FastAPI()
add_routes(app, agent, path="/agent")

# Provides:
# POST /agent/invoke
# POST /agent/batch
# POST /agent/stream
# GET /agent/playground (interactive UI)
```

**LangGraph Cloud**: Managed hosting for LangGraph applications (coming soon / in preview).

**Standard Deployment**: Also works with standard Python deployment.

**Checkpointing for Long-Running Workflows**:
```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory)

# Resume from checkpoint
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(state, config)
```

**Scaling Considerations**:
- LangGraph state can be persisted (SQLite, PostgreSQL, Redis)
- Human-in-the-loop requires external storage
- More infrastructure for stateful workflows

---

### 5.3 Deployment Comparison

| Aspect | Pydantic AI | LangGraph/LangChain |
|--------|-------------|---------------------|
| Deployment complexity | Low (standard Python) | Medium (LangServe adds features) |
| Stateful workflows | Manual implementation | Built-in checkpointing |
| Serverless friendly | Excellent | Good (watch cold start times) |
| Built-in playground | No | Yes (LangServe) |
| Managed hosting | No | LangGraph Cloud (preview) |
| API generation | Manual (FastAPI) | Automatic (LangServe) |

---

## 6. Complex Skill Development Patterns

### 6.1 Multi-Step Workflows

**Pydantic AI Approach**: Tools call other tools or maintain state in dependencies.

```python
@dataclass
class SkillExecutionState:
    current_step: int = 0
    loaded_resources: list[str] = field(default_factory=list)
    intermediate_results: dict = field(default_factory=dict)

@dataclass
class AgentDependencies:
    skill_loader: SkillLoader
    execution_state: SkillExecutionState = field(default_factory=SkillExecutionState)

@agent.tool
async def execute_skill_step(ctx: RunContext[AgentDependencies], step: int, data: str) -> str:
    state = ctx.deps.execution_state
    state.current_step = step
    # Process step...
    state.intermediate_results[step] = result
    return result
```

**LangGraph Approach**: Explicit state machine with typed state.

```python
class SkillState(TypedDict):
    messages: list
    current_step: int
    loaded_resources: list[str]
    intermediate_results: dict

def step_1(state: SkillState) -> SkillState:
    # Process step 1
    return {**state, "current_step": 2, "intermediate_results": {...}}

def step_2(state: SkillState) -> SkillState:
    # Process step 2
    return {**state, "current_step": 3}

workflow = StateGraph(SkillState)
workflow.add_node("step_1", step_1)
workflow.add_node("step_2", step_2)
workflow.add_edge("step_1", "step_2")
```

**Comparison**:
- Pydantic AI: State is implicit in dependencies, flow controlled by LLM
- LangGraph: State is explicit, flow defined by graph edges

---

### 6.2 Resource Loading (Progressive Disclosure)

**Pydantic AI Approach** (this project):
```python
@agent.tool
async def load_skill(ctx: RunContext[AgentDependencies], skill_name: str) -> str:
    """Level 2: Load full instructions."""
    return skill_content

@agent.tool  
async def read_skill_file(ctx: RunContext[AgentDependencies], skill_name: str, file_path: str) -> str:
    """Level 3: Load specific resource."""
    return file_content
```

**LangGraph Approach**:
```python
def should_load_resource(state: SkillState) -> str:
    if "need_api_docs" in state["messages"][-1]:
        return "load_resource"
    return "continue"

workflow.add_conditional_edges("agent", should_load_resource, {
    "load_resource": "resource_loader",
    "continue": "respond"
})
```

**Comparison**:
- Pydantic AI: LLM decides when to load resources via tool calls
- LangGraph: Graph structure can encode when resources should load

---

### 6.3 Error Handling and Recovery

**Pydantic AI Approach**:
```python
@agent.tool
async def safe_api_call(ctx: RunContext[AgentDependencies], url: str) -> str:
    try:
        response = await http_get(ctx, url)
        return response
    except Exception as e:
        logger.error(f"api_call_failed: url={url}, error={e}")
        return f"Error: {e}. Please try a different approach."
```

**LangGraph Approach**:
```python
def handle_error(state: SkillState) -> SkillState:
    error = state.get("last_error")
    return {**state, "messages": state["messages"] + [f"Error occurred: {error}"]}

workflow.add_node("error_handler", handle_error)
workflow.add_conditional_edges("api_call", lambda s: "error" if s.get("last_error") else "success", {
    "error": "error_handler",
    "success": "continue"
})
```

**Comparison**:
- Pydantic AI: Error handling is Python exception handling
- LangGraph: Errors can be graph nodes with explicit recovery paths

---

## 7. Recommendations by Use Case

### When to Choose Pydantic AI

1. **Type safety is critical**: You want compile-time/load-time validation of all data structures
2. **Simple to moderate complexity**: Linear or LLM-directed workflows
3. **Team knows Pydantic**: Leverage existing knowledge
4. **Cost-sensitive**: No paid observability requirements
5. **Clean architecture priority**: Minimal abstraction, maximal transparency
6. **Fast iteration**: Simple tools, quick debugging
7. **This project's pattern**: Progressive disclosure, content-based skills

### When to Choose LangGraph/LangChain

1. **Complex orchestration**: Cycles, parallel execution, human-in-the-loop
2. **Pre-built integrations needed**: Vector stores, document loaders, API connectors
3. **Visual debugging required**: LangSmith's trace visualization is essential
4. **Stateful long-running workflows**: Checkpointing and resumption
5. **Team already using LangChain**: Don't switch mid-project
6. **Enterprise features**: LangSmith's team collaboration, annotation queues

### Hybrid Approach

It's possible to use both:
- **Pydantic AI for core agent logic**: Type-safe tools, clean architecture
- **LangChain for integrations**: Use LangChain's document loaders, vector stores as utilities
- **Custom evals**: Use `pydantic-evals` for CI, LangSmith for production monitoring

```python
# Use LangChain's document loader with Pydantic AI agent
from langchain_community.document_loaders import WebBaseLoader

@agent.tool
async def load_webpage(ctx: RunContext[AgentDependencies], url: str) -> str:
    """Load and parse a webpage."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0].page_content
```

---

## 8. Migration Considerations

### From LangChain to Pydantic AI

**Effort**: Medium. Main work is rewriting tools and removing abstractions.

**Steps**:
1. Convert `@tool` decorators to `@agent.tool`
2. Replace chain/graph logic with LLM-directed flow or Python control flow
3. Replace LangSmith with Logfire or custom logging
4. Rewrite evals using `pydantic-evals`

### From Pydantic AI to LangGraph

**Effort**: Medium-High. Need to add abstraction layers.

**Steps**:
1. Define state schema as `TypedDict`
2. Convert tools to LangChain tools
3. Build graph structure for workflow
4. Set up LangSmith for observability
5. Migrate evals to LangSmith datasets

---

## 9. Summary Comparison Table

| Dimension | Pydantic AI | LangGraph/LangChain |
|-----------|-------------|---------------------|
| **Philosophy** | Minimal, type-safe, Python-native | Feature-rich, ecosystem-first |
| **Learning curve** | Low | Medium-High |
| **Type safety** | Excellent (Pydantic v2) | Moderate (Pydantic v1) |
| **Debugging** | Excellent (clean traces) | Moderate (abstraction layers) |
| **Evals** | `pydantic-evals` (free, YAML) | LangSmith (paid, visual) |
| **Traceability** | Logfire (optional) | LangSmith (recommended) |
| **Complex workflows** | LLM-directed | Graph-based state machine |
| **Pre-built tools** | Few | Many |
| **Deployment** | Standard Python | LangServe + options |
| **Community size** | Small but growing | Large |
| **Stability** | Good | Version churn |
| **Best for** | Clean, typed agents | Complex orchestration |

---

## 10. Conclusion

**Choose Pydantic AI** when you value type safety, clean architecture, and transparent debugging over ecosystem features. It's ideal for this project's pattern of progressive disclosure skills where the LLM decides what to load.

**Choose LangGraph/LangChain** when you need complex stateful workflows, extensive pre-built integrations, or visual debugging through LangSmith. The ecosystem benefits outweigh the abstraction costs for large, complex systems.

For **complex skills development with evals**, both frameworks are viable. Pydantic AI offers simpler, more deterministic testing with `pydantic-evals`. LangChain/LangSmith offers more sophisticated analytics and team collaboration features at the cost of vendor dependency and complexity.
