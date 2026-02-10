# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Framework-agnostic skill system for AI agents implementing **progressive disclosure** - a pattern for managing context efficiently. Skills load instructions in three levels:
- **Level 1**: Metadata (~100 tokens/skill) in system prompt
- **Level 2**: Full SKILL.md instructions loaded via `load_skill_tool`
- **Level 3**: Reference files loaded via `read_skill_file_tool`

**Key Innovation**: Unlike Claude Skills (vendor-locked), this makes skill capabilities available to ANY AI framework via Pydantic AI.

## Build & Development Commands

```bash
# Setup
uv sync                              # Install all dependencies
cp .env.example .env                 # Configure environment

# Run the agent
uv run python -m src.cli

# Tests
uv run pytest tests/ -v                          # All tests
uv run pytest tests/test_skill_loader.py -v      # Skill loader tests
uv run pytest tests/test_agent.py -v             # Agent integration tests (71 tests)
uv run pytest tests/test_skill_loader.py::TestSkillLoader::test_skill_loader_discovers_skills -v  # Single test

# Evaluations (25 test cases)
uv run python -m tests.evals.run_evals                        # All evals
uv run python -m tests.evals.run_evals --dataset skill_loading  # Specific dataset
uv run python -m tests.evals.run_evals --verbose              # With reasons

# Validation scripts
uv run python -m scripts.validate_skills         # Validate skill structure
uv run python -m scripts.test_agent              # Interactive testing
uv run python -m scripts.run_full_validation     # Full validation pipeline

# Linting & type checking (configured in pyproject.toml)
uv run ruff check src/
uv run mypy src/
```

## Architecture

### Core Components

**`src/agent.py`** - Main Pydantic AI agent with dynamic system prompt and Logfire integration
- Uses `@skill_agent.system_prompt` decorator to inject Level 1 skill metadata
- Registers skill toolset and HTTP tools

**`src/skill_loader.py`** - Skill discovery and metadata parsing
- `SkillLoader.discover_skills()` scans `skills/` directory
- `SkillLoader.get_skill_metadata_prompt()` generates Level 1 system prompt section
- `SkillMetadata` Pydantic model for type-safe frontmatter

**`src/skill_toolset.py`** - FunctionToolset for reusable skill tools
- Wraps `skill_tools.py` functions as Pydantic AI tools
- Registered with agent via `toolsets=[skill_tools]`

**`src/skill_tools.py`** - Progressive disclosure implementations
- `load_skill()` - Level 2: Returns SKILL.md body (frontmatter stripped)
- `read_skill_file()` - Level 3: Returns referenced files with directory traversal protection
- `list_skill_files()` - Lists available resources in a skill

**`src/providers.py`** - Multi-provider LLM support (OpenRouter, OpenAI, Ollama)

**`src/cli.py`** - Rich-based CLI with streaming via `agent.iter()`

### Skill Structure

```
skills/skill-name/
├── SKILL.md              # REQUIRED: YAML frontmatter + instructions
├── references/           # Optional: API docs, guides (Level 3)
└── scripts/              # Optional: Helper scripts
```

**SKILL.md format:**
```markdown
---
name: skill-name
description: Brief description for agent discovery (1-2 sentences)
version: 1.0.0
author: Your Name
---

# Skill Name
[Instructions, operations, examples...]
```

### Current Skills (5)

- **weather** - Open-Meteo API, 40+ city coordinates
- **research_assistant** - Semantic Scholar API, 214M+ papers
- **recipe_finder** - TheMealDB + Spoonacular APIs
- **world_clock** - WorldTimeAPI, 40+ timezone mappings
- **code_review** - Multi-reference (~45KB) security/best practices

## Key Patterns

### YAML Frontmatter Parsing
```python
# Strip frontmatter before returning to agent
if content.startswith("---"):
    parts = content.split("---", 2)
    if len(parts) >= 3:
        return parts[2].strip()  # Return only body
```

### Directory Traversal Prevention
```python
resolved_target = target_file.resolve()
resolved_skill = skill.skill_path.resolve()
if not resolved_target.is_relative_to(resolved_skill):
    return "Error: Access denied"
```

### Dependency Initialization
```python
@skill_agent.system_prompt
async def get_system_prompt(ctx: RunContext[AgentDependencies]) -> str:
    await ctx.deps.initialize()  # Initialize before using skill_loader
    return MAIN_SYSTEM_PROMPT.format(skill_metadata=...)
```

## Development Principles

1. **Type Safety**: All functions, methods, variables MUST have type annotations. Use Pydantic models for data structures.
2. **Google-style docstrings** for all functions/classes
3. **KISS/YAGNI**: Simple solutions, no premature abstractions
4. **Reference `examples/`** for patterns but DO NOT MODIFY those files
5. **Python 3.11+** required

## Configuration

Environment variables in `.env`:
```bash
LLM_PROVIDER=openrouter        # openrouter, openai, ollama
LLM_API_KEY=sk-or-v1-...
LLM_MODEL=anthropic/claude-haiku-4.5
LLM_BASE_URL=https://openrouter.ai/api/v1
SKILLS_DIR=skills
LOGFIRE_TOKEN=                 # Optional observability
```

## Test Structure

```
tests/
├── test_skill_loader.py   # SkillLoader/SkillMetadata unit tests
├── test_skill_tools.py    # Progressive disclosure tool tests
├── test_agent.py          # 71 integration tests with mocks
└── evals/
    ├── run_evals.py       # Evaluation runner
    ├── evaluators.py      # SkillWasLoaded, ToolWasCalled, etc.
    └── *.yaml             # Test datasets (skill_loading, response_quality, new_skills)
```
