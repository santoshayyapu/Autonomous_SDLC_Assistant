"""Architect Agent

Transforms a natural-language requirement into:
  - technical_spec  : a detailed description of modules, classes, functions
  - file_structure  : list of Python source filenames to create (1-3 files)
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# ── Schema ───────────────────────────────────────────────────────────────────

class ArchitectOutput(BaseModel):
    technical_spec: str = Field(
        description="A highly detailed Markdown string containing Executive Summary, Mermaid Diagram, Component Architecture, Data Models, Error Handling Strategy, Security & Validation."
    )
    file_structure: list[str] = Field(
        description="A list of Python filenames to structure the project (e.g. ['models.py', 'exceptions.py', 'processor.py'])"
    )

# ── Node ─────────────────────────────────────────────────────────────────────

def architect_node(state: dict) -> dict:
    """LangGraph node: produce a technical spec and file plan."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    structured_llm = llm.with_structured_output(ArchitectOutput)

    tone = state.get("tone", "Standard")
    prompt = f"""You are a Principal Software Architect designing a highly robust Python application.

REQUIREMENT:
{state['requirement']}

Target Tone/Style: {tone}

Your job is to OVER-ENGINEER the planning phase. Provide an exhaustive, senior-level technical specification. Do not just repeat the input.

1. "technical_spec" must contain:
   • **Executive Summary**: What the system does and its core value proposition.
   • **Mermaid Diagram**: A `mermaid` block containing either a flow chart or class diagram representing the system. IT MUST START WITH `%%{{init: {{'theme': 'dark'}}}}%%` immediately inside the fences.
   • **Component Architecture**: Breakdown of the logic (e.g. Data Parsing, Business Logic, Validation).
   • **Data Models**: Exact schemas / classes needed, types of inputs/outputs.
   • **Error Handling Strategy**: List the exact edge cases and the specific Custom Exceptions you will use (e.g. `InvalidDataError`, `MalformedFileError`).
   • **Security & Validation**: How inputs are sanitized.

2. "file_structure" must contain:
   • Break logic into 2-3 logical domain files if the tone warrants it (e.g., separate exceptions from core logic).
   • No test files (those are generated separately).
   • Python standard library only.
"""

    logs = list(state.get("logs", []))
    logs.append("🏗️  [Architect] Analyzing requirement and planning architecture…")

    try:
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        spec  = response.technical_spec
        files = response.file_structure

        logs.append(f"✅ [Architect] Spec ready. Planned file(s): {files}")
        return {"technical_spec": spec, "file_structure": files, "logs": logs}

    except Exception as exc:
        logs.append(f"❌ [Architect] Error: {exc}")
        return {
            "technical_spec": state.get("requirement", ""),
            "file_structure": ["main.py"],
            "logs": logs,
        }
