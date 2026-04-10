"""
Noise Cutter — Multi-Agent Orchestrator
========================================
Launches the Project Manager agent, which has access to the Developer,
Tester, and Researcher agents via the Agent tool.

Usage:
    python agents/orchestrate.py
    python agents/orchestrate.py "evaluate current model checkpoints"
    python agents/orchestrate.py "generate a progress report"
    python agents/orchestrate.py "continue upscaling and training"

If no directive is given, the PM assesses pipeline state and decides
what to do next on its own.
"""

import sys
import anyio
from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    AssistantMessage,
    TextBlock,
    SystemMessage,
    CLINotFoundError,
    CLIConnectionError,
)
from agent_definitions import (
    PROJECT_ROOT,
    PROJECT_MANAGER,
    DEVELOPER,
    TESTER,
    RESEARCHER,
)

load_dotenv(dotenv_path=f"{PROJECT_ROOT}\\.env")

# Default directive when none is provided on the command line
DEFAULT_DIRECTIVE = (
    "Assess the current state of the Noise Cutter pipeline: "
    "check how many Waterloo videos have been upscaled, review the latest "
    "training checkpoint loss, and decide whether to continue upscaling, "
    "resume training, or run a validation pass. Then carry out the highest-"
    "priority action by delegating to the appropriate agent."
)


async def run_orchestrator(directive: str) -> None:
    print(f"\n{'='*70}")
    print("  NOISE CUTTER — MULTI-AGENT ORCHESTRATOR")
    print(f"{'='*70}")
    print(f"  Directive: {directive}")
    print(f"{'='*70}\n")

    options = ClaudeAgentOptions(
        cwd=PROJECT_ROOT,
        system_prompt=PROJECT_MANAGER.prompt,
        allowed_tools=PROJECT_MANAGER.tools,
        agents={
            "developer":  DEVELOPER,
            "tester":     TESTER,
            "researcher": RESEARCHER,
        },
        permission_mode="acceptEdits",
        max_turns=50,
        model="claude-opus-4-6",
    )

    try:
        async for message in query(prompt=directive, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        print(block.text)

            elif isinstance(message, SystemMessage):
                if message.subtype == "init":
                    session_id = message.data.get("session_id", "unknown")
                    print(f"[session: {session_id}]\n")

            elif isinstance(message, ResultMessage):
                print(f"\n{'='*70}")
                print("  RUN COMPLETE")
                print(f"  Stop reason: {message.stop_reason}")
                print(f"{'='*70}\n")

    except CLINotFoundError:
        print("ERROR: Claude Code CLI not found.")
        print("Install with:  pip install claude-agent-sdk")
        sys.exit(1)
    except CLIConnectionError as e:
        print(f"ERROR: Connection failed — {e}")
        sys.exit(1)


def main() -> None:
    directive = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_DIRECTIVE
    anyio.run(run_orchestrator, directive)


if __name__ == "__main__":
    main()
