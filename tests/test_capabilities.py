from agents.types import Capability
from agents.cli.base import BaseCLIAgent
from agents.cli.cursor import CursorAgent


def test_capability_is_str_valued():
    # Must be str-valued, not StrEnum: the project floor is Python 3.10.
    assert Capability.WORKSPACE == "workspace"
    assert Capability.CONTEXT == "context_files"
    assert isinstance(Capability.TOOLS, str)


def test_base_cli_agent_defaults_to_workspace_and_tools():
    assert BaseCLIAgent.capabilities == frozenset(
        {Capability.WORKSPACE, Capability.TOOLS}
    )


def test_existing_backends_inherit_the_default():
    # cursor/codex/copilot must keep working with no edits.
    assert Capability.WORKSPACE in CursorAgent.capabilities
    assert Capability.TOOLS in CursorAgent.capabilities
