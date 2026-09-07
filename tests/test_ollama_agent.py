import pytest

from agents.inprocess.ollama import (
    OllamaAgent, TASK_MODELS, embed_files, size_num_ctx,
)
from agents.types import AgentConfig, Capability


def test_capabilities_are_context_only():
    assert OllamaAgent.capabilities == frozenset({Capability.CONTEXT})


def test_task_model_presets():
    assert TASK_MODELS["code"] == "qwen2.5-coder:32b"
    assert TASK_MODELS["summary"] == "gemma3:27b"
    assert TASK_MODELS["fast"] == "gemma4:12b"


@pytest.mark.parametrize("chars,expected", [
    (100, 4096),
    (30_000, 16384),
    (200_000, 131072),
])
def test_num_ctx_grows_with_input(chars, expected):
    # Ollama defaults to 4096 and SILENTLY truncates past it, so the context
    # has to be sized explicitly from the input.
    assert size_num_ctx(chars) == expected


def test_num_ctx_never_below_default():
    assert size_num_ctx(0) == 4096


def test_embed_files_includes_line_numbers(tmp_path):
    # Without line numbers the model estimates citations and drifts 5-15 lines.
    f = tmp_path / "a.py"
    f.write_text("first\nsecond\n")
    embedded = embed_files([f])
    assert "1\tfirst" in embedded
    assert "2\tsecond" in embedded
    assert str(f) in embedded


def test_embed_files_empty_list():
    assert embed_files([]) == ""


async def test_run_maps_telemetry(monkeypatch, tmp_path):
    agent = OllamaAgent(model="gemma4:12b")

    async def fake_post(self, prompt, num_ctx, model):
        return {
            "response": "the answer",
            "prompt_eval_count": 4249,
            "eval_count": 287,
        }

    monkeypatch.setattr(OllamaAgent, "_post_generate", fake_post)
    result = await agent.run("q", AgentConfig(workspace=tmp_path))

    assert result.success is True
    assert result.output == "the answer"
    assert result.run.llm_input_tokens == 4249
    assert result.run.llm_output_tokens == 287
    # The zero is the point: it is what makes savings legible in stats.
    assert result.run.llm_total_cost_usd == 0.0
    assert result.run.model == "gemma4:12b"
    assert result.run.status == "completed"


async def test_run_marks_failed_on_error(monkeypatch, tmp_path):
    agent = OllamaAgent()

    async def boom(self, prompt, num_ctx, model):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(OllamaAgent, "_post_generate", boom)
    result = await agent.run("q", AgentConfig(workspace=tmp_path))

    assert result.success is False
    assert result.run.status == "failed"
    assert "connection refused" in result.output


async def test_cancel_before_run_is_false():
    assert await OllamaAgent().cancel() is False
