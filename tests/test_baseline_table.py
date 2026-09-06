import importlib.util
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "baseline_run", Path(__file__).resolve().parents[1] / "experiments/00-env-baseline/run.py"
)
baseline = importlib.util.module_from_spec(SPEC)
sys.modules["baseline_run"] = baseline  # dataclass가 모듈을 찾을 수 있게 등록
SPEC.loader.exec_module(baseline)


def test_format_table_renders_one_row_per_measurement():
    row = baseline.Measurement(
        model="mlx-community/Qwen3-0.6B-4bit",
        disk_gb=0.35,
        load_seconds=0.8,
        prompt_tokens=500,
        ttft_seconds=0.12,
        prefill_tps=4000.0,
        decode_tps=180.0,
        peak_memory_gb=0.9,
        runs=3,
    )
    table = baseline.format_table([row, row])
    lines = table.splitlines()
    assert lines[0].startswith("| 모델 |")
    assert len(lines) == 4
    assert "Qwen3-0.6B-4bit" in lines[2]
    assert "180.0" in lines[3]


def test_build_prompt_falls_back_to_plain_text_without_chat_template():
    class Tokenizer:
        chat_template = None

    prompt = baseline.build_prompt(Tokenizer(), repeats=2)
    assert prompt == baseline.PROMPT_PARAGRAPH * 2


def test_presets_are_disjoint_and_all_is_their_union():
    small, studio, everything = (baseline.PRESETS[k] for k in ("small", "studio", "all"))
    assert not set(small) & set(studio)
    assert everything == small + studio
