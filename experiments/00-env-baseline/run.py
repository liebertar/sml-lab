"""로컬 SLM 기준선 측정: 모델별 적재 시간, TTFT, 프리필/디코드 tok/s, 최대 메모리.

사용법:
    uv run python experiments/00-env-baseline/run.py
    uv run python experiments/00-env-baseline/run.py --models mlx-community/Qwen3-0.6B-4bit --runs 2
    uv run python experiments/00-env-baseline/run.py --preset studio
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MODELS = [
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3.5-2B-MLX-4bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
    "mlx-community/gemma-4-e2b-it-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/SmolLM3-3B-4bit",
]

STUDIO_MODELS = [
    "mlx-community/Qwen3.8-27B-4bit",
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    "mlx-community/gemma-4-12B-it-qat-4bit",
    "mlx-community/gemma-4-26b-a4b-it-4bit",
]

PRESETS = {"small": DEFAULT_MODELS, "studio": STUDIO_MODELS, "all": DEFAULT_MODELS + STUDIO_MODELS}

PROMPT_PARAGRAPH = (
    "Apple Silicon unifies CPU and GPU memory, which changes how inference engines "
    "should schedule work. Explain, step by step, how memory bandwidth and matrix "
    "throughput each limit token generation and prompt processing, and give one "
    "concrete optimization for each limit. "
)


@dataclass
class Measurement:
    model: str
    disk_gb: float
    load_seconds: float
    prompt_tokens: int
    ttft_seconds: float
    prefill_tps: float
    decode_tps: float
    peak_memory_gb: float
    runs: int


def machine_info() -> dict[str, str]:
    def sysctl(key: str) -> str:
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "?"

    import mlx.core as mx

    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": f"{int(sysctl('hw.memsize') or 0) / 2**30:.0f}",
        "macos": platform.mac_ver()[0],
        "mlx": mx.__version__,
    }


def model_disk_gb(repo_id: str) -> float:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id))
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30


def build_prompt(tokenizer, repeats: int) -> str:
    text = PROMPT_PARAGRAPH * repeats
    messages = [{"role": "user", "content": text}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return text


def run_once(model, tokenizer, prompt: str, max_tokens: int) -> dict[str, float]:
    import mlx.core as mx
    from mlx_lm import stream_generate

    mx.reset_peak_memory()
    start = time.perf_counter()
    ttft = None
    last = None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if ttft is None:
            ttft = time.perf_counter() - start
        last = response
    assert last is not None and ttft is not None
    return {
        "prompt_tokens": last.prompt_tokens,
        "ttft": ttft,
        "prefill_tps": last.prompt_tps,
        "decode_tps": last.generation_tps,
        "peak_memory_gb": last.peak_memory,
    }


def measure(repo_id: str, repeats: int, max_tokens: int, runs: int) -> Measurement:
    import mlx.core as mx
    from mlx_lm import load

    disk_gb = model_disk_gb(repo_id)
    start = time.perf_counter()
    model, tokenizer = load(repo_id)
    load_seconds = time.perf_counter() - start
    prompt = build_prompt(tokenizer, repeats)

    run_once(model, tokenizer, prompt, max_tokens=8)  # 워밍업: 커널 컴파일·캐시 제외
    samples = [run_once(model, tokenizer, prompt, max_tokens) for _ in range(runs)]
    median = lambda key: statistics.median(s[key] for s in samples)  # noqa: E731

    result = Measurement(
        model=repo_id,
        disk_gb=round(disk_gb, 2),
        load_seconds=round(load_seconds, 2),
        prompt_tokens=int(samples[0]["prompt_tokens"]),
        ttft_seconds=round(median("ttft"), 3),
        prefill_tps=round(median("prefill_tps"), 1),
        decode_tps=round(median("decode_tps"), 1),
        peak_memory_gb=round(max(s["peak_memory_gb"] for s in samples), 2),
        runs=runs,
    )
    del model, tokenizer
    clear_cache = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
    if clear_cache:
        clear_cache()
    return result


def format_table(rows: list[Measurement]) -> str:
    columns = [
        "모델", "디스크 GB", "적재 s", "프롬프트 tok",
        "TTFT s", "프리필 tok/s", "디코드 tok/s", "최대 메모리 GB",
    ]
    header = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
    lines = [
        f"| {r.model.split('/')[-1]} | {r.disk_gb} | {r.load_seconds} | {r.prompt_tokens} | "
        f"{r.ttft_seconds} | {r.prefill_tps} | {r.decode_tps} | {r.peak_memory_gb} |"
        for r in rows
    ]
    return "\n".join([header, *lines])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small",
                        help="small: ≤9B 8종 / studio: 27B급 4종 / all: 둘 다")
    parser.add_argument("--models", default=None,
                        help="쉼표로 구분한 HF repo id. 지정하면 --preset 무시")
    parser.add_argument("--repeats", type=int, default=8,
                        help="프롬프트 문단 반복 수 (약 60 tok × 반복)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3,
                        help="워밍업 제외 측정 횟수, 중앙값 사용")
    parser.add_argument("--out", default="outputs/baseline.json")
    args = parser.parse_args()

    info = machine_info()
    print(f"# {info['chip']} / {info['memory_gb']} GB / "
          f"macOS {info['macos']} / mlx {info['mlx']}\n")
    rows: list[Measurement] = []
    selected = args.models.split(",") if args.models else PRESETS[args.preset]
    for repo_id in [m.strip() for m in selected if m.strip()]:
        print(f"→ {repo_id}", flush=True)
        try:
            rows.append(measure(repo_id, args.repeats, args.max_tokens, args.runs))
        except Exception as error:  # noqa: BLE001
            print(f"  skipped: {type(error).__name__}: {error}")
            continue
        print("  " + format_table(rows[-1:]).splitlines()[-1])

    print("\n" + format_table(rows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"machine": info, "results": [asdict(r) for r in rows]}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
