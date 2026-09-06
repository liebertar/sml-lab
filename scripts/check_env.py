"""현재 머신의 AI 실행 환경을 한눈에 출력한다."""

import platform
import subprocess


def sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "?"


def main() -> None:
    print(f"machine      : {platform.machine()} / {platform.mac_ver()[0] or platform.platform()}")
    print(f"chip         : {sysctl('machdep.cpu.brand_string')}")
    memory_gb = int(sysctl("hw.memsize") or 0) / 2**30
    print(f"memory       : {memory_gb:.0f} GB unified")

    try:
        import mlx.core as mx

        info = mx.device_info()
        print(f"mlx          : {mx.__version__} on {mx.default_device()}")
        print(f"metal device : {info.get('device_name')} / "
              f"max working set {info.get('max_recommended_working_set_size', 0) / 2**30:.0f} GB")
    except ImportError as error:
        print(f"mlx          : not installed ({error})")

    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        print(f"onnxruntime  : {onnxruntime.__version__} providers={providers}")
    except ImportError:
        print("onnxruntime  : not installed")

    try:
        import torch

        print(f"torch        : {torch.__version__} mps={torch.backends.mps.is_available()}")
    except ImportError:
        print("torch        : not installed (optional: uv sync --extra torch)")


if __name__ == "__main__":
    main()
