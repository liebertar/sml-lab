# sml-lab

Apple Silicon(MLX)에서 작은 모델을 **가장 효율적으로 돌리는 방법**을 실험하고 기록하는 저장소.
학습·양자화·서빙은 Mac Studio M4 Max 128GB에서, 배포는 브라우저(WebGPU)까지 간다.

> Studio에서 켜자마자 할 일은 **[TASKS.md](TASKS.md)** 에 있다. 명령, 예상 시간, 확인 방법, 성공 기준 순서로 적혀 있다.

## 우리가 하는 것

세 갈래를 하나의 파이프라인으로 잇는다. 각 갈래마다 "측정 → 가설 → 실험 → 검증 → 기록"을 반복한다.

| 갈래 | 질문 | 도구 |
|---|---|---|
| **A. MLX 최적화** | 같은 모델을 M4 Max에서 얼마나 더 빠르게, 더 적은 메모리로 돌릴 수 있나 | mlx 0.32, mlx-lm 0.31, `mx.fast.metal_kernel`, `mx.compile`, tinygrad Metal BEAM |
| **B. 로컬 SLM 서빙** | 0.6B~9B 모델을 로컬에서 서빙할 때 tok/s·TTFT·메모리·품질의 최적점은 어디인가 | `mlx_lm.server`, KV 캐시 양자화, 프롬프트 캐시, speculative decoding, 배치 생성 |
| **C. 브라우저 배포** | 어디까지 줄여야 방문자 브라우저에서 도는가 | ONNX export, Transformers.js v4, ONNX Runtime Web, WebGPU(Safari 26 포함) |

기준은 하나다. **숫자로 확인되지 않은 최적화는 기록하지 않는다.** 모든 실험은 같은 하드웨어에서 같은 측정 스크립트로 전후를 비교한다.

## 왜 MLX인가

- M4 Max는 H100 대비 연산 1/29, 대역폭 1/6이지만 메모리는 128GB로 H100의 1.6배다. 큰 모델을 올려두고 오래 돌리는 데 유리하고, 토큰 생성처럼 대역폭에 묶이는 일은 H100의 1/3~1/4 수준으로 쓸 만하다.
- M4에서 PyTorch MPS의 FP16이 느린 보고가 있다. MLX가 Apple GPU를 가장 잘 쓰는 프레임워크다.
- CUDA 전용 기법(Triton, FlashAttention-3, FP8 GEMM, INT4 전용 커널)은 Mac에서 못 쓴다. 대신 **같은 아이디어를 MLX로 옮겨서** 검증하는 것이 이 저장소의 핵심 작업이다.

자세한 비교는 [docs/hardware.md](docs/hardware.md).

## 공부할 방향

2026년 9월 기준으로 실제 설치돼 있고 바로 돌려볼 수 있는 것만 적는다.

### 1. 양자화 (A)
- **가중치**: `mlx_lm.convert --q-mode {affine,mxfp4,nvfp4,mxfp8} --q-bits --q-group-size`. 같은 4bit라도 affine과 mxfp4의 속도·품질 차이를 재는 것이 첫 질문.
- **학습형 양자화**: `mlx_lm.dwq`(증류로 양자화 오차 보정), `mlx_lm.dynamic_quant`(레이어 민감도 기반 혼합 비트), `mlx_lm.awq`, `mlx_lm.gptq`. 2B급에서 perplexity 차이를 표로.
- **KV 캐시**: `--kv-bits 4|8 --kv-group-size`. 긴 문맥에서 메모리와 속도가 어떻게 바뀌는지.
- 품질 기준은 `mlx_lm.perplexity`와 고정 프롬프트 회귀 세트 두 개를 같이 본다.

### 2. 디코드 가속 (B)
- **speculative decoding**: `--draft-model`(같은 토크나이저의 소형 모델) + `--num-draft-tokens`. 수락률과 배속의 관계.
- **프롬프트 캐시**: `mlx_lm.cache_prompt`로 시스템 프롬프트·문서를 미리 인코딩해 TTFT를 줄인다.
- **배치 생성**: `mlx_lm.benchmark -b 1|4|8`. 대역폭에 묶인 디코드는 배치를 키워도 시간이 거의 안 늘어난다는 가설 검증.
- **MTP 모델**: multi-token prediction 변환본(예: Qwen3.8 MTP)이 MLX에서 실제로 빨라지는지.

### 3. 커널과 컴파일 (A)
- `mx.compile`로 그래프 융합 효과 측정 → `mx.fast.metal_kernel`로 RMSNorm·RoPE 같은 작은 연산을 직접 작성해 비교.
- tinygrad의 Metal 백엔드에서 BEAM 커널 탐색을 돌려 MLX 기본 커널과 같은 연산을 비교.
- Metal System Trace(Xcode Instruments)로 커널 단위 시간을 읽는 법.

### 4. 서빙 (B)
- `mlx_lm.server`(OpenAI 호환)로 동시 요청 수에 따른 tok/s와 지연. `--draft-model`, `--prefill-step-size` 효과.
- 분산: `mlx.launch`로 Mac 2대(Studio + 노트북)에 파이프라인 분할이 되는지.

### 5. 브라우저 (C)
- ONNX export → q4f16 → Transformers.js v4(WebGPU 런타임)에서 실행. 다운로드 크기 기준: 200MB 이하 방문자용, 2GB 이하 클릭 후 로드, 그 이상 데모 전용.
- 직접 학습한 소형 생성 모델(rectified flow)을 수 MB로 내보내 페이지에서 생성.

### 6. 학습 (A)
- `mlx_lm.lora`로 2B급 LoRA/QLoRA. 학습 tok/s와 메모리를 기준선 표에 추가.
- 소형 rectified flow를 처음부터 학습(`third_party/minRF`)해 학습→양자화→배포 전 구간을 작게 통과.

## 실험 로드맵

| # | 실험 | 갈래 | 상태 | 요약 |
|---|---|---|---|---|
| 00 | [env-baseline](experiments/00-env-baseline) | B | 하네스 완료, Studio 측정 대기 | 4bit 8종의 tok/s·TTFT·메모리 기준선 |
| 01 | [minrf-web](experiments/01-minrf-web) | C | 계획 | 소형 rectified flow 학습 → ONNX → 브라우저 생성 |
| 02 | quant-sweep-mlx | A | 계획 | 모델 × 비트수 × 그룹 크기 × q-mode 전수 스윕 |
| 03 | metal-kernel | A | 계획 | 작은 연산을 커스텀 Metal 커널로 바꿔 기본 대비 측정 |
| 04 | slm-serving | B | 계획 | `mlx_lm.server` 부하 테스트, KV 캐시 양자화 효과 |
| 05 | tinygrad-metal-beam | A | 계획 | BEAM 커널 탐색을 Metal에서 재현, MLX와 비교 |
| 06 | slm-browser | C | 계획 | 0.5B급을 q4로 내보내 Transformers.js에서 실행 |

## 참고 프로젝트

라이선스 없는 원본은 코드를 가져오지 않고 구조만 참고한다.

| 원본 | 용도 | 라이선스 |
|---|---|---|
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | 프레임워크. 양자화·커스텀 Metal 커널 | MIT |
| [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | LLM 실행·양자화·LoRA·서버·벤치마크 | MIT |
| [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) | Metal용 FlashAttention 구현 참고 | MIT |
| [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) | Metal 백엔드 커널 탐색 | MIT |
| [transformerlab/transformerlab-app](https://github.com/transformerlab/transformerlab-app) | 대화식 실험 UI (MLX 네이티브) | AGPL-3.0 |
| [nunchaku-ai/nunchaku](https://github.com/nunchaku-ai/nunchaku) | SVDQuant: 저랭크 분기로 이상치 흡수 후 4bit. 아이디어를 MLX로 | Apache-2.0 |
| [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) | 타임스텝 임베딩 기반 디퓨전 캐싱 | Apache-2.0 |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | EXL3 양자화 설계, KV 캐시 양자화 | MIT |
| [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) | 배치 크기와 무관한 결정적 추론 | MIT |
| [huggingface/transformers.js](https://github.com/huggingface/transformers.js) | 브라우저 추론 (WebGPU 런타임 v4) | Apache-2.0 |
| [mlc-ai/web-llm](https://github.com/mlc-ai/web-llm) | TVM 컴파일 기반 브라우저 LLM | Apache-2.0 |
| [cloneofsimo/minRF](https://github.com/cloneofsimo/minRF) | 최소 rectified flow 학습 코드 (`third_party/minRF`) | Apache-2.0 |

## 구조

```
experiments/   실험 1개 = 디렉토리 1개. README에 목표·출처·결과·라이선스 기록
third_party/   git subtree로 가져온 원본 코드. 수정하지 않고 experiments/에서 import
scripts/       환경 점검 등 공용 스크립트
docs/          하드웨어 비교 등 참고 문서
```

## 시작

```bash
make env    # uv sync
make check  # 칩·메모리·MLX·ONNX Runtime 상태 출력
make test   # 스모크 테스트
```

## 라이선스

이 저장소의 코드는 MIT. `third_party/` 아래 코드는 각 디렉토리의 원본 라이선스를 따른다.
