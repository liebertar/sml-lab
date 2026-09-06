# sml-lab

Apple Silicon(MLX)에서 작은 모델을 **가장 효율적으로 돌리는 방법**을 실험하고 기록하는 저장소.
학습·양자화·서빙은 Mac Studio M4 Max 128GB에서, 배포는 브라우저(WebGPU)까지 간다.

## 우리가 하는 것

세 갈래를 하나의 파이프라인으로 잇는다. 각 갈래마다 "측정 → 가설 → 실험 → 검증 → 기록"을 반복한다.

| 갈래 | 질문 | 도구 |
|---|---|---|
| **A. MLX 최적화** | 같은 모델을 M4 Max에서 얼마나 더 빠르게, 더 적은 메모리로 돌릴 수 있나 | mlx, mlx-lm, `mx.fast.metal_kernel`, MLX 양자화(2~8bit), tinygrad Metal BEAM |
| **B. 로컬 SLM 서빙** | 0.5B~8B 모델을 로컬에서 서빙할 때 tok/s·TTFT·메모리·품질의 최적점은 어디인가 | mlx-lm server, 양자화 등급별 품질 회귀 세트, KV 캐시·speculative decoding |
| **C. 브라우저 배포** | 어디까지 줄여야 방문자 브라우저에서 도는가 | ONNX export, Transformers.js v4, ONNX Runtime Web, WebGPU |

기준은 하나다. **숫자로 확인되지 않은 최적화는 기록하지 않는다.** 모든 실험은 같은 하드웨어에서 같은 측정 스크립트로 전후를 비교한다.

## 왜 MLX인가

- M4 Max는 H100 대비 연산 1/29, 대역폭 1/6이지만 메모리는 128GB로 H100의 1.6배다. 큰 모델을 올려두고 오래 돌리는 데 유리하고, 토큰 생성처럼 대역폭에 묶이는 일은 H100의 1/3~1/4 수준으로 쓸 만하다.
- M4에서 PyTorch MPS의 FP16이 느린 보고가 있다. MLX가 Apple GPU를 가장 잘 쓰는 프레임워크다.
- CUDA 전용 기법(Triton, FlashAttention-3, FP8, nunchaku INT4 커널, EXL3)은 Mac에서 못 쓴다. 대신 **같은 아이디어를 MLX로 옮겨서** 검증하는 것이 이 저장소의 핵심 작업이다.

자세한 비교는 [docs/hardware.md](docs/hardware.md).

## 진행 방식

차근차근 간다. 한 번에 하나의 실험만 열고, 끝나면 README의 결과 표를 채운 뒤 다음으로 넘어간다.

1. **리서치**: 원본 논문·레포를 읽고 "무엇이 왜 빨라지는지"를 한 문단으로 적는다.
2. **기준선**: 아무것도 안 바꾼 상태의 숫자를 먼저 잰다.
3. **실험**: 변수 하나만 바꾼다.
4. **검증**: 속도만 보지 않는다. 품질 회귀 세트(고정 프롬프트·고정 시드·해시 고정 입력)로 전후를 비교한다.
5. **기록**: 실험 디렉토리 README에 결과 표와 배운 것을 남긴다. 블로그 글 하나가 실험 하나에 대응한다.

## 실험 로드맵

| # | 실험 | 갈래 | 상태 | 요약 |
|---|---|---|---|---|
| 00 | env-baseline | B | 계획 | M4 Max에서 Qwen3.5 0.6B~8B, Gemma 4 E2B를 4/8bit로 돌린 tok/s·TTFT·메모리 기준선 |
| 01 | [minrf-web](experiments/01-minrf-web) | C | 계획 | 소형 rectified flow 학습 → ONNX → 브라우저 생성. 전 구간을 가장 작게 한 번 통과 |
| 02 | quant-sweep-mlx | A | 계획 | 모델 × 비트수 × 그룹 크기 전수 스윕. 크기·속도·품질 매트릭스 |
| 03 | metal-kernel | A | 계획 | RMSNorm·RoPE 같은 작은 연산을 커스텀 Metal 커널로 바꿔 기본 대비 측정 |
| 04 | slm-serving | B | 계획 | mlx-lm server 부하 테스트. 동시 요청 수에 따른 tok/s, KV 캐시 양자화 효과 |
| 05 | tinygrad-metal-beam | A | 계획 | tinygrad BEAM 커널 탐색을 Metal에서 재현, MLX 기본 커널과 비교 |
| 06 | slm-browser | C | 계획 | 0.5B급 모델을 q4로 내보내 Transformers.js에서 실행. 다운로드 크기 대비 품질 |

## 참고 목록

배울 것과 가져올 것을 구분한다. 라이선스 없는 원본은 코드를 가져오지 않고 구조만 참고한다.

### 방법론 (실험 설계를 배운다)

| 원본 | 배우는 것 | 라이선스 |
|---|---|---|
| [ai-compiler-study/quanto](https://github.com/ai-compiler-study/quanto) | 체크포인트 × 정밀도 × 융합 여부 전수 스윕 표 | 없음 → 구조만 |
| [ai-compiler-study/test_attn](https://github.com/ai-compiler-study/test_attn) | 백엔드별 TFLOP/s·대역폭 자동 수집 방법론 | 없음 → 구조만 |
| [ai-compiler-study/flux-tinygrad-opt](https://github.com/ai-compiler-study/flux-tinygrad-opt) | 커널 자동탐색(BEAM/MCTS) 실험 설계 | MIT |
| [ai-compiler-study/triton-kernels](https://github.com/ai-compiler-study/triton-kernels) | 커널 레포 구조(kernels/ops/modules + tests + benchmarks) | MIT |
| [carpedm20/images](https://github.com/carpedm20/images) | 해시 고정 회귀 픽스처와 매니페스트 | CC0 |
| [love-fury/challenge](https://github.com/love-fury/challenge) | 세션 내 재사용(고정 입력은 한 번만 계산) 사고방식 | 없음 → 인용만 |

### MLX 생태계 (가져와 쓴다)

| 원본 | 용도 | 라이선스 |
|---|---|---|
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | 프레임워크. 양자화·커스텀 Metal 커널 | MIT |
| [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | LLM 실행·양자화·LoRA·서버 | MIT |
| [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) | Metal용 FlashAttention 구현 참고 | MIT |
| [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) | Metal 백엔드 커널 탐색 | MIT |
| [transformerlab/transformerlab-app](https://github.com/transformerlab/transformerlab-app) | 대화식 실험 UI (MLX 네이티브) | AGPL-3.0 |

### 양자화·캐싱 (아이디어를 MLX로 옮긴다)

| 원본 | 옮길 아이디어 | 라이선스 |
|---|---|---|
| [nunchaku-ai/nunchaku](https://github.com/nunchaku-ai/nunchaku) | SVDQuant: 저랭크 분기로 이상치 흡수 후 4bit | Apache-2.0 |
| [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) | 타임스텝 임베딩 기반 디퓨전 캐싱 | Apache-2.0 |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | EXL3 양자화 설계, KV 캐시 양자화 | MIT |
| [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) | 배치 크기와 무관한 결정적 추론 | MIT |
| [huggingface/optimum-quanto](https://github.com/huggingface/optimum-quanto) | fp8/int8/int4 양자화 API | Apache-2.0 |

### 브라우저 (배포 타깃)

| 원본 | 용도 | 라이선스 |
|---|---|---|
| [huggingface/transformers.js](https://github.com/huggingface/transformers.js) | 브라우저 추론 (WebGPU 런타임 v4) | Apache-2.0 |
| [mlc-ai/web-llm](https://github.com/mlc-ai/web-llm) | TVM 컴파일 기반 브라우저 LLM | Apache-2.0 |
| [mlc-ai/web-stable-diffusion](https://github.com/mlc-ai/web-stable-diffusion) | 브라우저 디퓨전 | Apache-2.0 |
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
