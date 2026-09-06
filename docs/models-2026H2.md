# 2026년 하반기 기준 학습·실험할 만한 공개 가중치 모델

기준일 2026-09-07. 출시일·파라미터·라이선스는 Hugging Face API에서 읽은 값이고, MLX 변환본은 `mlx-community`에 있는 것만 표기했다.
"M4 Max 128GB에서" 열은 이 저장소의 실험 머신 기준이다. 판정 규칙은 맨 아래 [메모리 계산](#메모리-계산-규칙)에 있다.

## 먼저 답: Qwen은 뭐가 최신인가

| 계열 | 출시 | 크기 | 상태 |
|---|---|---|---|
| Qwen3.5 | 2026-02 | 0.8B / 2B / 4B / 9B / 27B / 122B-A10B / 397B-A17B | **소형(≤9B)은 여전히 이게 최신.** Base 체크포인트 있음 → 학습용 |
| Qwen3.6 | 2026-04 | 27B, 35B-A3B | 소형 없음 |
| Qwen3.8 | 2026-08 | 27B(dense, VLM 내장, Apache-2.0), 2.4T-A95B, **Flash-Next**(125B-A6B + 51B n-gram + 4B MTP, Qwen4 아키텍처 예고판, 커스텀 라이선스) | 27B가 Studio에서 돌릴 최상위 dense |
| Qwen 4 | 미정 | 소문만 (2026-09?) | 공식 확인 없음 |

결론: 기준선의 소형 3종(Qwen3.5 2B/4B/9B)은 그대로 유효하고, Studio 등급에 Qwen3.8-27B를 추가한다.

## A. 텍스트 / 멀티모달 LLM

### 학습 대상 (≤9B, Base 체크포인트 있음)

| 모델 | 출시 | 파라미터 | 라이선스 | MLX | M4 Max 128GB에서 | 비고 |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B / 2B / 4B / 9B | 2026-02 | 0.9 / 2.3 / 4.7 / 9.7B | Apache-2.0 | 4bit·6bit·8bit 전부 | 0.8~4B **풀 파인튜닝 가능**, 9B는 LoRA | 이미지 입력 내장. `-Base` 있음 |
| Gemma 4 E2B / E4B | 2026-03 | 5.1B(유효 2B) / 8.0B(유효 4B) | Apache-2.0 | 4bit, qat-mobile | E2B 풀 FT 가능, E4B LoRA | 텍스트+이미지+오디오 any-to-any. `-pt` 있음 |
| Gemma 4 12B | 2026-05 | 12.0B | Apache-2.0 | qat-4bit | LoRA(bf16 24GB) | 12B가 Gemma 4 소형 중 최상 |
| Ministral 3 3B / 8B | 2025-10 | 3.8 / 8.9B | Apache-2.0 | 있음 | 3B 풀 FT, 8B LoRA | Reasoning 변종 있음 |
| Granite 4.2 3B / 8B | 2026-08 | 3.7 / 8.8B | Apache-2.0 | 확인 필요 | 3B 풀 FT | IBM, 하이브리드 Mamba |
| Nanbeige4.2-3B | 2026-07 | 4.2B | Apache-2.0 | 확인 필요 | 풀 FT | mlx-dspark에서 spec decoding 4.25× 기록 |
| LFM2.5 230M / 350M / 1.2B / 2.6B / 8B-A1B | 2026-01~08 | 0.2~8.5B | LFM 오픈 라이선스(other) | 확인 필요 | 전부 풀 FT 범위 | Liquid, 엣지 특화 conv-hybrid. 1.2B가 spec decoding 3.8× |
| SmolLM3-3B | 2025-07 | 3.1B | Apache-2.0 | 있음 | 풀 FT | 학습 레시피 전부 공개 |
| Phi-4-mini | 2025-02 | 3.8B | MIT | 있음 | 풀 FT | 구세대지만 추론 특화 |

### Studio 등급 (추론·LoRA·QLoRA)

| 모델 | 출시 | 파라미터 | 라이선스 | MLX 4bit 크기 | M4 Max 128GB에서 | 비고 |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | 2026-08 | 27.8B dense | Apache-2.0 | 15.0 GB (+MTP 0.25 GB) | 추론 25~38 tok/s, LoRA(bf16 56GB) 가능 | VLM 내장, 262K 문맥. MTP·mxfp4·nvfp4 변환본 있음 |
| Qwen3.6-35B-A3B | 2026-04 | 36B MoE, 활성 3B | Apache-2.0 | 19.0 GB | 추론 100 tok/s 이상 기대(활성 3B) | Mac에 가장 유리한 형태 |
| Gemma 4 26B-A4B | 2026-03 | 25.8B MoE, 활성 4B | Apache-2.0 | 14.3 GB | 추론 빠름, QLoRA 가능 | 멀티모달 |
| Gemma 4 31B | 2026-03 | 31.3B dense | Apache-2.0 | ~17 GB | 추론 20~30 tok/s, LoRA 경계(bf16 63GB) | |
| Muse-Glimmer-30B | 2026-08 | 29.8B dense | Apache-2.0 | 확인 필요 | 추론, QLoRA | Meta. 에이전트·툴콜 특화, 멀티모달 |
| Ornith-1.5 9B / 35B-A3B | 2026-08 | 9.7B / 36B-A3B | MIT | 확인 필요 | 9B LoRA, 35B 추론 | 코딩 에이전트. 벤치마크는 전부 벤더 자체 보고 |
| Bonsai-27B (Qwen3.6-27B 저비트판) | 2026-07 | 27B, 1.125bpw / 2bpw | 확인 필요 | 3.9 GB / 5.9 GB | 추론. **A 갈래 연구 대상** | 1bit 89.5%, ternary 94.6% 성능 유지 주장. MLX 전용 커널 |
| Nemotron 3.5 Lightning 30B-A3B | 2026-08 | 30B MoE | NVIDIA 오픈 | 확인 필요 | 추론 | DSpark/DFlash 드래프트 동봉 |
| Mistral Small 4 119B | 2026-01 | 119B | Apache-2.0 | ~62 GB | 추론만 | |
| Qwen3.8-Flash-Next | 2026-08 | 125B-A6B (+55B) | 커스텀 | 4bit ~95 GB | 추론 경계 | Gated DeltaNet + 희소 어텐션 + n-gram 메모리. 구조 공부용 |

### 못 올리는 것 (참고만)
DeepSeek-V4-Flash 304B(MIT), GLM-5.3-Flash 321B(MIT), Qwen3.8-2.4T, Ornith-1.5-397B. 4bit로도 128GB를 넘는다.

## B. 디코드 가속 (2026년의 핵심 변화)

| 도구 | 라이선스 | 내용 | 우리 쓰임 |
|---|---|---|---|
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec) | MIT | DeepSeek의 spec decoding 학습·평가 프레임워크. DSpark / DFlash / Eagle3 드래프터. Qwen3 4B·8B·14B, Gemma 4 12B용 학습된 드래프트 공개 | 드래프트 **학습** 레시피 (CUDA) |
| [mlx-dspark](https://github.com/ARahim3/mlx-dspark) | MIT | DSpark·DFlash의 **MLX 네이티브 포트**. 무손실. Gemma 4 12B 2.6~3.1×, Qwen3.8-27B 8bit 최대 4.06×, LFM2.5-1.2B 3.8× | TASKS T4b. 기준선 대비 배속 재현 |
| mlx-lm `--draft-model` | MIT | 소형 모델을 드래프트로 쓰는 고전 방식 | T4. mlx-dspark와 비교 |
| Qwen3.8-27B-MTP | Apache-2.0 | 모델 내장 multi-token prediction 헤드 (0.25 GB) | mlx-lm이 MTP를 쓰는지 확인 |

## C. 이미지 생성 (Mac에서 학습 가능한 것 위주)

| 모델 | 출시 | 파라미터 | 라이선스 | Mac 실행 | Mac 학습 | 비고 |
|---|---|---|---|---|---|---|
| **FLUX.2 Klein 4B** (+base) | 2026-01 | 3.9B | Apache-2.0 | mflux 4bit, MLX 4bit 변환본. 512px 4스텝 ~9초, 1024px 30~40초(M1 Max) | **LoRA: mflux, flux-2-swift-mlx** | 상업 사용 OK. 텍스트 인코더 Qwen3 4B/8B |
| FLUX.2 Klein 9B | 2026-01 | 9.1B | 비상업 | mflux, 29 GB | LoRA | 품질 ↑, 라이선스 주의 |
| **Z-Image / Z-Image-Turbo** | 2025-11 / 2026-01 | 6.2B | Apache-2.0 | mflux, MLX bf16 | **LoRA: mflux(공식 지원)** | 빠르고 사실적. 첫 LoRA 실험 후보 |
| Qwen-Image-2512 | 2025-12 | 20.4B | Apache-2.0 | MLX 3~8bit(4bit ~12 GB), mflux | 학습 불가(크기) | 프롬프트 이해·텍스트 렌더링 최상 |
| HiDream-O1-Image | 2026-05 | 8.8B | MIT | PyTorch MPS | LoRA 가능성 | 생성+편집 통합 |
| Krea 2 / Ideogram 4 / ERNIE-Image / Lens / Boogu / FIBO | 2026 | 8~12B | 각각 확인 | mflux 지원 | 아니오 | 스타일·타이포·다국어 등 특화 비교용 |
| Sana Sprint 0.6B / 1.6B | 2025-03 | 0.6 / 1.6B | Apache-2.0 | PyTorch MPS | **풀 FT 가능한 크기** | 1~4스텝. 소형 디퓨전 학습 실험용 |
| Lumina-Image 2.0 | 2025-01 | 2.6B | Apache-2.0 | PyTorch MPS | LoRA | |
| minRF (직접 학습) | — | 수십 M | Apache-2.0 | MPS/MLX | **처음부터 학습** | 실험 01. 브라우저 배포 대상 |
| SeedVR2 3B / 7B | 2025-06 | 3 / 7B | 확인 | mflux | 아니오 | 업스케일러 |

비디오: SANA-Video 2.0 5B(2026-08, Apache-2.0), Wan2.2-TI2V-5B(Apache-2.0). MPS에서 추론은 되지만 분 단위. 학습은 대상 아님.

## D. 음성

| 모델 | 출시 | 파라미터 | 라이선스 | MLX | 비고 |
|---|---|---|---|---|---|
| Qwen3-TTS 0.6B / 1.7B (Base·CustomVoice·VoiceDesign) | 2026-01 | 0.9 / 1.9B | Apache-2.0 | bf16·8bit 있음 | 목소리 지정·설계. 0.6B는 풀 FT 범위 |
| Qwen3-ASR 0.6B / 1.7B | 2026-01 | 0.9 / 2.3B | Apache-2.0 | 4bit·8bit 있음 | Whisper 대체. ForcedAligner 동반 |
| Kokoro-82M | 2024-12 | 82M | Apache-2.0 | kokoro.js(브라우저) | 브라우저 배포 대상 (C 갈래) |
| Nemotron-3 Diarization | 2026-08 | — | NVIDIA | — | 화자 분리 |

## E. 임베딩·검색

| 모델 | 출시 | 파라미터 | 라이선스 | 비고 |
|---|---|---|---|---|
| Qwen3-Embedding 0.6B / 4B / 8B | 2025-06 | 0.6~8B | Apache-2.0 | 0.6B가 브라우저 경계(ONNX q8 ~600MB) |
| Qwen3-VL-Embedding 2B / 8B | 2026-01 | 2.1 / 8B | Apache-2.0 | 이미지+텍스트 임베딩 |
| EmbeddingGemma 300M | 2025-08 | 0.3B | Gemma | 브라우저 배포 1순위 (300M, QAT q4/q8) |
| Nemotron-3-Embed 1B / 8B | 2026-07 | 1.1 / 8B | NVIDIA | |

## 메모리 계산 규칙

M4 Max 128GB, 실제 사용 가능 ~110GB 기준. mlx-lm `lora`는 `--fine-tune-type {lora,dora,full}`.

| 방식 | 파라미터당 메모리 | 상한 (활성화 여유 포함) |
|---|---|---|
| 풀 파인튜닝 bf16 + AdamW | ~12 byte (가중치 2 + 그래드 2 + m,v 8) | **~5B** |
| LoRA, bf16 베이스 | ~2 byte + 어댑터 | **~31B dense**, 35B-A3B MoE |
| QLoRA, 4bit 베이스 | ~0.6 byte + 어댑터 | **~120B** (Qwen3.5-122B-A10B 가능) |
| 추론 4bit | ~0.55 byte | **~180B** (Flash-Next 경계) |

속도는 대역폭 546 GB/s가 정한다. 디코드 tok/s 상한 ≈ 546 ÷ (토큰당 읽는 바이트). 27B 4bit(15 GB)는 이론상 36 tok/s, 35B-A3B는 활성 3B만 읽어 150 tok/s대. **Mac에서는 "총 파라미터"보다 "활성 파라미터"가 속도를 정한다.** MoE 소활성 모델(Qwen3.6-35B-A3B, Gemma 4 26B-A4B, LFM2.5-8B-A1B)이 유리한 이유다.

## 우선순위 제안

1. **기준선(T1)**: Qwen3.5 2B/4B/9B + Gemma 4 E2B/E4B + Studio 등급 4종(Qwen3.8-27B, Qwen3.6-35B-A3B, Gemma 4 12B/26B-A4B).
2. **디코드 가속(T4b)**: mlx-dspark로 Gemma 4 12B·Qwen3.8-27B 배속 재현. 2026년에 가장 값싼 2~4×.
3. **저비트(A)**: Bonsai-27B ternary/1bit vs Qwen3.8-27B 4bit. 같은 27B가 3.9 GB에서 어디까지 버티나.
4. **학습(A)**: Qwen3.5-2B-Base 또는 Gemma 4 E2B(pt)로 LoRA → 풀 FT 순서. 학습 tok/s를 기준선 표에 추가.
5. **이미지(C)**: Z-Image 6B LoRA(mflux) → FLUX.2 Klein 4B LoRA. 그다음 minRF 처음부터 학습 → 브라우저.
6. **음성**: Qwen3-TTS 0.6B 목소리 파인튜닝, Kokoro 브라우저 배포.
