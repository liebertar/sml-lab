# 2026년 하반기, "참조 편집 모델로 옷 갈아입히기"보다 앞선 것들

기준일 2026-09-07. 가상 피팅(VTO)과 그 주변에서 올해 나온 것 중 **공개 가중치·코드가 있고 라이선스가 확인된 것**만 적었다.
"Mac" 열은 M4 Max 128GB에서 돌릴 수 있는지에 대한 판단이다. 검증 전이면 "확인 필요"로 둔다.

## 요약

1. **범용 참조 편집 모델(Klein 계열)을 VTO로 쓰는 방식은 2026년 기준 주류가 맞다.** VTEdit-Bench(ECCV'26)가 24,220쌍으로 검증한 결론: 범용 편집 모델은 표준 시나리오에서 전용 모델과 대등하고, 다각도·다중 인물 같은 어려운 설정에서 더 안정적이다. 약점은 **여러 벌 동시 착용과 의류·인물 일관성**.
2. 그 위에 얹을 수 있는 것이 이미 공개돼 있다. Klein 4B/9B용 **try-on LoRA**(Apache-2.0, 88 MB)는 mflux로 바로 로드된다.
3. 전용 모델 쪽의 신호는 **픽셀 공간(VAE 없음)** 이다. FASHN VTON 1.5(972M, 마스크 불필요, Apache-2.0), HiDream-O1(8B 통합 픽셀 트랜스포머, MIT), Pixel MeanFlow(1스텝, 잠재공간 없음). 잠재공간의 손실이 옷감 디테일을 망친다는 문제의식이 공통이다.
4. 상용 최전선은 **실시간 비디오**(Decart Lucy 2 VTON, 30fps, 40ms 미만)와 **셀카 한 장 → 전신 아바타**(Google, Nano Banana)로 갔다. 공개 가중치로는 Vanast(단일 사진 + 옷 → 애니메이션), Wan-Animate-2(14B)가 가장 가깝다.
5. **3D 신체 파라미터**(SAM 3D Body → MHR, Anny, GNM)가 전부 Apache-2.0으로 풀렸다. "맞는 사이즈인가"까지 답하려면 이 층이 필요하고, FIT 데이터셋(SIGGRAPH'26)이 그 방향을 연다.

## A. 2D 가상 피팅

| 항목 | 출시 | 크기 | 라이선스 | 무엇이 새로운가 | Mac |
|---|---|---|---|---|---|
| **tryon-FLUX.2-klein-4B-lora** (xocialize) | 2026-07 | LoRA 88 MB, rank 32 | Apache-2.0 | Klein 4B 편집 LoRA. 입력 3장 순서 고정: 옷 영역을 지운 인물, 상의, 하의. 트리거 `TRYON ...`. base-4B로 ai-toolkit 학습, 273 튜플 | **mflux `--lora-paths`로 바로** |
| flux-klein-9b-virtual-tryon-lora (fal) | 2026-02 | LoRA | 확인 필요 | 9B 편집 LoRA, 같은 3장 입력. 다운로드 5.6K | 9B는 비상업 라이선스 |
| **FASHN VTON 1.5** | 2026-01 | 972M | Apache-2.0 | 픽셀 공간 디퓨전, 세그멘테이션 마스크 불필요. H100 ~5초. 상용 서비스에서 쓰던 모델을 공개 | 크기상 가능. 포즈 검출이 onnxruntime-gpu → CPU/CoreML로 교체 필요 |
| **HiDream-O1-Image** | 2026-05 | 8B | MIT | VAE·별도 텍스트 인코더 없이 픽셀·텍스트·조건을 한 토큰 공간에. T2I·편집·주체 보존 통합. README에 try-on 프롬프트 예시. FP8 ~10 GB | PyTorch. CUDA 전제라 MPS 패치 필요 |
| Oxygen-TryOn | 2026-07 | (Qwen3-VL-8B + DiT) | Apache-2.0 | "패션 네이티브" 기초 모델. 아이템 수 제한 없음, 포즈 변경 같은 편집을 같은 패스에서 | 가중치 미공개 |
| Tstars-Tryon 1.0 (Taobao) | 2026-04 | 비공개 | — | 참조 6장, 8개 카테고리, 3.9~6.7초. Nano Banana Pro·GPT-Image-2보다 낫다고 보고. 벤치마크만 공개 | 불가 |
| ChordEdit | CVPR'26 oral | SD-Turbo 기반 | MIT | 1스텝 편집(low-energy transport) | 가능 |

벤치마크·데이터셋(우리 회귀 세트의 원천):
- **VTEdit-Bench** (ECCV'26, 24,220쌍, 5개 난이도, VLM 기반 채점 VTEdit-QA)
- **OpenVTON-Bench** (2026-01, ~100K쌍, 최대 1536², 5축 채점: 배경·정체성·질감·형태·사실감, HF 데이터셋 공개)
- **Garments2Look** (CVPR'26, 80K 코디 단위, 참조 3~12장)
- **FIT** (SIGGRAPH'26, 핏 인지 VTO, 100K 프리뷰)
- MV-Fashion (CVPR'26, 다시점, 연구용 라이선스), TripVVT (비디오 삼중항)

## B. 비디오·실시간

| 항목 | 출시 | 라이선스 | 내용 | Mac |
|---|---|---|---|---|
| Decart Lucy 2 VTON Realtime | 2026 | API 전용 | 웹캠 라이브 영상에 옷을 입힘. 30fps, 40ms 미만, WebRTC | 불가(폐쇄) |
| **Vanast** | CVPR'26 highlight | CC BY 4.0 | 인물 1장 + 옷 + 포즈 영상 → 옷 갈아입힌 애니메이션. "돌려보기"에 가장 가까운 공개 연구 | 가중치 2026-05 예고, 확인 필요 |
| iTryOn | ICML'26 | 확인 필요 | Wan2.1-VACE 위 대화형 비디오 VTO | 무거움 |
| **Wan-Animate-2 14B** | 2026-08 | Apache-2.0 | 구동 영상을 DiT가 직접 소비. 정체성 드리프트 개선. 피팅 결과 1장을 돌려보기 영상으로 | 14B, H100 대여 |
| Helios / Causal Forcing / minWM | 2026-02~05 | Apache-2.0 | 실시간 대화형 비디오 월드 모델. minWM은 Wan2.1-1.3B부터 레시피 공개 | 1.3B 레시피는 확인 필요 |
| AnyFlow (NVlabs) | 2026-03 | Apache-2.0 | 임의 스텝 비디오 증류(flow map) | 학습은 GPU |

## C. 3D 신체 (사이즈·핏)

| 항목 | 라이선스 | 내용 | Mac |
|---|---|---|---|
| SAM 3D Body (CVPR'26) + Fast SAM 3D Body (ECCV'26) | SAM 라이선스(게이트) | 사진 1장 → 전신 메시. Fast 버전 10.9× 실시간. ComfyUI 코어 내장(2026-08) | DINOv3 인코더, MPS 확인 필요 |
| **MHR** (Meta) | Apache-2.0 | 골격과 표면을 분리한 파라메트릭 인체. `pip install pymomentum-cpu` | **CPU로 됨** |
| **Anny** (Naver) | Apache-2.0(코드) | 전 연령 해석 가능한 인체 모델, PyTorch | 됨 |
| **GNM Head** (Google) | Apache-2.0 | 파라메트릭 두부 모델. macOS CI 있음 | 됨 |

## D. 통합 소형 모델

| 항목 | 출시 | 크기 | 라이선스 | 내용 | Mac |
|---|---|---|---|---|---|
| **Lance** (ByteDance) | 2026-05 | 활성 3B MoE | Apache-2.0 | 이미지+비디오 이해·생성·편집 한 모델. 768² / 480p 12fps. 파인튜닝 코드 공개(06-17) | 크기는 맞음, MPS 확인 필요 |
| HiDream-O1-Image | 2026-05 | 8B | MIT | 위 A 참조 | |
| Qwen-Image-2.0 | 2026-02 | 7B | — | 생성+편집 통합, AI Arena 1위. **가중치 비공개**(API만) | 불가 |
| GLM-Image | 2026-01 | — | Apache-2.0 | 자기회귀 이미지 생성 | 80 GB VRAM 요구 |

## E. 생성 효율 (우리 A 갈래와 직결)

| 항목 | 출시 | 라이선스 | 핵심 |
|---|---|---|---|
| **Pixel MeanFlow** | 2026-01 | MIT (JAX) | 잠재공간 없이 픽셀에서 1스텝. ImageNet 256² FID 2.22 |
| Self-Flow (BFL) | ICML'26 | Apache-2.0 | 자기지도 flow matching으로 멀티모달 합성 확장 |
| AnyFlow | 2026-03 | Apache-2.0 | 하나의 모델이 임의 스텝 수에 적응 |
| Causal Forcing++ | 2026-05 | Apache-2.0 | 자기회귀 디퓨전 증류의 "올바른" 초기화 |

## F. 폐쇄 진영이 어디까지 갔나 (목표선)

- Google: 셀카 1장 → 스튜디오 전신 아바타 생성 후 피팅(2025-12, Nano Banana). Doppl은 쇼핑 피드에 AI 영상.
- Taobao Tstars-Tryon: 수천만 요청 규모 운영, 참조 6장, 6.7초.
- Decart: 실시간 30fps 비디오 피팅.
- OpenAI GPT-Image-2, Google Nano Banana Pro: 범용 편집기로서 VTO 벤치마크의 비교 대상.

## 우리 로드맵에 넣을 것 (Mac에서 되는 순서)

| # | 실험 | 왜 |
|---|---|---|
| 08c | Klein 4B + 공개 try-on LoRA | 프롬프트만으로 안 되는 부분이 LoRA 88 MB로 얼마나 메워지는지. 즉시 가능 |
| 09 | FASHN VTON 1.5 (픽셀 공간, 972M) | 전용 모델 vs 범용 편집 모델을 같은 회귀 세트로. 픽셀 공간의 디테일 이점 확인 |
| 10 | SAM 3D Body / MHR로 신체 파라미터 | 아바타 1장에서 체형 수치를 뽑아 "맞는 사이즈" 층의 씨앗 |
| 11 | Lance 3B 편집·짧은 영상 | 3B로 이미지 편집과 돌려보기 영상이 한 모델에서 되는지 |
| 12 | Pixel MeanFlow를 minRF 대신 | 1스텝·픽셀 공간 소형 생성 모델 → 브라우저(잠재공간 디코더 불필요) |
| 대여 | Vanast(가중치 공개 시), Wan-Animate-2 | 돌려보기 영상. H100 시간제 |

채점은 OpenVTON-Bench의 5축(배경·정체성·질감·형태·사실감)을 그대로 쓴다.
