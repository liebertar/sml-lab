# TASKS — Mac Studio M4 Max 128GB에서 켜자마자

위에서부터 순서대로. 각 태스크는 **명령 → 뭐로 도나 → 예상 시간 → 확인 방법 → 성공 기준 → 기록** 순서다.
예상 시간은 M4 Max 기준이고, 첫 실행은 모델 다운로드가 대부분이다. 끝난 태스크는 체크박스를 채운다.

모니터링은 터미널을 하나 더 열어 `mactop`을 띄워 둔다. GPU 사용률, GPU 전력, 메모리 압력, 온도를 1초 단위로 보여준다.
GUI가 편하면 활성 상태 보기 → 윈도우 → GPU 기록. 다운로드 진행은 하네스가 자체 진행 막대로 보여준다.

---

## T0. 준비 (약 5분)

- [ ] `brew install mactop` — 1분
- [ ] `git clone https://github.com/liebertar/sml-lab && cd sml-lab && make env`
  - 도는 것: uv가 mlx 0.32.2, mlx-lm 0.31.3, onnxruntime, torch 2.14(MPS)를 잠긴 버전으로 설치
  - 시간: 3~5분 (torch 휠이 큼)
  - 확인: 마지막 줄에 패키지 목록이 찍히고 에러가 없다
- [ ] `make check`
  - 시간: 10초
  - 성공 기준: `chip : Apple M4 Max`, `memory : 128 GB`, `mlx : 0.32.2 on Device(gpu, 0)`, `metal device ... max working set` 이 100 GB 안팎, `torch ... mps=True`
- [ ] `make test` — 3개 통과

---

## T1. 기준선 8종 측정 → 실험 00 완료 (첫 실행 12~20분, 재실행 3분)

- [ ] 두 번째 터미널에서 `mactop` 실행
- [ ] `make baseline`
  - 도는 것: `experiments/00-env-baseline/run.py`. mlx-lm으로 모델 8종을 차례로 적재해 워밍업 1회 후 3회 측정, 중앙값 기록
  - 모델과 다운로드 크기 (합계 21.9 GB):

    | 모델 | 크기 | 예상 디코드 tok/s (M4 Max, 4bit) |
    |---|---|---|
    | Qwen3-0.6B-4bit | 0.33 GB | 300~450 |
    | Qwen3.5-2B-MLX-4bit | 1.63 GB | 120~180 |
    | Qwen3.5-4B-MLX-4bit | 2.85 GB | 70~100 |
    | Qwen3.5-9B-MLX-4bit | 5.57 GB | 30~45 |
    | gemma-4-e2b-it-4bit | 3.34 GB | 80~120 |
    | gemma-4-e4b-it-4bit | 4.82 GB | 50~80 |
    | Llama-3.2-3B-Instruct-4bit | 1.70 GB | 90~130 |
    | SmolLM3-3B-4bit | 1.63 GB | 90~130 |

  - 시간: 다운로드 5~12분(회선 속도에 따라) + 측정 2~3분. 재실행은 3분 안쪽
  - 확인: 모델마다 `→ 모델명` 다음 줄에 결과 행이 한 줄씩 찍힌다. mactop에서 측정 중 GPU 사용률이 90% 이상, 메모리는 최대 6~7 GB만 더 쓴다
  - 성공 기준: 8행 전부 나오고 `skipped`가 없다. 디코드 tok/s가 위 예상 범위에 들어온다. 크게 벗어나면 mactop의 GPU 전력이 낮게 유지되는지(스로틀) 먼저 본다
  - 기록: 출력 표를 `experiments/00-env-baseline/README.md`의 "Mac Studio M4 Max 128GB" 자리에 붙이고, 노트북 행과 비율(프리필·디코드)을 "배운 것"에 한 줄 적는다
- [ ] 커밋: `git add -A && git commit -m "docs(baseline): record M4 Max results"` 후 push

### T1b. Studio 등급 4종 (다운로드 58.6 GB, 첫 실행 20~35분, 재실행 5분)

128GB에서만 의미 있는 큰 모델. 왜 이 넷인지는 [docs/models-2026H2.md](docs/models-2026H2.md).

- [ ] `make baseline ARGS="--preset studio --out outputs/baseline-studio.json"`

    | 모델 | 크기 | 예상 디코드 tok/s |
    |---|---|---|
    | Qwen3.8-27B-4bit (2026-08, dense) | 15.0 GB | 25~36 |
    | Qwen3.6-35B-A3B-4bit (MoE, 활성 3B) | 19.0 GB | 100~150 |
    | gemma-4-12B-it-qat-4bit | 10.3 GB | 40~55 |
    | gemma-4-26b-a4b-it-4bit (MoE, 활성 4B) | 14.3 GB | 80~120 |

  - 확인: mactop 메모리가 최대 20 GB 안팎까지만 오른다. MoE 둘이 dense 27B보다 3~4배 빠르면 "활성 파라미터가 속도를 정한다"는 가설 확인
  - 기록: 같은 README 표에 4행 추가

---

## T2. KV 캐시 양자화 (약 5분)

긴 생성에서 KV 캐시가 메모리를 얼마나 먹고, 4bit로 줄이면 속도와 출력이 어떻게 달라지는지.

- [ ] 기준
  ```bash
  uv run mlx_lm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit --max-tokens 2048 --temp 0 \
    --prompt "Write a detailed, sectioned engineering guide to memory bandwidth limits in LLM decoding."
  ```
- [ ] KV 4bit
  ```bash
  uv run mlx_lm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit --max-tokens 2048 --temp 0 --kv-bits 4 \
    --prompt "Write a detailed, sectioned engineering guide to memory bandwidth limits in LLM decoding."
  ```
  - 도는 것: mlx-lm 생성기. 마지막에 `Prompt: N tokens, X tokens-per-sec`, `Generation: ...`, `Peak memory: ... GB` 세 줄이 찍힌다
  - 시간: 각 1~2분 (2048 tok ÷ 35 tok/s)
  - 성공 기준: Peak memory가 눈에 띄게 줄고(수백 MB 이상), 디코드 tok/s는 같거나 오르고, 앞 200토큰의 문장이 거의 같다
  - 기록: 두 줄을 `experiments/04-slm-serving/README.md`(새로 만든다, `_template` 복사)의 결과 표에

---

## T3. 프롬프트 캐시로 TTFT 줄이기 (약 3분)

시스템 프롬프트나 긴 문서를 매번 다시 인코딩하지 않는다.

- [ ] 캐시 생성
  ```bash
  uv run mlx_lm.cache_prompt --model mlx-community/Qwen3.5-9B-MLX-4bit \
    --prompt-cache-file outputs/hardware.cache --prompt "$(cat docs/hardware.md README.md)"
  ```
- [ ] 캐시 사용 vs 미사용
  ```bash
  uv run mlx_lm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit --prompt-cache-file outputs/hardware.cache \
    --max-tokens 64 --prompt "위 문서를 세 줄로 요약해."
  uv run mlx_lm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit \
    --max-tokens 64 --prompt "$(cat docs/hardware.md README.md) 위 문서를 세 줄로 요약해."
  ```
  - 시간: 캐시 생성 30초, 비교 각 30초
  - 성공 기준: 캐시 사용 시 `Prompt:` 줄의 토큰 수가 거의 0이고 첫 토큰이 즉시 나온다
  - 기록: TTFT 두 값을 04 README에

---

## T4. speculative decoding (약 5분)

9B가 답을 쓰고 2B가 초안을 대는 구조. 같은 토크나이저(Qwen3.5 계열)라 가능하다.

- [ ] 
  ```bash
  uv run mlx_lm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit --max-tokens 512 --temp 0 \
    --prompt "Explain how a KV cache works, with a small worked example."
  uv run mlx_lm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit --max-tokens 512 --temp 0 \
    --draft-model mlx-community/Qwen3.5-2B-MLX-4bit --num-draft-tokens 4 \
    --prompt "Explain how a KV cache works, with a small worked example."
  ```
  - 시간: 각 20~40초
  - 확인: mactop에서 두 모델이 동시에 메모리에 올라간다(약 7 GB)
  - 성공 기준: `Generation:` tok/s가 1.3배 이상 오르고 temp 0이라 출력 텍스트가 동일하다. `--num-draft-tokens`를 2·4·8로 바꿔 최적을 찾는다
  - 기록: 세 값(초안 없음 / 2 / 4 / 8)을 04 README 표에

### T4b. DSpark/DFlash 블록 드래프터 (약 10분)

2026년 방식. 소형 모델 대신 타깃의 은닉 상태를 읽는 드래프터가 여러 토큰을 한 번에 낸다. 무손실.

- [ ] `uv tool install mlx-dspark` 또는 `uv add mlx-dspark` (README의 설치 방법을 따른다)
- [ ] Gemma 4 12B로 기준 vs DSpark
  - 도는 것: mlx-dspark CLI. 지원 타깃은 Gemma 4, Qwen3/3.6/3.8, LFM2.5, Muse-Glimmer, Ornith, Nemotron, Bonsai
  - 시간: 드래프터 다운로드 1~2분 + 실행 각 1분
  - 성공 기준: 채팅 프롬프트에서 2× 이상, 수학·코드 프롬프트에서 2.5× 이상 (저자 측정 2.6~3.1×). temp 0에서 출력 동일
  - 기록: T4의 `--draft-model` 결과와 같은 표에. "소형 드래프트 vs 블록 드래프터" 비교가 04 실험의 핵심 표

---

## T5. 배치 스케일링 (약 3분)

디코드가 대역폭에 묶여 있다면 배치를 키워도 시간이 거의 안 늘어야 한다. 서버 동시 요청 설계의 근거가 된다.

- [ ] 
  ```bash
  for b in 1 4 8; do uv run mlx_lm.benchmark --model mlx-community/Qwen3.5-4B-MLX-4bit -p 512 -g 128 -b $b -n 3; done
  ```
  - 도는 것: mlx-lm 내장 벤치마크. 프롬프트 512 tok, 생성 128 tok, 배치 1·4·8
  - 시간: 합계 2~3분
  - 성공 기준: 배치 8의 총 처리량(tok/s × 8)이 배치 1의 5배 이상. 이보다 낮으면 어디서 꺾이는지 `-b 2 16`을 추가해 본다
  - 기록: 04 README 표

---

## T6. 로컬 서버 + 동시 요청 (약 5분)

- [ ] 서버
  ```bash
  uv run mlx_lm.server --model mlx-community/Qwen3.5-4B-MLX-4bit --port 8080
  ```
- [ ] 다른 터미널에서 동시 4요청
  ```bash
  for i in 1 2 3 4; do curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"Count from 1 to 50 in words."}],"max_tokens":200}' \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["usage"])' & done; wait
  ```
  - 시간: 1분
  - 확인: 서버 로그에 요청 4개, mactop GPU 사용률
  - 성공 기준: 4개 모두 응답. 응답 시간이 단일 요청의 4배에 가까우면 서버가 직렬 처리 중이라는 뜻이고, 그 자체가 04 실험의 첫 발견이다
  - 기록: 04 README

---

## T7. 양자화 스윕 시작 → 실험 02 (약 15~25분)

같은 2B를 affine 4bit와 mxfp4로 직접 변환해 perplexity를 비교한다. 이후 스윕 스크립트의 씨앗이다.

- [ ] 변환 (원본 bf16 약 4 GB 다운로드 포함)
  ```bash
  uv run mlx_lm.convert --hf-path Qwen/Qwen3.5-2B -q --q-bits 4 --q-group-size 64 --q-mode affine --mlx-path models/qwen3.5-2b-affine4
  uv run mlx_lm.convert --hf-path Qwen/Qwen3.5-2B -q --q-bits 4 --q-group-size 32 --q-mode mxfp4 --mlx-path models/qwen3.5-2b-mxfp4
  ```
  - 시간: 다운로드 2~5분 + 변환 각 1~2분
- [ ] 품질
  ```bash
  for m in models/qwen3.5-2b-affine4 models/qwen3.5-2b-mxfp4 mlx-community/Qwen3.5-2B-MLX-4bit; do
    uv run mlx_lm.perplexity --model $m --num-samples 64 --sequence-length 1024; done
  ```
  - 시간: 각 1~2분
- [ ] 속도: 같은 세 모델을 `make baseline ARGS="--models models/qwen3.5-2b-affine4,models/qwen3.5-2b-mxfp4"`
  - 성공 기준: perplexity·디코드 tok/s·디스크 크기 세 축이 표로 나온다. mxfp4가 빠르지만 perplexity가 높다면 그 폭을 적는다
  - 기록: `experiments/02-quant-sweep-mlx/README.md` 생성. 이 표가 이후 `mlx_lm.dwq`(증류 보정, 2B 기준 30~60분)와 `mlx_lm.dynamic_quant`(혼합 비트) 비교의 기준이 된다

---

## T8. 이미지 생성 기준선 (약 15분, 다운로드 ~10 GB)

텍스트와 같은 방식으로 이미지 모델도 숫자를 먼저 잰다. 학습 가능한 두 모델만.

- [ ] `uv tool install --upgrade mflux` (MLX 네이티브, MIT)
- [ ] Z-Image Turbo
  ```bash
  mflux-generate-z-image-turbo --prompt "a product photo of a ceramic mug on a wooden desk, soft light" \
    --width 1024 --height 1024 --steps 8 --seed 1 --output outputs/zimage.png
  ```
- [ ] FLUX.2 Klein 4B (4bit)
  ```bash
  mflux-generate-flux2 --model flux2-klein-4b --quantize 4 --prompt "a product photo of a ceramic mug on a wooden desk, soft light" \
    --width 1024 --height 1024 --steps 4 --seed 1 --output outputs/klein4b.png
  ```
  - 시간: 첫 실행 다운로드 5~8분, 생성은 1024px 기준 Z-Image 15~30초, Klein 4B 20~40초 예상
  - 확인: mactop GPU 사용률, 최대 메모리(Klein 4B 4bit ~8 GB)
  - 성공 기준: 두 장 다 생성되고 초당 스텝 수가 로그에 찍힌다
  - 기록: `experiments/07-image-baseline/README.md` 생성. 이 둘이 이후 LoRA 학습 대상이다 (mflux는 Z-Image·FLUX.2 학습을 지원, base 모델 사용)

## 다음에 열 것

- 01 minrf-web: `third_party/minRF/rf.py`를 MPS로 MNIST 학습(10~20분) → ONNX export → 브라우저. Safari 26·Chrome에서 WebGPU 확인.
- 03 metal-kernel: `mx.compile` 전후 비교부터. 커스텀 커널은 RMSNorm 하나로 시작.
- 05 tinygrad-metal-beam: `uv add tinygrad` 후 `METAL=1 BEAM=2`로 matmul 하나 탐색.

## 문제가 생기면

- 다운로드가 멈춤: `~/.cache/huggingface/hub`의 해당 모델 폴더를 지우고 다시. `HF_HUB_ENABLE_HF_TRANSFER=1`은 쓰지 않는다(의존성 없음).
- 메모리 압력이 노랑/빨강: 9B + draft 조합에서도 10 GB 안쪽이어야 정상. 다른 앱(브라우저 탭)이 원인일 때가 많다.
- tok/s가 예상의 절반: mactop의 GPU 전력이 20 W 아래에 머물면 저전력 모드다. 시스템 설정 → 배터리/에너지에서 확인.
