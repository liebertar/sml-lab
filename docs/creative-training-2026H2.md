# M4 Max 128GB에서 직접 학습해서 만드는 것

기준일 2026-09-08. 관점 8개(본인 데이터 / 패키징 / 블로그 / 한글·도시 / 도구 / 3D·공간 / 모델 자체 / 움직임)에서
"이 기계에서 학습하거나 적응시킬 수 있는" 컨셉 48개를 뽑았다. 원안은 [research/creative-concepts.json](research/creative-concepts.json)에 있다.
아래 판정은 **학습 도구가 Apple Silicon에서 실제로 존재하는지**를 리포에서 직접 확인해 다시 매긴 것이다.

큰 그림은 이렇다. 이 기계의 강점은 큰 모델을 돌리는 게 아니라 **작은 모델을 처음부터 끝까지 소유하는 것**이다.
그리고 좋은 아이디어의 공통점은 하나였다. **데이터를 만들 수 있느냐.**
폰트 150종은 한글 글리프 35만 쌍을 공짜로 준다. Blender에 이미 있는 병 3D 20개는 재질 라벨을 무한히 준다.
사진 앱의 즐겨찾기는 이미 취향 라벨이다. 차트는 렌더한 값이 곧 정답이다.
남이 못 만드는 데이터가 있으면 작은 모델이 이긴다.

---

## 0. 이 기계에서 실제로 학습되는 것

판정의 기준선이다. 리포를 직접 열어 확인했다.

| 도구 | 학습 대상 | 근거 | 라이선스 |
|---|---|---|---|
| `mflux-train` | **Z-Image 6B, FLUX.2 Klein base 4B/9B만.** 이미지 조건부(edit) 파인튜닝도 지원 | 모델 표의 Training 열이 Yes인 것은 이 둘뿐. 예제 config에 lora_layers·optimizer·checkpoint가 다 있다 | MIT |
| `mlx_vlm.lora` | Qwen2/3/3.5 VL, LLaVA, Deepseek-VL, Mllama 등. Gemma3n과 Qwen3 Omni는 제외 | `mlx_vlm/trainer/`에 lora·dora·sft·orpo 트레이너 실재. 그래디언트 체크포인팅과 누적 지원 | MIT |
| `mlx_lm.lora` | 텍스트 LLM. lora / dora / full | 이 저장소에 설치돼 있음 | MIT |
| `ltx-2-mlx` `packages/ltx-trainer` | LTX-2 비디오 LoRA. flow matching, T2V와 V2V | `configs/lora_t2v.yaml`, `lora_v2v.yaml` 실재 | MIT |
| PyTorch 2.14 MPS | 처음부터 만드는 소형 모델 전부 | 설치 확인, `mps=True` | BSD |
| `metal-gauss` | 3D 가우시안 스플랫. **PyTorch MPS 기반**이라 커스텀 헤드를 붙일 수 있다 | Metal 커널 런타임 컴파일, Xcode 불필요. 1분 예산에서 PSNR 24.7로 Brush 14.4를 앞선다 | MIT |
| `ai-toolkit` `run_mac.zsh` | 디퓨전 LoRA (실험적) | Apple Silicon 실험 지원 명시 | MIT |

**여기서 갈리는 것**

- **Qwen-Image-Edit-2511은 학습 대상이 아니다.** mflux 모델 표에서 Training이 "No"다. 추론과 데이터 생성(교사)에만 쓴다.
  편집 LoRA를 학습하려면 **FLUX.2 Klein base**로 간다. mflux가 Klein의 이미지 조건부 파인튜닝을 지원한다.
- **SANA-WM은 학습 경로가 없다.** MLX 포트는 추론 전용이고, 원본 학습은 8-GPU 분산에 Triton·xformers·flash-attn을 쓴다.
- **TripoSplat은 학습 코드가 공개돼 있지 않다.** 추론만 있다.
- **RF-DETR의 MPS 학습은 미검증이다.** README에 Apple 언급이 없다. 백본 특징을 캐시해 헤드만 학습하는 우회로가 안전하다.

---

## 1. 지금 바로 되는 것

도구가 검증됐고, 데이터를 스스로 만들 수 있고, 결과가 눈에 보이는 순서로 골랐다.

### 1) 손글씨 200자로 한글 11,172자 완성

한 장에 200자만 쓴다. 학습셋은 **무료 한글 폰트 150종**이다. 2,350자 × 150종이면 35만 쌍이 공짜로 나온다.
25M짜리 자모 조건부 DiT(초성·중성·종성 임베딩 + 참조 글리프의 스타일 벡터)를 PyTorch MPS로 8~10시간 학습한다.
방문자가 아무 문장이나 치면 **당신이 한 번도 써본 적 없는 글자까지** 당신 손글씨로 나온다.

왜 좋은가. 한국어라 유일하고, 데이터가 공짜이고, "쓴 적 없는 글자"라는 결과가 설명 없이 전달된다.
ONNX로 내보내 브라우저에서 돌린다.

**획 버전으로 한 단계 더** 갈 수 있다. 폰트 외곽선을 골격화해 획 순서로 정렬하면 4~6M짜리 획 시퀀스 모델이 된다.
8MB, ONNX Runtime Web에서 제목이 **획순대로 그려진다**. 새로고침할 때마다 미세하게 다르다. 학습 1~2시간.

### 2) 히어로가 이미지 파일이 아니라 가중치가 된다

책상에 아이폰을 고정하고 5분마다 한 장씩 4일에서 4주. (x, y, 시각, 날짜) → RGB를 예측하는 좌표망을 학습한다.
Instant-NGP식 해시그리드 + 작은 MLP, 25만~150만 파라미터, MLX로 25~40분, 메모리 4GB 미만.

결과물은 900KB 가중치 파일 하나다. 프래그먼트 셰이더가 픽셀마다 평가한다.
방문자가 24시간 링을 돌리면 **당신 책상의 빛이 실제로 움직인다.** 비디오 재생이 아니다.
`today` 모드를 붙이면 방문자의 현지 시각에 맞춰 지금 당신 창밖을 렌더한다.

### 3) 40K 파라미터 유체가 페이지를 젓는다

MLX로 2D Stable-Fluids 솔버를 짜서 교사로 쓴다. 강제력은 **당신이 블로그에서 실제로 움직인 마우스 궤적**이다.
3층 컨볼루션 40K 파라미터 학생을 5스텝 언롤로 증류하고, 가중치를 GLSL에 구워 넣는다.
커서가 페이지를 물처럼 젓고, 물리 전체가 읽을 수 있는 `.glsl` 한 장이다. 학습 2시간.

### 4) 사진 앱 즐겨찾기가 이미 취향 라벨이다

이게 가장 영리한 데이터 아이디어다. **라벨링 시간이 0이다.**
`osxphotos`로 사진 1만 2천 장을 내보내면 즐겨찾기 600장이 약한 양성이 된다.
결정적인 부분은 **hard negative**다. 즐겨찾기한 사진의 전후 10초 안에 찍힌 사진 중 즐겨찾기가 아닌 것.
피사체는 같고 선택만 다르니, 모델이 배우는 건 주제가 아니라 **취향**이다.

DINOv3 ViT-S 임베딩 1만 3천 장이 10분, Bradley-Terry 헤드 학습은 몇 초다. 브라우저용은 2MB 헤드 하나.
방문자가 사진 20장을 떨구면 실시간으로 "내 취향 순"으로 재정렬되고, 어느 부분이 점수를 끌어올렸는지 히트맵이 뜬다.

### 5) 취향 모델을 보상으로 쓰는 닫힌 RL 루프

위의 취향 모델을 보상 함수로 삼아 Klein 4B에 Diffusion-DPO LoRA를 돈다.
프롬프트 200개(블로그 글에서 채굴) × 후보 8장 × 4스텝 = 라운드당 1,600장을 1.2시간에 생성하고,
취향 모델이 점수를 매겨 승자·패자 쌍을 만들고, LoRA를 갱신한다.

메모리는 Klein 4B bf16 8GB + 텍스트 인코더 8GB + DPO 참조 사본 8GB + 활성화, 합쳐 45GB 정도다. 128GB에서 여유롭다.
같은 프롬프트·같은 시드로 라운드 0에서 5까지 격자를 만들면 **모델이 한 사람 쪽으로 이동하는 게 눈에 보인다.**
Mac 한 대에서 생성·평가·학습이 다 도는 완결된 루프이고, 엔지니어링으로서 자랑할 만하다.

### 6) 이미 가진 3D 자산으로 재질 다이얼

작업 폴더에 화장품 병 3D 컬렉션이 이미 있고 Blender도 설치돼 있다. 이걸 데이터 공장으로 쓴다.
헤드리스 Blender가 병 20개에 재질 프리셋 12종(투명 유리, 프로스티드, 브러시드 알루미늄, 소프트터치, 펄 PET 등)을 입혀
768px로 하룻밤에 렌더한다. EEVEE Next 기준 프레임당 3초.

그 쌍으로 **Klein 4B의 이미지 조건부 편집 LoRA**를 mflux로 학습한다. rank 16, 2,500 스텝, 7~9시간, 25~30GB.
방문자가 실제 샴푸병 사진 위에서 다이얼을 돌리면 형태와 라벨과 그림자는 그대로인 채 재질만 바뀐다.

### 7) 곡면 라벨을 펴고 다시 감는다

같은 병 3D로 라벨 사각형을 무작위 각도·높이로 감아 렌더하면 (사진 → UV 필드) 정답 쌍이 무한히 나온다.
ConvNeXt-Tiny 인코더 + 가벼운 디코더 30M이 픽셀마다 (u, v, 라벨 마스크, 실린더 축)을 예측한다. MPS로 3시간.

폰 카메라를 실제 병에 대면 **평평하게 펴진 라벨이 옆에 나타나고**, 새 PNG를 떨구면 실시간으로 감긴다.
ONNX fp16 60MB, 브라우저에서 512px 30~60ms. beu의 핵심 기능이 될 수 있다.

### 8) 합성 데이터로 정답이 공짜인 도구 세 개

- **화이트보드 → Mermaid**: Mermaid 그래프 4,000개를 무작위 생성해 손그림 느낌으로 렌더하면 (이미지 → 소스) 쌍이 공짜다.
  Qwen3.5-2B VLM LoRA를 mlx-vlm으로 3~4시간. 폰을 스케치에 대면 편집 가능한 다이어그램이 나온다.
- **차트 → CSV**: 차트 1만 2천 장을 렌더하면 값이 곧 정답이다. Grafana·Langfuse 다크 테마까지 흉내 낸다.
  슬랙에서 퍼온 캡처를 붙이면 표와 다시 그린 차트가 돌아온다. 6~7시간, 밤에 돌린다.
- **목마름 게이지**: 화분 5개 × 위치 3 × 하루 2장 × 35일이면 1,050장. 라벨은 캘린더의 물 준 기록이다.
  DINOv3 특징을 한 번 캐시하면(2분) MLP 헤드는 **몇 초 만에** 학습된다. 가장 싸고 가장 실용적이다.

### 9) 시간이 축인 스플랫

같은 책상을 아침·정오·해질녘·램프 네 번 찍는다. 기하는 한 번만 맞추고 얼려둔다.
그 위에 (가우시안 잠재, 시간) → 색을 예측하는 5만 파라미터 MLP를 얹는다.
metal-gauss가 PyTorch MPS 기반이라 이 훅을 붙일 수 있다. 1분 예산 PSNR에서 다른 구현을 앞선다.

6MB 파일 하나로 방문자가 **글이 쓰인 실제 책상을 궤도로 돌면서** 슬라이더로 아침을 저녁으로 민다.
캡처 12주 버전으로 확장하면 커피잔이 옮겨다니고 몬스테라가 잎을 펴는 "시간 축 스크러버"가 된다.

### 10) 5MB 히어로

이 저장소의 01번 실험(minrf-web)을 끝까지 밀어붙인 형태다.
30M짜리 픽셀 공간 DiT를 처음부터 학습 → MeanFlow 1스텝 목적함수로 미세조정 → 삼진(ternary) QAT → 브라우저.
데이터는 **당신 블로그의 기존 히어로 장면을 Playwright로 2만 장 렌더한 것**이다. 데이터셋 자체가 당신 것이다.

MLX 배치 128, 128px에서 초당 3.5 스텝, RF 10만 스텝이 8시간. 증류 2시간, QAT 2시간.
방문자가 폰트가 다 로드되기도 전에 **자기 탭에서 한 번도 존재한 적 없는 이미지**를 받는다.
배지에는 "30M 파라미터 · 1 스텝"이라고 적는다.

---

## 2. 조건부로 되는 것

| 컨셉 | 무엇이 걸리나 | 수정안 |
|---|---|---|
| 방 조명을 배운 합성 LoRA | Qwen-Image-Edit은 mflux 학습 대상이 아니다 | 교사로만 쓰고 학생은 Klein 4B edit LoRA. Blender 렌더 1,500쌍이면 충분 |
| 제품 회전 비디오 LoRA | ltx-2-mlx 트레이너는 실재하나 60~90GB 주장이 빠듯 | 512px 49프레임에서 시작해 메모리를 먼저 재고, 안 되면 Wan 5B로 |
| 간판 조사 / 냅킨 → 컴포넌트 / 내 목소리 비평가 | mlx-vlm LoRA는 되지만 **라벨 품질이 전부**다 | 자동 라벨을 초안으로만 쓰고 600장 정도는 손으로 고친다 |
| 손글씨 OCR / 법정 표기 검사 | GLM-OCR이 mlx-vlm 지원 목록에 없다 | 지원 확인 후 진행. 안 되면 Qwen3.5-2B VLM으로 같은 데이터를 쓴다 |
| 동글 서랍 / 바둑판 기보 | RF-DETR의 MPS 학습이 미검증 | 백본 특징을 캐시하고 헤드만 학습. 원안에도 폴백으로 적혀 있다 |
| 걷는 길 스플랫 + 위치 아는 VLM | 스플랫과 LoRA는 되지만 ONNX 재수출이 관건 | Mac 로컬 데모로 먼저 만들고 브라우저는 나중에 |

## 3. 전제가 틀린 것

| 컨셉 | 이유 |
|---|---|
| 아파트 월드 모델 (SANA-WM LoRA) | SANA-WM에 Apple Silicon 학습 경로가 없다. MLX 포트는 추론 전용. 원본 학습은 8-GPU + Triton + flash-attn |
| 병 전용 TripoSplat 파인튜닝 | TripoSplat이 학습 코드를 공개하지 않았다 |
| OmniSVG로 손글씨 SVG 생성 | SVG 토크나이저가 커스텀이라 MPS 학습 경로가 검증되지 않았다. 획 시퀀스 모델(1절 1번)이 같은 결과를 더 싸게 낸다 |

**월드 모델은 버리지 말고 방향만 바꾼다.** 남의 월드 모델을 내 아파트로 미세조정하는 대신,
**우리 집만 본 적 있는 작은 월드 모델을 직접 학습한다.** 스플랫에서 카메라 궤적 20만 프레임을 렌더해
100M짜리 DiT를 MLX로 밤새 돌리면 된다. 브라우저에 올릴 수 있는 크기이고, 이 저장소의 방향(갈래 C)과도 맞는다.

---

## 4. 로드맵에 넣을 것

| # | 실험 | 무엇을 학습 | 예상 |
|---|---|---|---|
| 18 | hangul-hand | 자모 조건부 글리프 DiT 25M + 획 모델 4M. 폰트 150종이 학습셋 | MPS 8~10h, ONNX 20MB |
| 19 | window-field | (x, y, 시각, 날짜) → RGB 좌표망 250k~1.5M | MLX 40분, 900KB |
| 20 | favorites-model | 사진 앱 즐겨찾기로 취향 보상 모델. DINOv3 + BT 헤드 | 임베딩 10분, 헤드 몇 초 |
| 21 | taste-grpo | 20번을 보상으로 Klein 4B DPO 루프 | 라운드당 1.2h + 학습, 45GB |
| 22 | material-dial | 보유 병 3D + Blender 렌더로 Klein 4B edit LoRA | 렌더 5h, 학습 7~9h |
| 23 | label-unwrap | 곡면 라벨 → UV 필드 30M | 렌더 5h, MPS 3h, ONNX 60MB |
| 24 | tiny-tools | 화이트보드→Mermaid, 차트→CSV, 목마름 게이지 | 각 3~7h, 합성 데이터 |
| 25 | desk-chronosplat | metal-gauss 기하 + 시간 조건 색 MLX 50k | 캡처 4회, 학습 2h, 6MB |

01번(minrf-web)이 끝나면 **5MB 히어로**로 확장한다. 별도 실험 번호를 주지 않고 01의 마지막 단계로 둔다.

---

## 출처

- [mflux 학습](https://github.com/filipstrand/mflux/tree/main/src/mflux/models/common/training) · [mlx-vlm LoRA](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/LORA.MD) · [ltx-2-mlx trainer](https://github.com/dgrauet/ltx-2-mlx/tree/main/packages/ltx-trainer) · [ai-toolkit](https://github.com/ostris/ai-toolkit)
- [metal-gauss](https://github.com/nandometzger/metal-gauss) · [Brush](https://github.com/ArthurBrussee/brush) · [splat-local](https://github.com/michael-L-i/splat-local)
- [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [FLUX.2 Klein base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) · [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) · [DINOv3 ONNX](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX)
- [UnifiedReward](https://github.com/CodeGoat24/UnifiedReward) · [CapRL](https://github.com/InternLM/CapRL) · [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) · [RF-DETR](https://github.com/roboflow/rf-detr)
- [osxphotos](https://github.com/RhetTbull/osxphotos) · [MoGe](https://github.com/microsoft/MoGe) · [Marigold](https://github.com/prs-eth/Marigold)
