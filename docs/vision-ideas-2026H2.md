# 옷 갈아입히기 말고, 비전 모델로 만들 수 있는 것 (2026년 하반기)

기준일 2026-09-08. 8개 도메인을 훑어 후보 64개를 모으고, 빠진 영역 3개를 더 찾아 총 74개를 검증했다.
검증 항목은 라이선스(SPDX 원문), 가중치 실재 여부, 파라미터 수, 그리고 **M4 Max 128GB 또는 브라우저에서 실제로 도는가**이다.
근거가 없으면 탈락시켰고, 가장 보수적인 등급을 매겼다. 원자료는 [research/vision-candidates.json](research/vision-candidates.json)에 있다.

큰 그림은 이렇다. 2026년의 재미있는 지점은 "더 큰 생성 모델"이 아니라 **기하와 이해가 공짜가 된 것**이다.
깊이·법선·분할·3D가 34M~800M 크기로 내려와 브라우저에서 돌고, 그 위에 20B급 편집 모델을 Mac이 8bit로 감당한다.
그래서 지금 만들 수 있는 것은 "이미지를 만든다"가 아니라 **사진 한 장을 만질 수 있는 표면으로 바꾸는 일**이다.
빛을 끌고, 라벨을 곡면에 감고, 클릭한 물체만 살아 움직이고, 병이 돌아간다. 이게 옷 갈아입히기보다 눈에 띈다.

| 실행 등급 | 개수 | 뜻 |
|---|---|---|
| browser | 17 | 방문자 브라우저에서 WebGPU/ONNX로 돌아가는 포트가 실재 |
| mac-native | 22 | MLX 포트 또는 공식 MPS 경로 확인 |
| mac-slow | 23 | CUDA 전용 커널은 없으나 Mac 경로가 검증되지 않음 |
| gpu-only | 12 | flash-attn, nvcc, 24GB+ VRAM 등으로 막힘 |

---

## 1. 지금 바로 만들 수 있는 것

### 1) 사진 한 장을 만질 수 있는 표면으로 (블로그 히어로)

**MoGe-2 WebGPU + 드래그 조명**
법선을 깊이에서 근사하지 않고 **법선 헤드가 직접 예측**한다. 그래서 곡면 유리에 림라이트와 부드러운 그림자가 제대로 붙는다.
330M, fp16 661MB, M4 Max에서 2.5초, 순수 WebGPU 컴퓨트 셰이더라 ONNX 런타임도 WASM도 필요 없다. MIT.
지금 블로그 히어로가 쓰는 깊이 기반 법선을 이걸로 갈아끼우고 커서를 광원으로 만들면 된다. Safari 26 확인은 필요하다.

**Depth Anything 3 Small**
34M, ONNX 105MB, Apache-2.0. 방문자가 사진을 떨구면 1초 안에 깊이가 나오고, ogl 평면을 변위시켜 2.5D 패럴랙스가 된다.
기존 LivingTitle의 밀기·들기 규칙과 그대로 붙는다. 다만 Transformers.js 파이프라인 호출이 후처리를 다 해주지는 않아 전처리·후처리를 직접 써야 한다.

**Marigold-IID (고유 이미지 분해)**
광택 병 사진을 알베도 / 확산 셰이딩 / 비확산 잔차로 쪼갠다. 866M, 공식 리포에 `--apple_silicon` 플래그와 MPS 분기가 있다.
분해 결과를 A × S' + k·R로 다시 합치면 라벨 색은 고정된 채 셰이딩만 실시간으로 움직인다. 코드 Apache-2.0, 가중치 OpenRAIL++.

### 2) 페이지가 보고 반응한다

**FastVLM-0.5B WebGPU**
759M, 공식 정적 Space가 서버 없이 돈다. 웹캠을 페이지가 초당 여러 번 설명한다.
"이 페이지가 당신을 봅니다" 한 줄과 함께 히어로에 캡션을 흘려보내면 방문자가 기억한다. 라이선스가 apple-amlr(연구 목적)이라 상업 사용은 SmolVLM2-256M으로 폴백해야 한다.

**EdgeTAM (온디바이스 SAM 2)**
13.9M, ONNX fp16 20MB. 웹캠에서 아무 물체나 클릭하면 실시간으로 마스크가 따라다닌다.
그 마스크를 ogl 포스트프로세스에 넘기면 "클릭한 물체만 색이 남고 배경은 생성 필드로 녹는다". Apache-2.0, Transformers.js에 클래스가 이미 등록돼 있다.

**SAM 3 텍스트 프롬프트 분할**
문장을 타이핑하면 해당하는 모든 물체가 마스크된다. beu 쪽에서 "cap", "label", "pump"를 타이핑해 부위별 마스크를 얻으면
디자인 오버레이가 병 전체가 아니라 **라벨 영역에만** 붙는다. 브라우저 INT8 포트가 있고, Mac에서는 MLX로 자동 라벨러를 쓸 수 있다. SAM 라이선스(상업 허용, 게이트).

**LFM2.5-VL-450M**
449M. 라이브 캡션에 더해 **바운딩 박스**를 준다. 손·얼굴 박스를 생성 필드의 인력점으로 쓰면 화면이 방문자 쪽으로 휜다.
450M이 브라우저에서 박스를 뱉는 건 2026년에 새로 생긴 일이다. 라이선스가 매출 1000만 달러 미만 조건부라 회사 프로젝트에는 주의.

### 3) 제품이 돌아간다 (beu)

**TripoSplat (MLX 포트)**
사진 한 장 → 3D 가우시안 스플랫. 추론 경로 1.58B, MLX 포트가 있고 코드는 MIT.
BiRefNet으로 누끼 → tripo_splat_mlx → .spz 압축 → Spark로 페이지에 심으면 드래그로 도는 실물 병이 된다.
주의: 번들된 FLUX.2 VAE가 비상업 표기인데, 검증 결과 Apache-2.0인 Klein-4B VAE와 **텐서 251개가 수치적으로 동일**하다. 상업용은 Klein-4B에서 VAE를 가져오면 깨끗하다.

**TRELLIS.2 (trellis-mac / mlx-spatial)**
사진 → 텍스처와 PBR 재질이 있는 GLB 메시. 4B, 가중치 MIT.
업스트림은 CUDA 전용(flash-attn, 24GB VRAM)이지만 Mac 포트가 셋 있다. 라벨 PNG를 베이스컬러로 갈아끼우면 3D 실물에 디자인을 입혀볼 수 있다.
배경 제거를 RMBG-2.0 대신 BiRefNet으로 바꾼 포트를 써야 비상업 조항을 피한다.

**metal-gauss / splat-local / Brush**
Mac에서 스플랫을 **직접 학습**한다. metal-gauss는 2026-08-31 첫 커밋으로 Metal 커널만 쓰고 Xcode도 CUDA도 필요 없다.
splat-local은 아이폰 영상을 100% 로컬로 걸어다닐 수 있는 씬으로 만든다. Brush는 크롬 탭 안에서 학습이 수렴하는 걸 보여준다.
실물 병을 턴테이블로 20초 찍어 스플랫으로 올리면 렌더가 아니라 진짜 물건이 페이지에서 돈다.

### 4) 20B 편집 모델이 Mac에서 돈다

**Qwen-Image-Edit-2511 + Lightning 4스텝**
20.4B, Apache-2.0. 8bit MLX 패키지 30.3GB이므로 128GB에 올라간다. mlx-gen과 mflux 둘 다 경로가 있다.
"이미지 2의 디자인을 이미지 1 병에 인쇄된 라벨처럼 감되 반사와 그림자는 유지하라" 한 문장이 그대로 먹힌다.
Relight LoRA(236MB, Apache-2.0)를 얹으면 같은 병으로 조명 4종 시트를 뽑는다. 2026년 9월 기준 Mac에서 돌릴 수 있는 가장 강한 공개 편집기다.

**Boogu-Image-0.1 Turbo**
DiT 10.3B에 Qwen3-VL-8B 지시 인코더까지 붙은 18B급. mflux에 공식 MLX 구현이 병합돼 있다(v0.19.1).
라벨에 **읽을 수 있는 제품명과 문구를 직접 렌더링**한다. 브랜드 브리프에서 라벨 시안 4~6장을 뽑는 용도.

**Z-Image-Turbo / SeedVR2-3B**
6.15B 프롬프트→제품샷, 3B 1스텝 복원. 둘 다 mflux 네이티브, Apache-2.0.
저해상 공급사 사진을 먼저 복원한 뒤 디자인을 얹는 전처리 라인으로 쓴다. 브라우저 쪽 대응물로 Swin2SR x4(15MB)를 슬라이더에 붙이면 대비가 확실하다.

### 5) 도구가 되는 것

**GLM-OCR (1.33B, MIT)**
공식 리포에 mlx-vlm 배포 예제가 있다. 전성분 라벨을 찍으면 브랜드·용량·성분 배열·주의문구를 JSON으로 뱉는다.
beu 디자인 브리프를 자동으로 채우고, 오버레이한 디자인의 법정 표기가 가려지지 않았는지 검사할 수 있다.

**RF-DETR-Seg (33M, Apache-2.0)**
증류 이야기의 완성형이다. SAM 3.1로 300장을 자동 라벨링 → RF-DETR-Seg-N을 MPS에서 학습 → ONNX로 내보내 브라우저에서 웹캠 실시간.
873M 교사에서 33M 학생으로 내려오는 전 과정이 한 주말에 들어가고, 결과물이 블로그에서 60fps로 돈다.

**DINOv3 ViT-S/16 (21.6M, q4 14.7MB)**
사진 폴더를 떨구면 PCA 특징 히트맵이 물체 부위를 보여주고 중복이 자동으로 묶인다.
beu 렌더 라이브러리 중복 제거와 "이 실루엣의 병 찾기" 검색에 그대로 쓴다.

**UnifiedReward-Flex 4B (MIT)**
생성 결과를 심사하는 VLM. 라벨 가독성·유리 사실감·환각 텍스트 같은 기준을 글로 적어주면 쌍대 비교로 순위를 매긴다.
생성 루프를 닫는 마지막 조각이고, 평가를 우리가 소유한다는 점에서 값지다.

---

## 2. 멋지지만 지금은 안 되는 것

| 항목 | 무엇이 매력인가 | 막는 것 |
|---|---|---|
| IC-Custom | 사용자가 그린 마스크 위치에 참조 물체 삽입 | flash-attn, Apple 경로 없음. 라이선스도 상업·프로덕션 전면 금지 |
| Wan2.2-Animate-2 14B | 사진 1장에서 캐릭터 애니메이션 | flash-attn + cu126 휠 |
| LAM / LHM++ | 셀카 → 브라우저에서 움직이는 3D 아바타 | 설치 스크립트가 cu118/cu121 전용, xformers |
| HairPort / MagicMakeup | 본인 얼굴 헤어·메이크업 | CUDA 필요, HairPort는 CC BY-NC-ND |
| Waypoint / LongLive / FlashWorld / WorldGen / HY-World 2.0 | 걸어다니는 3D 씬, 실시간 월드 모델 | flex_attention, gsplat nvcc JIT, nunchaku linux 휠 |
| Stand-In | 얼굴 1장으로 정체성 유지 비디오 | flash-attn |
| DealMaTe | 재질 스와치로 마감 교체 | A100 40GB 권장. 다만 아이디어는 Qwen-Edit LoRA로 옮길 수 있다 |
| Qwen-Image-Layered | RGBA 레이어 분해 편집 | 공식 코드가 CUDA 하드코딩. 비공식 MLX 포트는 M3 Max에서 8분, 21GB |
| SANA-WM | 카메라 제어 이미지→비디오 | xformers, flash-linear-attention |

라이선스 때문에 상업 사용이 막히는 것도 따로 적어둔다. Apple SHARP과 LiTo는 연구 전용(apple-amlr), PIXLRelight 가중치는 CC-BY-NC,
ObjectClear는 비상업, FLUX.2 Klein 9B와 그 파생 LoRA는 비상업이다. Klein **4B**는 Apache-2.0이라 안전하다.

---

## 3. 눈에 띄는 발견 세 가지

1. **MoGe-3의 유일한 Mac 경로가 MLX다.** 업스트림 README가 "macOS is not supported"라고 못 박는다. Triton 기반 FlexGEMM 때문이다.
   그런데 MLX 포트가 2026-09-02에 나왔고 mlx-vlm v0.7.0(2026-09-07)에 실렸다. 우리 스택이 아니면 로컬에서 돌릴 방법이 없는 3주 된 모델이다.
2. **브라우저 쪽 자산이 최근 몇 주에 몰려 있다.** BiRefNet dynamic ONNX는 2026-09-01, MoGe-2 WebGPU 가중치는 2026-09-01, metal-gauss 첫 커밋은 2026-08-31이다.
3. **번들 가중치의 라이선스가 실제와 다를 수 있다.** TripoSplat이 비상업 표기의 FLUX.2 VAE를 들고 있지만, 텐서를 비교하면 Apache-2.0 Klein-4B VAE와 같다.
   라이선스는 파일 단위로 확인해야 한다.

---

## 4. 로드맵에 넣을 것

| # | 실험 | 갈래 | 요약 |
|---|---|---|---|
| 12 | depth-parallax-web | C | Depth Anything 3 Small(105MB)로 히어로에 2.5D 패럴랙스. 방문자 사진도 받는다 |
| 13 | moge-relight-web | C | MoGe-2 WebGPU 법선으로 드래그 조명. 깊이 기반 근사와 A/B |
| 14 | page-sees-you | C | FastVLM 또는 LFM2.5-VL로 웹캠 캡션·박스를 생성 필드의 입력으로 |
| 15 | photo-to-splat | C/A | TripoSplat MLX + metal-gauss 학습 + Spark 뷰어. 사진과 실촬영 두 경로 비교 |
| 16 | label-drop-mlx | A | Qwen-Image-Edit-2511 8bit + Lightning 4스텝으로 라벨을 곡면에 감기. 지연·품질 측정 |
| 17 | sam3-to-rfdetr | A/C | SAM 3.1 자동 라벨 → RF-DETR-Seg 33M 학습 → ONNX → 브라우저 실시간 |

---

## 출처

- [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) · [Relight LoRA](https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Relight) · [mlx-gen](https://github.com/lpalbou/mlx-gen) · [mflux](https://github.com/filipstrand/mflux)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) · [ONNX](https://huggingface.co/onnx-community/depth-anything-v3-small)
- [MoGe](https://github.com/microsoft/MoGe) · [moge-webgpu](https://github.com/lyonsno/moge-webgpu)
- [Marigold-IID](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1)
- [FastVLM-0.5B ONNX](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX) · [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) · [Gemma 4 E2B ONNX](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX)
- [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) · [SAM 3](https://github.com/facebookresearch/sam3) · [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) · [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) · [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- [Brush](https://github.com/ArthurBrussee/brush) · [metal-gauss](https://github.com/nandometzger/metal-gauss) · [splat-local](https://github.com/michael-L-i/splat-local) · [Spark](https://github.com/sparkjsdev/spark)
- [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) · [RF-DETR](https://github.com/roboflow/rf-detr) · [DINOv3 ONNX](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX) · [UnifiedReward](https://github.com/CodeGoat24/UnifiedReward)
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) · [Boogu-Image](https://github.com/boogu-project/Boogu-Image) · [Bonsai-Image ternary](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-mlx-2bit)
- [Trazor](https://github.com/PhenX/Trazor) · [OmniSVG](https://github.com/OmniSVG/OmniSVG) · [PosterOmni](https://github.com/MeiGen-AI/PosterOmni)
