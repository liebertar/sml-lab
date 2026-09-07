# 08-klein-tryon

FLUX.2 Klein 4B의 다중 참조 편집으로 **아바타 1장 + 의류 1장 → 착용 이미지**를 Mac에서 만든다.
전용 VTO 모델 없이 범용 참조 편집 모델이 어디까지 되는지, 그리고 몇 초가 걸리는지 잰다.

## 목표
- 참조 2장(인물, 의류)을 `--image-paths`로 넣고 4스텝 생성
- 의류의 색·패턴이 옮겨지고 인물의 얼굴·체형·배경이 유지되는지 눈으로 채점
- 스텝 4/8, 시드 3개, 양자화 4/8bit에서 시간과 품질 비교

## 이미지 정책
참조 이미지는 **합성 인물(모델이 생성한 사람) 또는 본인, 동의한 사람**만 쓴다. 이 실험은 인물 사진을 T8에서 직접 생성해 쓴다.
타인의 얼굴을 합성하는 것은 이 저장소 범위 밖이다.

## 참고한 원본
- mflux (MIT): `mflux-generate-flux2-edit`, 참조 다중 입력 지원, `--lora-paths`
- black-forest-labs/FLUX.2-klein-4B (Apache-2.0)
- xocialize/tryon-FLUX.2-klein-4B-lora (Apache-2.0): 입력 3장(옷 지운 인물, 상의, 하의), 트리거 `TRYON`
- mattmdjaga/segformer_b2_clothes: `agnostic.py`가 옷 영역을 회색으로 지울 때 사용

## 실행
```bash
# 1) 참조 이미지 생성 (합성 인물 + 의류 상품 사진)
mflux-generate-z-image-turbo --width 768 --height 1024 --steps 8 --seed 1 --output outputs/tryon/avatar.png \
  --prompt "full-body studio photo of one adult standing straight, front-facing, plain light-gray tank top and charcoal shorts, neutral expression, seamless pale gray background, soft even light"
mflux-generate-z-image-turbo --width 1024 --height 1024 --steps 8 --seed 2 --output outputs/tryon/garment.png \
  --prompt "flat-lay ecommerce product photo of a red cable-knit sweater on a white background, no person, no logo"

# 2) 합성
mflux-generate-flux2-edit --model flux2-klein-4b --quantize 4 \
  --image-paths outputs/tryon/avatar.png outputs/tryon/garment.png \
  --prompt "The person from the first image wearing the sweater from the second image. Keep the same face, body proportions, pose and background. Photorealistic." \
  --steps 4 --seed 42 --width 768 --height 1024 --output outputs/tryon/result_s4_seed42.png
```

## 결과

| 설정 | 시간 (s) | 최대 메모리 (GB) | 의류 전이 | 얼굴·체형 유지 | 배경 유지 | 비고 |
|---|---|---|---|---|---|---|
| q4, 4스텝, seed 42 | | | | | | |
| q4, 8스텝, seed 42 | | | | | | |
| q8, 4스텝, seed 42 | | | | | | |
| q4, 4스텝, seed 42, **try-on LoRA** | | | | | | 입력 3장: 옷 지운 인물, 상의, 하의 |

채점은 0/1/2 (실패/부분/성공) 세 단계로만 한다. 정량 지표는 다음 단계에서 붙인다.

## 배운 것
(작성 예정)
