# sml-lab

작은 모델을 Mac에서 학습·양자화하고 브라우저(WebGPU)까지 배포하는 실험 기록.

- 학습·실험 머신: Mac Studio M4 Max 128GB (MLX / PyTorch MPS)
- 배포 타깃: 브라우저 (Transformers.js, ONNX Runtime Web, WebGPU)
- CUDA 전용 실험은 H100 시간제 대여로만 진행

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

## 실험 목록

| # | 실험 | 상태 | 요약 |
|---|---|---|---|
| 01 | [minrf-web](experiments/01-minrf-web) | 계획 | 소형 rectified flow 학습 → ONNX → 브라우저 생성 |

## 라이선스

이 저장소의 코드는 MIT. `third_party/` 아래 코드는 각 디렉토리의 원본 라이선스를 따른다.
라이선스가 없는 원본은 가져오지 않고 구조만 참고해 다시 작성한다.
