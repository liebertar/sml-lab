# 하드웨어 비교

| | Mac Studio M4 Max 128GB (실험 머신) | MacBook Pro M5 Max 64GB | H100 SXM |
|---|---|---|---|
| FP16 연산 | ~34 TFLOPS (행렬 유닛 없음) | ~70 TFLOPS (Neural Accelerator) | 989 TFLOPS |
| 메모리 대역폭 | 546 GB/s | 614 GB/s | 3.35 TB/s |
| 메모리 | 128 GB | 64 GB | 80 GB |
| H100 대비 연산 | 1/29 | 1/14 | 1× |

## 역할 분담

- Studio: 큰 모델(70B Q4까지), 장시간 학습, 상시 실행. 메모리가 결정 요인.
- 노트북: 빠른 반복 실험. 프리필·파인튜닝·디퓨전은 Studio보다 약 2배 빠름.
- H100 대여: CUDA 전용 실험(Triton, FlashAttention-3, FP8, nunchaku, exllamav3)만.

## 주의

- M4에서 PyTorch MPS의 FP16이 느린 보고가 있다. 가능하면 MLX를 쓴다.
- 브라우저 배포 기준은 다운로드 크기다. 200MB 이하 방문자용, 2GB 이하 클릭 후 로드, 그 이상 데모 전용.
