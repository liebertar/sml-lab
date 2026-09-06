# 01-minrf-web

소형 rectified flow 모델을 직접 학습해 ONNX로 내보내고, 브라우저(WebGPU)에서 생성까지 돌린다.
학습 → 양자화 → 웹 배포 전 구간을 가장 작은 크기로 한 번 통과하는 것이 목적이다.

## 목표
- MNIST/CIFAR급 RF 모델을 M4 Max에서 학습 (MPS 또는 MLX 포팅)
- ONNX export 후 fp16/int8 크기·품질 비교
- Transformers.js 또는 ONNX Runtime Web으로 페이지에서 샘플 생성

## 참고한 원본
- cloneofsimo/minRF (Apache-2.0) → `third_party/minRF`

## 실행
(작성 예정)

## 결과
(작성 예정)

## 배운 것
(작성 예정)
