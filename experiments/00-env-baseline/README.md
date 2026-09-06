# 00-env-baseline

로컬 SLM 서빙의 **기준선**. 아무 최적화도 하지 않은 mlx-lm 기본 설정에서 모델별 숫자를 잰다.
이후 모든 A/B 갈래 실험은 이 표와 비교한다.

## 목표
- 0.6B~9B, 4bit 모델 8종의 적재 시간·TTFT·프리필/디코드 tok/s·최대 메모리
- 같은 프롬프트(약 380 tok), 같은 출력 길이(128 tok), 워밍업 1회 후 3회 중앙값

## 참고한 원본
- mlx-lm `stream_generate`가 돌려주는 prompt_tps / generation_tps / peak_memory를 그대로 사용
- 모델은 전부 `mlx-community` 4bit 변환본 (HF 다운로드 수 기준 상위)

## 실행
```bash
make baseline                                     # 기본 8종, 각 3회
make baseline ARGS="--models mlx-community/Qwen3-0.6B-4bit --runs 2"
```
결과는 `outputs/baseline.json`(git 제외)과 표준출력의 마크다운 표로 나온다. 표를 아래에 붙인다.

## 결과

### Mac Studio M4 Max 128GB (실험 머신)

(측정 대기)

### MacBook Pro M5 Max 64GB (참고, 하네스 검증용 1종)

| 모델 | 디스크 GB | 적재 s | 프롬프트 tok | TTFT s | 프리필 tok/s | 디코드 tok/s | 최대 메모리 GB |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-4bit | 0.33 | 0.55 | 385 | 0.063 | 23140.1 | 592.7 | 0.68 |

## 배운 것
- 0.6B 4bit는 M5 Max에서 디코드 590 tok/s. 대역폭 614 GB/s로 0.33 GB 가중치를 초당 ~1,800회 읽을 수 있으니 이 크기에서는 대역폭이 아니라 커널 오버헤드가 한계다.
- M4 Max는 행렬 유닛이 없어 프리필(23k tok/s)에서 차이가 가장 클 것으로 예상. Studio 측정 후 비율을 기록한다.
