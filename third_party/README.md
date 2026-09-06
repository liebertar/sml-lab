# third_party

`git subtree --squash`로 가져온 원본 코드. 각 디렉토리의 LICENSE가 그대로 적용된다.

| 디렉토리 | 원본 | 라이선스 |
|---|---|---|
| minRF | https://github.com/cloneofsimo/minRF | Apache-2.0 |

갱신:
```bash
git subtree pull --prefix=third_party/minRF https://github.com/cloneofsimo/minRF main --squash
```
