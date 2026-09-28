# V3 독립 내부 Test 실행 결과

## 결론

- Test 4,992건을 전부 생성·평가했다.
- 이 Test는 AI Hub 공식 Test가 아니라, 공식 Train에서 기존 학습 265,000건과 Dev 10,000건에 사용하지 않은 잔여 대화다.
- 기존에 고정한 `개발 loss 최선 가중치 / step 20000`과 `greedy` 디코딩을 그대로 사용했다.
- Test 결과를 본 뒤 모델이나 생성 설정을 다시 선택하지 않았다.
- dialogueID와 SHA-256 fingerprint 기준으로 학습·Dev·공식 Validation과 겹치는 항목은 모두 0건이다.

## Test 점수

| 방법 | 어절 R-1 | 어절 R-2 | 어절 R-L | 글자 R-L |
|---|---:|---:|---:|---:|
| Longest utterance | 6.05 | 1.74 | 5.93 | 22.32 |
| Fine-tuned KoBART | 24.53 | 9.96 | 23.42 | 43.16 |

## 공식 Validation과의 차이(Test - Validation)

- 어절 R-1: +0.23
- 어절 R-2: +0.28
- 어절 R-L: +0.23
- 글자 R-L: +0.15

차이가 작으면 기존 공식 Validation 결과가 특정 분할에만 우연히 잘 맞은 것이 아니라는 근거가 된다. 다만 두 집합은 모두 같은 AI Hub 구축 체계에서 왔으므로 외부 서비스 데이터에 대한 일반화까지 증명하지는 않는다.

## 생성 진단

- 입력 잘림 비율: 0.581%
- 생성 길이 제한 도달 비율: 0.000%
- 빈 생성 비율: 0.000%
- 평균 novel word 비율: 75.530%
- 가장 낮은 유형: 토론 대화 / n=32 / 어절 R-L=14.67

## 해석 시 한계

- 공식 Test 파일은 현재 제공 폴더에 없으므로 공식 리더보드 Test 점수는 아니다.
- 내부 Test는 모델이 보지 않은 독립 표본이지만 공식 Train의 잔여분이므로 동일 출처 내부 일반화 성능이다.
- ROUGE는 표현 겹침을 측정한다. 사실성·화자 정확성·부정 반전·자연스러움은 test_human_review.csv로 사람이 함께 확인해야 한다.
- Test 점수에 맞춰 재학습하면 이 Test도 개발 데이터가 된다. 개선 실험은 새 Dev에서 결정하고 Test는 마지막에만 다시 평가해야 한다.

## 생성 파일

- test_manifest.csv: Test 4,992건의 고정 명세
- test_predictions.csv/jsonl: 정답·생성문·건별 점수와 오류 진단
- test_metric_comparison.csv: 기준 모델과 KoBART 평균
- test_metric_ci95.csv: bootstrap 95% 신뢰구간
- test_metrics_by_topic.csv / test_metrics_by_type.csv: 그룹별 결과
- test_human_review.csv: 사람 평가용 표본
- test_evaluation_status.json: 완료 여부·누수 검사·시간·설정