# 02. 파이프라인 설계 — Part 0 ~ Part 6

각 Part는 **목적 / 왜 필요한가 / 할 일 / 산출물 / 완료 조건 / 함정** 순서로 적는다.
Part는 순서대로 진행하고, 완료 조건을 못 채우면 다음 Part로 넘어가지 않는다.
각 Part를 끝낼 때마다 `docs/04_experiment_log.md`에 1줄 이상 기록한다.

---

## Part 0 — 환경 점검

**목적** 학습을 시작하기 전에 이 컴퓨터에서 무엇이 가능한지 숫자로 확인한다.

**왜 필요한가** 배치 크기, 학습 데이터 규모, 정밀도(bf16/fp32) 선택이 전부 GPU 메모리와
연산 지원 여부에 달려 있다. 이걸 모르고 설정을 잡으면 OOM이나 NaN으로 시간을 버린다.

**할 일**

1. `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
2. GPU 이름·VRAM, bf16 지원 여부(`torch.cuda.is_bf16_supported()`) 확인
3. `transformers`, `sentencepiece`, `rouge-score` 임포트 확인
4. 사전학습 모델 1개를 다운로드해 1 스텝 forward/backward가 도는지 확인 (더미 배치)

**산출물** `artifacts/env_report.json` (torch/cuda/GPU/VRAM/bf16/라이브러리 버전)

**완료 조건** 더미 배치 1스텝이 에러 없이 돌고, VRAM 사용량이 기록되어 있다.

**함정**
- CPU만 있는 경우 Stage A(2천 건)까지만 현실적이다. 그때는 `mt5-small` 또는 `pko-t5-small`로 내린다.
- bf16 미지원 GPU(예: 튜링 이전)에서는 **fp32로 학습**한다. T5 계열은 fp16에서 NaN이 잘 난다.

---

## Part 1 — 데이터 탐색 및 프로파일링

**목적** `docs/01_data_spec.md`의 수치를 이 컴퓨터에서 직접 재현한다.

**왜 필요한가** 전처리·길이 설정·분할 전략이 전부 이 통계에서 나온다.
남이 정리한 표를 믿고 넘어가면 나중에 원인 모를 성능 저하를 만난다.

### valid를 어디까지 읽는가 — 이 Part의 경계 (반드시 지킨다)

valid는 최종 holdout이다. Part 1에서 valid에 대해 읽어도 되는 것은 **딱 두 가지**다.

| valid에서 | 허용 | 이유 |
|---|---|---|
| 건수, `dialogueID` | **O** | train↔valid 중복(누수) 검사에 반드시 필요하다. ID는 정답 정보를 담고 있지 않다 |
| `type`, `topic` 분포 | **O** | 층화가 유지됐는지 확인용. 메타 라벨이고 설정에 영향을 주지 않는다 |
| `summary` 내용·길이 통계 | **X** | 정답 통계를 보면 `max_target_length` 등을 holdout에 맞추게 된다 |
| 대화 길이·태그 빈도 통계 | **X** | train으로 충분하다. 굳이 holdout을 열 이유가 없다 |
| 샘플 원문/요약 열람 | **X** | Part 5 최종 평가 전까지 눈으로도 보지 않는다 |

**길이·태그·요약 관련 통계는 전부 train에서만 낸다.** valid 쪽 산출물은
`artifacts/profile/valid_integrity.json` 하나이고, 그 안에는 건수·중복 검사 결과·유형/주제 분포만 들어간다.
(`valid_stats.json`이라는 이름은 쓰지 않는다. 이름이 곧 유혹이다.)

**할 일**

1. `scripts/profile_data.py` 작성. 두 모드로 나눈다.
   - `--split train`: 건수 / 유형 / 주제 / 발화 수 / 글자 수 / 요약 길이 / 태그 빈도 전부 집계
   - `--split valid --integrity-only`: 건수 / `dialogueID` 집합 / 유형·주제 분포만. **`summary` 필드를 읽지 않는다**
2. `dialogueID` 중복 검사 (train 내부, valid 내부, train↔valid 교차)
3. train 요약문 품질 점검: 빈 문자열, 한 문장 초과(마침표 2개 이상), 태그 포함, 비정상 길이
4. 토크나이저를 붙여 **train의 토큰 길이 분포** 측정 (글자 수가 아니라 토큰 수가 실제 제약이다)
5. train 샘플 20건을 원문-요약 나란히 출력해 눈으로 확인

**산출물**
- `artifacts/profile/train_stats.json` (전체 통계)
- `artifacts/profile/valid_integrity.json` (건수·중복·유형/주제 분포만)
- `artifacts/profile/length_hist.png` (train)
- `docs/01_data_spec.md` 수치 갱신

**완료 조건**
- train↔valid `dialogueID` 교차 중복이 0으로 확인됐다
- train 기준 토큰 길이 p95/p99가 나왔고, 그걸 근거로 `max_source_length` / `max_target_length`를 정했다

**함정**
- `개인및관계.json` 하나가 250MB다. `json.load`로 9개를 동시에 올리면 메모리가 터진다. 파일 단위로 처리하고 즉시 해제한다.
- 인코딩은 UTF-8. Windows에서 `open()` 기본 인코딩(cp949)으로 열면 깨진다. **항상 `encoding="utf-8"` 명시.**
- "한 번만 보면 어때"가 누수의 시작이다. valid를 본 순간은 되돌릴 수 없으므로, 궁금하면 train의 dev를 본다.

---

## Part 2 — 전처리 및 학습 데이터 구축

**목적** 원본 JSON을 모델 입력 형식의 학습셋으로 바꾼다.

**왜 필요한가** 대화를 어떻게 직렬화하느냐에 따라 입력 정보와 길이가 달라질 수 있다. 이 프로젝트의 Stage A에서는 직렬화 간 유의한 ROUGE 차이를 확인하지 못했다.
가이드라인 2.6.1도 "화자와 발화를 이어붙여 인코더에 입력"하라고 명시한다.

**할 일**

1. **입력 직렬화 포맷 설계.** 최소 2가지를 만들어 Stage A에서 비교한다.
   - `plain`: 발화만 개행으로 연결
   - `speaker`: `P01: 발화` 형태로 화자 표시 후 연결 (기본값 후보)
   - (선택) `turn`: 같은 turnID의 연속 발화를 한 줄로 병합 — 길이가 줄고 화자 전환이 또렷해진다
2. **정규화 규칙 확정.** 각 규칙은 켜고 끌 수 있게 만든다.
   - `#@이모티콘#`, `#@시스템#` 제거 여부
   - 반복 자모(`ㅋㅋㅋㅋㅋ` → `ㅋㅋ`), 반복 문장부호 축약
   - 공백 정리, 빈 발화 제거
3. **T5 접두어(prefix) 부착.** T5 계열은 태스크 접두어를 쓴다. 예: `"대화 요약: "`
4. **데이터 분할.** `dialogueID` 해시 기반으로 train/dev 고정 분할, dev는 주제 층화
5. **단계별 subset 생성.** Stage A 2,000 / Stage B 20,000 / Stage C 전체 — **주제 층화 유지**
6. 결과를 `data/processed/*.jsonl` (필드: `id`, `topic`, `type`, `source`, `target`)로 저장

**산출물**
- `src/preprocess.py` (정규화·직렬화·자체 테스트), `src/split.py` (해시 층화 분할), `src/data_loader.py`
- `scripts/build_dataset.py` (실행 진입점)
- `data/processed/{train,dev,dev_debate}.jsonl`, `data/subsets/stage_{a,b}.jsonl`
- `artifacts/preprocess/<ts>_build_dataset/` (config, metrics, samples, log)
- `docs/04_experiment_log.md`에 포맷/정규화 선택과 그 근거

> **구현 결정 (2026-09-19)**: jsonl에는 `source` 문자열을 저장하지 않는다.
> 필드는 `id, topic, type, num_participants, dialogue[{pid, turn, text}], target`이고, `dialogue`는 원문 그대로다.
> `source`는 학습·평가 시 `src.preprocess.build_source(record, cfg["preprocess"])`로 만든다.
> 포맷 3종 × 시스템 태그 2종을 파일로 굳히면 train만 6벌(각 약 290MB)이 되고, 어떤 파일로 학습했는지 헷갈린다.
> 전처리 버전은 run마다 복사하는 `config.yaml`의 `preprocess` 블록이 기록한다.

**완료 조건** 임의 5건을 골라 원본 JSON → 최종 `source`/`target` 변환이 눈으로 맞는다.
토큰 길이 p95가 설정한 `max_source_length` 안에 들어간다.

**함정**
- `build_source`를 바꾸면 이전 run과 같은 config여도 입력이 달라진다. 규칙을 바꿀 때는 새 설정 키를 추가해 이전 동작을 켤 수 있게 남긴다.
- **정답(target)에서 태그를 지우면 안 된다.** 원본 요약문에는 태그가 없다(실측 0건). 손대지 말 것.
- `#@시스템#`은 항상 `#@시스템#사진#`처럼 하위 유형과 함께 나온다(Part 1 실측 100%). `#@이모티콘#`도 9.4%가 그렇다.
  `#@시스템#`만 지우면 `사진#`이 남는다. 제거 정규식은 하위 유형까지 포함하고, 붙어 있는 다음 태그의 `#`을 먹지 않는지 테스트한다.
- 정규화는 **입력에만** 적용하고, 정답 요약문은 원본 그대로 쓴다. 정답을 건드리면 평가가 무의미해진다.
- 전처리 버전을 바꿀 때마다 파일명에 버전을 남긴다. 어떤 데이터로 학습했는지 모르면 실험이 죽는다.

---

## Part 3 — 모델·토크나이저 선정

**목적** seq2seq 사전학습 모델을 고르고, 그 선택의 근거를 남긴다.

**왜 필요한가** 가이드라인은 2020년 기준 dynamic convolution을 제안하지만,
그 핵심 요구는 (a) seq2seq 구조, (b) 구어체에 강한 SentencePiece 계열 토크나이저 두 가지다.
현재는 사전학습 T5 계열이 같은 요구를 훨씬 낮은 비용으로 만족한다.

**후보 (T5 계열로 진행)**

| 모델 | 특징 | 비고 |
|---|---|---|
| `paust/pko-t5-base` | 한국어 전용 T5, BBPE 토크나이저 | **1순위.** 구어·비표준어에서 UNK가 거의 없다 |
| `paust/pko-t5-small` | 위의 소형 | VRAM 부족하거나 Stage A 빠른 반복용 |
| `KETI-AIR/ke-t5-base` | 한/영 T5 | 대안 |
| `google/mt5-base` | 다국어 | 최후 대안. 한국어 전용보다 토큰 효율이 나쁘다 |

**할 일**

1. 1순위 모델 다운로드, 토크나이저로 샘플 100건을 인코딩해 **UNK 비율과 토큰 길이** 측정
2. 한국어 특수 표기(`ㅋㅋ`, `ㅠㅠ`, `#@이름#`)가 어떻게 쪼개지는지 확인
3. 필요하면 `#@...#` 태그를 special token으로 추가 (추가하면 `resize_token_embeddings` 필수)
4. 선택 결과와 측정치를 문서화

**산출물** `artifacts/tokenizer_report.json`, `docs/04_experiment_log.md` 기록

**완료 조건** UNK 비율과 평균 토큰 길이 수치를 근거로 모델 1개를 확정했다.

**함정**
- T5는 **fp16에서 NaN/inf가 잘 난다.** bf16 또는 fp32를 쓴다. `fp16=True`는 기본적으로 쓰지 않는다.
- 이 프로젝트에서는 AdamW 학습률 1e-4를 시작값으로 썼다. Stage B에서 5e-5·1e-4·3e-4를 비교했으며 BART와의 학습률 차이나 3e-5의 실패는 검증하지 않았다.
- special token을 추가했으면 반드시 임베딩을 resize하고, 저장/로드 시 토크나이저를 같이 저장한다.

---

## Part 4 — 학습 (단계적 확대)

**목적** 작게 시작해 매 단계 근거를 확인하고 규모를 올린다.

**왜 필요한가** 큰 데이터로 바로 돌리면 버그와 성능 문제를 구분할 수 없다.
단계마다 "여기까지는 정상"이라는 기준점을 만들어 둬야 원인을 좁힐 수 있다.

### Stage 0 — 과적합 점검 (필수, 5분)

32~64건만 반복 학습해 **학습 데이터를 통째로 외우는지** 본다.
학습 ROUGE-L이 90 이상이면 파이프라인(라벨·마스킹·EOS·디코딩)의 기본 동작 근거가 되지만 모든 오류를 배제하지는 못한다.
이게 안 되면 데이터·손실·학습 예산·디코딩을 점검해야 하므로 **절대 다음 단계로 넘어가지 않는다.**
이 점수는 일반화 성능이 아니다. 보고서에 성능으로 쓰지 않는다.

### Stage A — 2,000건

- 목적: 하이퍼파라미터 탐색, 입력 포맷 비교(`plain` vs `speaker`), 정규화 on/off 비교
- 1 epoch 기준 수 분 이내로 끝나야 한다. 더 걸리면 배치·길이 설정을 줄인다
- 비교 실험은 **한 번에 한 변수만** 바꾼다. 두 개를 같이 바꾸면 해석이 불가능해진다

### Stage B — 20,000건

- 목적: Stage A에서 고른 설정이 규모를 키워도 유지되는지 확인
- 여기서 dev ROUGE가 Stage A 대비 의미 있게 오르지 않으면 **데이터·학습 설정·평가 변동을 함께 점검**한다. 규모를 더 올리지 않는다

### Stage C — 학습 분할 전체 (278,292건)

- 목적: 최종 모델
- Stage B에서 개선이 확인됐을 때만 진행한다
- 체크포인트는 dev 기준 best만 유지하고, 시간 예산과 조기 종료를 반드시 건다

**공통 학습 설정 기본값** (Part 0/1 결과로 조정)

| 항목 | 기본값 | 비고 |
|---|---|---|
| optimizer | AdamW | Adafactor는 메모리 부족 시 |
| learning rate | 1e-4 (탐색 범위 5e-5 ~ 3e-4) | T5는 BART보다 높게 |
| scheduler | linear warmup 5% | |
| precision | bf16 (미지원 시 fp32) | **fp16 금지** |
| max_source_length | 384 (Part 1 측정으로 확정) | |
| max_target_length | 64 | 요약 중앙값 40자 |
| label_smoothing | 0.1 | 반복 생성 완화에 도움 |
| eval | epoch 또는 N step마다 dev ROUGE-L | |
| early stopping | 3회 연속 무개선 | |

**산출물**
- `src/train.py`, `configs/stage_{a,b,c}.yaml`
- `artifacts/runs/<타임스탬프>_<설명>/` 에 config, 로그, 메트릭, 예측 샘플, best 체크포인트
- 매 run마다 `docs/04_experiment_log.md` 1행 추가

**완료 조건** 각 Stage마다 dev ROUGE와 생성 예시 5건이 저장되어 있고,
다음 단계로 갈지 말지의 판단 근거가 로그에 적혀 있다.

**함정**
- 생성 반복(같은 구절 되풀이)은 이 과제에서 흔하다. `no_repeat_ngram_size=3`, `repetition_penalty`로 완화하되,
  **디코딩 설정은 학습된 모델을 바꾸지 않는다.** 디코딩으로 가린 문제를 성능 개선으로 보고하지 않는다.
- 체크포인트 폴더는 금방 수십 GB가 된다. `save_total_limit`을 반드시 건다.
- 학습 시간 측정은 Trainer 시작 시점부터 잰다. 다운로드·전처리 시간을 섞지 않는다.

---

## Part 5 — 평가 및 오류 분석

**목적** 점수 하나가 아니라 "무엇을 왜 틀리는가"를 설명한다.

**왜 필요한가** ROUGE는 어휘 겹침만 본다. 사실 오류(누가 무엇을 했는지 뒤바뀜)를 잡지 못한다.
가이드라인의 요약 규칙 준수 여부도 ROUGE에는 안 나온다.

**할 일** — 자세한 정의는 `docs/03_evaluation.md`

1. dev에서 ROUGE-1/2/L 측정 (공백 기준 + 형태소 기준 둘 다)
2. **가이드라인 준수 자동 점검**: 한 문장인가 / 태그가 없는가 / 길이가 정상인가 / 키워드 2개 이상 겹치는가
3. 주제별·유형별 슬라이스 점수 (특히 토론 대화 188건)
4. 오류 유형 표본 분류 30건 (현재 Claude 단독 판정, 사람 검증 미실시): 사실 오류 / 핵심 누락 / 반복 / 화자 혼동 / 형식 위반
5. 비교 모델 간 차이는 **paired bootstrap 95% 신뢰구간**으로 제시 (점수 차이만 쓰지 않는다)
6. 최종 1회만 test(공식 valid)에서 측정

**산출물**
- `src/evaluate.py`, `src/guideline_check.py`
- `artifacts/eval/<run>/RESULT.md`, `predictions.csv`
- 오류 분류표 30건

**완료 조건** ROUGE + 자동 점검 통과율 + 오류 분류 + 신뢰구간이 한 문서에 정리돼 있다.

**함정**
- 학습 데이터 점수를 성능으로 보고하지 않는다.
- test는 마지막 1회. 여러 번 보면 그 순간부터 dev가 된다.
- 한국어 ROUGE는 토큰화 기준에 따라 점수가 크게 흔들린다. **기준을 명시하지 않은 점수는 비교 불가**다.

---

## Part 6 — 서빙/데모 (선택)

**목적** 가이드라인 2.6.2의 API 규격대로 모델을 호출 가능하게 만든다.

**할 일**

1. `POST /summarize`, 요청 `{"input": ["<s>발화</s><s>발화</s>"]}`, 응답 `{"summary": "..."}`
2. 간단한 웹 UI (대화 붙여넣기 → 요약 출력)
3. 응답 시간 측정

**산출물** `src/serve.py`, 실행 방법을 README에 기록

**완료 조건** 로컬에서 요청 1건이 규격대로 응답한다.
