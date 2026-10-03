# Agent 협업 규칙 — Nuri

Nuri는 encoder-only 모델 프로젝트다. 현재 **설계 단계**이며, 여러 에이전트(Claude, Codex 등)가 문서를 통해 아이디어를 주고받으며 구조를 완성한다.

## 파일 역할

| 파일 | 역할 | 수정 규칙 |
|---|---|---|
| `docs/ARCHITECTURE.md` | 설계의 **단일 기준(source of truth)** | 부분 수정만 한다. **전체 재작성 금지.** §14 결정 로그의 사용자 결정(D-001~D-008)은 사용자 승인 없이 바꾸지 않는다 |
| `docs/DISCUSSION.md` | 제안·반론·리뷰 스레드 | **append-only**. 남의 글을 고치지 않는다 |
| `docs/design_review.md` | Codex 서브에이전트 공동 검토 기록 (v0 시점) | 기록용. 갱신은 DISCUSSION에 |
| `docs/archive/` | 이전 판 보관 (Codex v0, Claude v0.2) | 읽기 전용 |
| `configs/nuri_v0.design.json` | 구조 선언 (실행 설정 아님) | ARCHITECTURE.md와 함께 갱신 |
| `.dev/` | 사용자의 원본 아이디어 메모 (git 미포함) | **읽기 전용** |
| `.private/` | 데이터 출처 등 비공개 메모 (git 미포함) | 데이터 관련 메모는 여기에만 |

## 데이터 비공개 규칙 (D-008)

tracked 파일(문서, 설정, 코드, 모델 카드, 커밋 메시지)에 **학습 데이터셋 이름·링크·출처를 쓰지 않는다.** 데이터 관련 내용은 `.private/`에만 기록한다. 코드는 데이터 경로를 환경변수나 `.private/` 설정으로 받는다.

## 작업 흐름

1. 시작 전에 `docs/ARCHITECTURE.md` 전체(특히 §14 결정 로그, §17 열린 질문)와 `docs/DISCUSSION.md`의 최근 항목을 읽는다.
2. 새 아이디어·반론은 먼저 `DISCUSSION.md`에 항목으로 올린다.
3. 사소한 수정(오타, 수식 오류, 명백한 누락)은 `ARCHITECTURE.md`를 바로 고치고 DISCUSSION에 한 줄 남긴다.
4. 구조를 바꾸는 변경은 DISCUSSION에서 합의되거나 사용자가 승인한 뒤 `ARCHITECTURE.md`에 반영하고, 확정되면 §14에 추가한다.
5. 열린 질문(Q-*)에 답할 근거가 있으면 DISCUSSION에 의견을 남긴다. 최종 결정은 사용자가 한다.

## 변경 알림 (`tools/shared_watch.py`)

공유 파일(`AGENTS.md`, `CLAUDE.md`, `docs/**/*.md`(archive 제외), `configs/**/*.json`)의 변경을 에이전트별 snapshot 기준으로 알려준다. Snapshot은 `.watch/<agent>.json`에 있다(git 미포함).

| 모드 | 명령 | 용도 |
|---|---|---|
| watch | `python tools/shared_watch.py watch --agent <이름>` | 변경마다 diff 요약 한 묶음을 출력하는 상주 프로세스. 출력 이벤트를 받을 수 있는 에이전트용 (Claude Code: Monitor) |
| wait | `python tools/shared_watch.py wait --agent <이름>` | 다음 변경이 생길 때까지 대기하다가 diff를 출력하고 종료(기본 timeout 1800초). 명령 실행만 가능한 에이전트용 (Codex) |
| check | `python tools/shared_watch.py check --agent <이름>` | 마지막 확인 이후 변경분을 즉시 출력하고 snapshot 갱신 |
| ack | `python tools/shared_watch.py ack --agent <이름>` | 출력 없이 snapshot만 갱신. 자기 편집 직후 실행하면 자기 변경이 알림으로 돌아오지 않는다 |

규칙:
- **Claude**: 세션 동안 `watch --agent claude`를 Monitor로 상주시킨다.
- **Codex**: 작업 시작 시와 공유 파일 편집 직전에 `check --agent codex`를 실행한다. 할 일을 마쳐 대기할 때는 `wait --agent codex`를 실행해 두고, 변경이 출력되면 원문을 읽고 대응한 뒤 다시 `wait`한다. timeout으로 끝나면 다시 `wait`한다.
- 공유 파일을 편집한 직후에는 `ack --agent <이름>`을 실행한다.
- 알림에 나온 diff는 요약이다. 판단 전에 원문을 읽는다.

## 동시 편집

- 편집 전에 `DISCUSSION.md` 맨 아래에 `🔒 <에이전트> editing <파일> (시작 시각)` 한 줄을 남기고, 끝나면 `🔓 <에이전트> done <파일>`을 남긴다. 다른 에이전트의 잠금이 30분 넘게 풀리지 않았으면 무시해도 된다.
- 파일을 덮어쓰지 말고 부분 편집(Edit)을 쓴다. 쓰기 전에 다시 읽어 그 사이 바뀐 내용이 없는지 확인한다.

## ID 규칙

새 항목은 해당 접두사의 다음 번호를 쓰고, 번호는 재사용하지 않는다.

- `D-` 결정 · `Q-` 열린 질문 (ARCHITECTURE.md §14, §17)

## DISCUSSION 항목 형식

```markdown
### [YYYY-MM-DD] <에이전트 이름> — <제목>
- 대상: §n, D-xxx, Q-xxx
- 유형: 제안 | 반론 | 리뷰 | 질문 | 답변
- 내용: ...
- 근거: (논문, 계산, 실험 결과)
- 상태: open | accepted | rejected | merged
```

## 문체

- 한국어로 쓰고 기술 용어는 영어를 그대로 쓴다.
- 주장은 근거(논문, FLOPs 계산, 실험)와 함께 쓴다. 실측 전 수치는 추정치임을 밝힌다.
- 설계가 확정되기 전에는 학습 코드를 작성하지 않는다 (사용자가 요청한 프로토타입 제외).
