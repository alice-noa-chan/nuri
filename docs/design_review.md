# Nuri 구조 공동 검토 기록

세 서브에이전트가 동일 `.dev/idea1.md`~`idea4.md`를 읽고 역할별 제안을 작성한 뒤, 서로 메시지를 주고받아 root가 구조를 통합했다. 실제 학습·성능 검증과 구분되는 설계 리뷰 기록이다.

## 역할과 교차 검토

| 참여자 | 담당 | 다른 참여자와 검토한 내용 |
|---|---|---|
| reasoning_core | latent recurrence, gradients, 위치 표현, reasoning 검증 | generation_design과 X/R 고정·재사용 및 direct-X 경로의 우회 위험; efficiency_review와 공유 K/V·gradient 유지 |
| generation_design | noisy canvas, leakage, 길이, masked decoding | reasoning_core와 canvas가 core 입력에 들어가지 않는 계약; efficiency_review와 S round·vocab projection 비용 |
| efficiency_review | modern encoder, MoE, 파라미터·FLOPs, 하루 예산 | generation_design과 전체 생성 비용; reasoning_core와 gate·routing·source projection 재사용 |
| root | 범용 head 구성, budget 통합, 실험 순서 | 공유 embedding P의 cross-mode 정렬 문제, 명칭의 범위, feature 기본값과 평가 조건 |

## 합의된 결정과 이유

1. **X와 R을 출력 canvas에서 분리한다.** Input encoder와 core는 생성 요청당 한 번 계산한다. 출력 수정 round마다 source를 다시 인코딩하는 비용을 줄인다. Output→R feedback은 별도 실험이다.
2. **Read-only memory는 detach가 아니다.** 고정 X의 K/V를 한 forward 안에서 재사용하고 모든 recurrent step의 gradient를 합친다. Encoder freeze는 별도 전략이다.
3. **Latent RoPE와 step embedding은 기본 OFF다.** 슬롯에는 문장상의 위치가 없고, train 범위 밖 반복이 저절로 안정화된다는 근거도 없다. 입력과 canvas 내부 self-attention에는 각각 token RoPE를 사용한다.
4. **생성 v0는 masked denoising + Mask-Predict 계열 refinement다.** Diffusion process와 reverse transition을 완전히 정의하지 않은 상태에서 diffusion이라고 부르지 않는다.
5. **길이 예측은 별도 head로 시작한다.** Gold-length training이 암묵적 길이 조건을 주는 문제를 명시하고 predicted-length 결과를 기본으로 보고한다. Oracle-length는 진단용이다.
6. **Confidence는 마지막 masked prediction에서 저장한다.** 이미 노출된 token이 자기 embedding을 보고 낸 logit을 confidence로 재사용하면 자기복사 경로가 평가를 오염시킬 수 있다.
7. **Direct-X memory access는 허용한다.** 생성의 copying과 긴 입력 보존에 유리할 수 있다. 대신 X-only/R-only/X+R, R 제거·shuffle로 실제 core 기여를 검사한다.
8. **임베딩 fast/reasoning mode는 별도 측정한다.** 공유 projection이 같은 공간 정렬을 보장하지 않는다. 기본 검색은 query/document에 동일 mode를 사용한다.
9. **Dense 모델로 첫 기준을 잡는다.** MoE는 input FFN 일부부터 켜고 recurrent core MoE는 나중에 본다. Expert 수와 top-k를 늘리는 것의 연산·dispatch 비용을 함께 기록한다.
10. **T와 S는 별개다.** 이 기준에서 T4→T8 증가보다 S4→S8 증가가 훨씬 비싸다. Output vocabulary projection까지 비용 계산에 포함한다.
11. **정답 confidence 증가나 state 변화 자체를 reasoning 증거로 사용하지 않는다.** Final task 성능, harder split, depth curve, 개입 실험을 우선한다. 변화 없는 정답 fixed point는 실패가 아닐 수 있다.
12. **Encoder-family 명칭의 범위를 명시한다.** Autoregressive decoder는 없지만 cross-attention output readout을 가진다. 엄격한 BERT 단일 스택이라고 소개하지 않는다.

## `.dev` 아이디어 반영 위치

| 원본 아이디어 | 채택한 내용 | 독립 실험으로 둔 내용 |
|---|---|---|
| idea1 | read-only X, learned R, shared recurrence, bidirectional R, residual/SwiGLU | adaptive halting, intermediate supervision, random depth |
| idea2 | instruction-conditioned output, masked canvas, noncausal refinement | diffusion objective, 다른 길이/decoding 방식 |
| idea3 | FFN factory와 tokenwise sparse MoE 배치 계약 | input/core MoE 활성화, 전문가 수·폭·top-k |
| idea4 | pre-norm, RMSNorm, SwiGLU, RoPE, bias-free linear | QK norm, local/global, GQA, MLA, backend 최적화 |

추가 실험으로 reasoning-conditioned Matryoshka embedding, grouped hypothesis slots, 학습된 step 선택기를 제안했다. 이 조합의 성능은 아직 확인되지 않았다.

## 근거와 적용 한계

- [ModernBERT](https://arxiv.org/abs/2412.13663): 현대적인 bidirectional encoder 구성의 참고. Nuri의 v0는 모든 입력층을 global로 두며 후보 schedule은 자체 실험이다.
- [Perceiver](https://arxiv.org/abs/2103.03206), [Perceiver IO](https://arxiv.org/abs/2107.14795): 입력→latent, latent 기반 output query의 참고. Input encoder를 포함한 Nuri 전체가 입력 길이에 선형이라는 근거는 아니다.
- [Universal Transformers](https://arxiv.org/abs/1807.03819): depth recurrence와 adaptive computation의 참고.
- [Recurrent-depth 연구](https://arxiv.org/abs/2502.05171): 공유 반복과 depth 확장의 학습·안정화 참고. Decoder 연구 결과를 Nuri의 성능 보장으로 옮기지 않는다.
- [Mask-Predict](https://aclanthology.org/D19-1633/): conditional masked prediction과 confidence 기반 수정의 참고. Nuri의 length/core 조합은 독립 제안이다.
- [LLaDA](https://arxiv.org/abs/2502.09992): masked diffusion variant를 정의할 때의 참고. v0 휴리스틱과 동일한 sampler로 취급하지 않는다.
- [Switch Transformers](https://arxiv.org/abs/2101.03961): sparse expert routing과 안정화 참고. Nuri 후보는 top-2이며 compute-equivalent가 아니다.
- [SimCSE](https://arxiv.org/abs/2104.08821), [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147): embedding loss와 차원별 표현 학습의 참고.

## 미정인 사항

GPU 개수와 정확한 H100 형태, 학습 언어·데이터, tokenizer, 사전학습 checkpoint 재사용 여부, 모델 runtime과 optimizer 설정은 아직 정하지 않았다. 설계 JSON은 구조 기준값을 선언하며 실행 설정이 아니다. 하루 안의 throughput, batch size, 최종 언어 품질은 실측 전 확정하지 않는다.
