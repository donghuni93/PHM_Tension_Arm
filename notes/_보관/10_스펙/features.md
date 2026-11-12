---
type: spec
status: 초안
tags: [feature]
publish: none
updated: 2026-11-13
---
# 피처 스펙

- 입력 신호: 평균제거 (MA 미적용, [[decisions]] D3 결정)
- 대역 경계는 [[decisions]] D4, 개수 불일치는 [[decisions]] D5
- 전처리·분할 규칙은 [[pipeline]]

| id | 이름 | 입력 신호 | 정의(식) | 대역(Hz) | 상태 |
|---|---|---|---|---|---|
| T01 | RMS | 평균제거 | $\sqrt{\sum x_i^2/N}$ | — | core |
| T02 | Peak | 평균제거 | $\max\lvert x_i\rvert$ | — | core |
| T03 | Peak-to-Peak | 평균제거 | 최대 진폭 편차 | — | 확장 |
| T04 | MAV (절대평균) | 평균제거 | $\frac{1}{N}\sum\lvert x_i\rvert$ | — | 확장 |
| T05 | Std | 평균제거 | 표준편차 | — | 후보 ([[decisions]] D7) |
| T06 | Mean | 원본 | 평균 | — | 확장 ([[decisions]] D7) |
| T07 | 첨도 (Kurtosis) | 평균제거 | $E[(x-\mu)^4]/\sigma^4$ | — | core |
| T08 | 왜도 (Skewness) | 평균제거 | 3차 모멘트 | — | core |
| T09 | 파고율 (Crest Factor) | 평균제거 | Peak / RMS | — | 확장 |
| T10 | 충격 계수 (Impulse Factor) | 평균제거 | Peak / MAV | — | 확장 |
| T11 | 형상 계수 (Shape Factor) | 평균제거 | RMS / MAV | — | 확장 |
| T12 | 여유 계수 (Clearance Factor) | 평균제거 | $\text{Peak}\big/\left(\frac{1}{N}\sum\sqrt{\lvert x_i\rvert}\right)^2$ | — | 확장 |
| T13 | Hjorth Activity | 평균제거 | $\mathrm{var}(x)$ | — | 확장 |
| T14 | Hjorth Mobility | 평균제거 | $\sqrt{\mathrm{var}(x')/\mathrm{var}(x)}$ | — | 확장 |
| T15 | Hjorth Complexity | 평균제거 | $\mathrm{Mob}(x')/\mathrm{Mob}(x)$ | — | core |
| T16 | TKEO Max | 평균제거 | $\max\Psi[x(n)]$, $\Psi[x(n)]=x(n)^2-x(n-1)x(n+1)$ | — | core |
| T17 | TKEO Mean | 평균제거 | $\Psi$의 평균 | — | 확장 |
| T18 | TKEO Kurtosis | 평균제거 | $\Psi$의 첨도 | — | 확장 |
| T19 | TKEO Energy | 평균제거 | $\Psi$의 Energy | — | 확장 |
| T20 | 영교차율 (ZCR) | 평균제거 | 부호가 바뀌는 비율 | — | 확장 |
| W01 | 윈도우 Max-to-Mean RMS | 평균제거 | 서브윈도우 10개(0.2초)별 RMS의 최댓값 / 평균 | — | core |
| W02 | 윈도우 RMS 표준편차 | 평균제거 | 서브윈도우별 RMS의 표준편차 | — | 확장 |
| W03 | 윈도우 첨도 최댓값 | 평균제거 | 서브윈도우별 첨도의 최댓값 | — | 확장 |
| W04 | 윈도우 ZCR 변동성 | 평균제거 | 서브윈도우별 ZCR의 변동성 | — | 확장 |
| F01 | 스펙트럼 중심 (Centroid) | 평균제거 | $f_c=\sum f_k\lvert X(f_k)\rvert/\sum\lvert X(f_k)\rvert$ | 0–12,500 | core |
| F02 | Spectral Bandwidth | 평균제거 | $\sqrt{\sum(f_k-f_c)^2P_k/\sum P_k}$ | 0–12,500 | 확장 |
| F03 | Spectral Roll-off 85% | 평균제거 | 누적 스펙트럼 에너지 85% 도달 주파수 | 0–12,500 | 확장 |
| F04 | Spectral Roll-off 95% | 평균제거 | 누적 스펙트럼 에너지 95% 도달 주파수 | 0–12,500 | 확장 |
| F05 | 스펙트럼 엔트로피 | 평균제거 | 원문 정의 없음 | 0–12,500 | 확장 |
| F06 | 스펙트럼 첨도 | 평균제거 | 대역별 첨도 (커토그램) | 0–12,500 | 후보 |
| F07 | 1차 피크 주파수 | 평균제거 | 진폭 최대 주파수 | 20–12,500 | core |
| F08 | 1차 피크 진폭 | 평균제거 | F07에서의 진폭 | 20–12,500 | core |
| F09 | 2·3차 피크 주파수 | 평균제거 | 진폭 상위 2·3번째 주파수 | 0–12,500 | 확장 |
| F10 | 2·3차 피크 진폭 | 평균제거 | F09에서의 진폭 | 0–12,500 | 확장 |
| F11 | 공진 이동량 $f_{shift}$ | 평균제거 | $f_{peak,sample}-f_{peak,baseline}$ (기준선: 정상 평균 스펙트럼) | 0–12,500 | 후보 |
| F12 | 고조파 에너지 비율 | 평균제거 | $(E(2f_0)+E(3f_0))/E_{total}$ | $2f_0$, $3f_0$ | 후보 ([[decisions]] D1) |
| B01 | 저주파 대역 에너지 비율 | 평균제거 | $E_{band}/E_{total}$ | 0–2,000 | core |
| B02 | 중주파 대역 에너지 비율 | 평균제거 | $E_{band}/E_{total}$ | 2,000–5,000 | core |
| B03 | 고주파 대역 에너지 비율 | 평균제거 | $E_{band}/E_{total}$ | 5,000–12,500 | core |
| B04 | 균등 대역 에너지 | 평균제거 | 구간별 진폭 제곱합 (10개 또는 20개) | 1,250 또는 625 간격 | 확장 |
| B05 | 물리 대역 절대 에너지 | 평균제거 | 구간별 진폭 제곱합 | B01–B03과 같음 | 확장 ([[decisions]] D6) |
| P01 | 필터 대역 총 에너지 | Bandpass | 필터 통과 파형 진폭 제곱합 | EDA 후 결정 | 후보 |
| P02 | Filtered RMS·Peak·첨도·파고율 | Bandpass | T01·T02·T07·T09와 같은 식 | 1,250 균등 10칸 (첫 칸 20 Hz부터) | 확장 |
| E01 | 포락선 RMS·첨도·주요 피크 | Bandpass | 힐베르트 포락선에서 계산 | 1,250 균등 10칸 (첫 칸 20 Hz부터) | 확장 |
| A01 | ACF 피처 | 평균제거 | 원문 정의 없음 (자기상관 감쇠율 등) | — | 후보 |
| S01 | 스펙트로그램 시간축 요약 | 평균제거 | STFT 대역별 에너지의 최댓값·표준편차·첨도 | 1,250 균등 10칸, STFT 창 1024 | 확장 |
| O01 | WPD 노드 에너지 | 평균제거 | 웨이블릿 패킷 노드 에너지 | 노드별 | 후보 |
| O02 | EMD IMF 피처 | 평균제거 | IMF별 피처 | IMF별 | 후보 |
| O03 | 켑스트럼 피크 | 평균제거 | 스펙트럼의 스펙트럼 피크 | — | 후보 |
| X01 | 축별 에너지 비율 | — | $E_x/E_{total}$ 등 | — | 제외 |
| X02 | 축간 상관계수 | — | $\rho_{xy},\rho_{yz},\rho_{xz}$ | — | 제외 |
| X03 | 좌/우 교차 상관 | — | 두 센서 신호의 위상차·유사도 | — | 제외 |

core: T01 T02 T07 T08 T15 T16 W01 F01 F07 F08 B01 B02 B03 (13개)
확장: core 13 + 위 `확장` 137개 = 150개, `src/features.py`의 `extract_features_ext` ([[30_실험/EXP-002]])
- 생략: T05(평균 제거 후 RMS와 같음, D7), F06(P02 대역별 첨도가 같은 역할), F09·F10은 F07처럼 20 Hz 이상·피크 간격 10 Hz 초과에서 탐색
