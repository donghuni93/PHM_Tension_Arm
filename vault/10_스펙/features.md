---
type: spec
status: 초안
tags: [feature]
publish: none
updated: 2026-11-13
---
# 피처 스펙

- 입력 신호 `평균제거/MA`: MA 적용 여부는 [[decisions]] D3
- 대역 경계는 [[decisions]] D4, 개수 불일치는 [[decisions]] D5
- 전처리·분할 규칙은 [[pipeline]]

| id | 이름 | 입력 신호 | 정의(식) | 대역(Hz) | 상태 |
|---|---|---|---|---|---|
| T01 | RMS | 평균제거/MA | $\sqrt{\sum x_i^2/N}$ | — | core |
| T02 | Peak | 평균제거/MA | $\max\lvert x_i\rvert$ | — | core |
| T03 | Peak-to-Peak | 평균제거/MA | 최대 진폭 편차 | — | 후보 |
| T04 | MAV (절대평균) | 평균제거/MA | $\frac{1}{N}\sum\lvert x_i\rvert$ | — | 후보 |
| T05 | Std | 평균제거/MA | 표준편차 | — | 후보 ([[decisions]] D7) |
| T06 | Mean | 원본 | 평균 | — | 후보 ([[decisions]] D7) |
| T07 | 첨도 (Kurtosis) | 평균제거/MA | $E[(x-\mu)^4]/\sigma^4$ | — | core |
| T08 | 왜도 (Skewness) | 평균제거/MA | 3차 모멘트 | — | core |
| T09 | 파고율 (Crest Factor) | 평균제거/MA | Peak / RMS | — | 후보 |
| T10 | 충격 계수 (Impulse Factor) | 평균제거/MA | Peak / MAV | — | 후보 |
| T11 | 형상 계수 (Shape Factor) | 평균제거/MA | RMS / MAV | — | 후보 |
| T12 | 여유 계수 (Clearance Factor) | 평균제거/MA | $\text{Peak}\big/\left(\frac{1}{N}\sum\sqrt{\lvert x_i\rvert}\right)^2$ | — | 후보 |
| T13 | Hjorth Activity | 평균제거/MA | $\mathrm{var}(x)$ | — | 후보 |
| T14 | Hjorth Mobility | 평균제거/MA | $\sqrt{\mathrm{var}(x')/\mathrm{var}(x)}$ | — | 후보 |
| T15 | Hjorth Complexity | 평균제거/MA | $\mathrm{Mob}(x')/\mathrm{Mob}(x)$ | — | core |
| T16 | TKEO Max | 평균제거/MA | $\max\Psi[x(n)]$, $\Psi[x(n)]=x(n)^2-x(n-1)x(n+1)$ | — | core |
| T17 | TKEO Mean | 평균제거/MA | $\Psi$의 평균 | — | 후보 |
| T18 | TKEO Kurtosis | 평균제거/MA | $\Psi$의 첨도 | — | 후보 |
| T19 | TKEO Energy | 평균제거/MA | $\Psi$의 Energy | — | 후보 |
| T20 | 영교차율 (ZCR) | 평균제거/MA | 부호가 바뀌는 비율 | — | 후보 |
| W01 | 윈도우 Max-to-Mean RMS | 평균제거/MA | 서브윈도우별 RMS의 최댓값 / 평균 | — | core |
| W02 | 윈도우 RMS 표준편차 | 평균제거/MA | 서브윈도우별 RMS의 표준편차 | — | 후보 |
| W03 | 윈도우 첨도 최댓값 | 평균제거/MA | 서브윈도우별 첨도의 최댓값 | — | 후보 |
| W04 | 윈도우 ZCR 변동성 | 평균제거/MA | 서브윈도우별 ZCR의 변동성 | — | 후보 |
| F01 | 스펙트럼 중심 (Centroid) | 평균제거/MA | $f_c=\sum f_k\lvert X(f_k)\rvert/\sum\lvert X(f_k)\rvert$ | 0–12,500 | core |
| F02 | Spectral Bandwidth | 평균제거/MA | $\sqrt{\sum(f_k-f_c)^2P_k/\sum P_k}$ | 0–12,500 | 후보 |
| F03 | Spectral Roll-off 85% | 평균제거/MA | 누적 스펙트럼 에너지 85% 도달 주파수 | 0–12,500 | 후보 |
| F04 | Spectral Roll-off 95% | 평균제거/MA | 누적 스펙트럼 에너지 95% 도달 주파수 | 0–12,500 | 후보 |
| F05 | 스펙트럼 엔트로피 | 평균제거/MA | 원문 정의 없음 | 0–12,500 | 후보 |
| F06 | 스펙트럼 첨도 | 평균제거/MA | 대역별 첨도 (커토그램) | 0–12,500 | 후보 |
| F07 | 1차 피크 주파수 | 평균제거/MA | 진폭 최대 주파수 | 0–12,500 | core |
| F08 | 1차 피크 진폭 | 평균제거/MA | F07에서의 진폭 | 0–12,500 | core |
| F09 | 2·3차 피크 주파수 | 평균제거/MA | 진폭 상위 2·3번째 주파수 | 0–12,500 | 후보 |
| F10 | 2·3차 피크 진폭 | 평균제거/MA | F09에서의 진폭 | 0–12,500 | 후보 |
| F11 | 공진 이동량 $f_{shift}$ | 평균제거/MA | $f_{peak,sample}-f_{peak,baseline}$ (기준선: 정상 평균 스펙트럼) | 0–12,500 | 후보 |
| F12 | 고조파 에너지 비율 | 평균제거/MA | $(E(2f_0)+E(3f_0))/E_{total}$ | $2f_0$, $3f_0$ | 후보 ([[decisions]] D1) |
| B01 | 저주파 대역 에너지 비율 | 평균제거/MA | $E_{band}/E_{total}$ | 0–2,000 | core |
| B02 | 중주파 대역 에너지 비율 | 평균제거/MA | $E_{band}/E_{total}$ | 2,000–5,000 | core |
| B03 | 고주파 대역 에너지 비율 | 평균제거/MA | $E_{band}/E_{total}$ | 5,000–12,500 | core |
| B04 | 균등 대역 에너지 | 평균제거/MA | 구간별 진폭 제곱합 (10개 또는 20개) | 1,250 또는 625 간격 | 후보 |
| B05 | 물리 대역 절대 에너지 | 평균제거/MA | 구간별 진폭 제곱합 | B01–B03과 같음 | 후보 ([[decisions]] D6) |
| P01 | 필터 대역 총 에너지 | Bandpass | 필터 통과 파형 진폭 제곱합 | EDA 후 결정 | 후보 |
| P02 | Filtered RMS·Peak·첨도·파고율 | Bandpass | T01·T02·T07·T09와 같은 식 | EDA 후 결정 | 후보 |
| E01 | 포락선 RMS·첨도·주요 피크 | Bandpass | 힐베르트 포락선에서 계산 | EDA 후 결정 | 후보 |
| A01 | ACF 피처 | 평균제거/MA | 원문 정의 없음 (자기상관 감쇠율 등) | — | 후보 |
| S01 | 스펙트로그램 시간축 요약 | 평균제거/MA | STFT 대역별 에너지의 최댓값·표준편차·첨도 | 0–12,500 | 후보 |
| O01 | WPD 노드 에너지 | 평균제거/MA | 웨이블릿 패킷 노드 에너지 | 노드별 | 후보 |
| O02 | EMD IMF 피처 | 평균제거/MA | IMF별 피처 | IMF별 | 후보 |
| O03 | 켑스트럼 피크 | 평균제거/MA | 스펙트럼의 스펙트럼 피크 | — | 후보 |
| X01 | 축별 에너지 비율 | — | $E_x/E_{total}$ 등 | — | 제외 |
| X02 | 축간 상관계수 | — | $\rho_{xy},\rho_{yz},\rho_{xz}$ | — | 제외 |
| X03 | 좌/우 교차 상관 | — | 두 센서 신호의 위상차·유사도 | — | 제외 |

core: T01 T02 T07 T08 T15 T16 W01 F01 F07 F08 B01 B02 B03 (13개)
