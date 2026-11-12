---
type: spec
status: 초안
tags: [pipeline]
publish: none
updated: 2026-11-13
---
# 파이프라인 규칙

피처 목록은 [[features]], 미결정은 [[decisions]].

## 전처리
- 순서: 평균값 제거 → 이동평균(MA) → 대역 분할 / Bandpass → 특징 추출
- 평균값 제거: $x_1(t)=x(t)-\mu$ (필수)
- MA: $x_2[n]=\mathrm{MA}\{x_1[n]\}$, 적용 여부·윈도우 M은 [[decisions]] D3
- 대역 분할: 균등 분할로 베이스라인, Bandpass는 정상 vs 고장 FFT 비교(EDA) 이후
- 대역·필터 피처는 원본 피처를 대체하지 않고 열로 추가
- 결과 형태: 파일당 1행 → train 150행 × N열
- val·test에 train과 같은 전처리·대역 설정·특징 추출 적용

## 교차검증 분할
- 파일 단위 그룹 분할 (예: `StratifiedGroupKFold`)
- 모든 실험은 같은 그룹 분할 사용

## 폴드 안에서 fit
- 스케일러(RobustScaler), 변환 학습, 특징 선택, 하이퍼파라미터 튜닝(C·gamma 등), 임계값 선택은 모두 학습 fold 안에서 수행
- 상관 계산·PCA·표준화는 train에서 fit, val·test는 transform만
- 특징 선택: 학습 fold에서 Mann–Whitney U 또는 Mutual Information 상위 12–15개
- 상관 정리: |r| ≥ 0.85 쌍은 물리적 의미가 뚜렷한 쪽만 유지, VIF 5–10 이상이면 공선성 의심

## 임계값
- 기본값(확률 0.5, 결정함수 0.0) 사용 금지
- 교차검증 OOF 예측에서 F₂가 최대인 지점으로 결정
- 확률이 아닌 출력(예: Ridge 결정 점수)도 내부 검증에서 임계값 선택

## 증강
- 증강은 train에만, val·test는 원본으로 평가
- 원신호 증강(슬라이싱·노이즈·스케일·시프트)은 특징 추출 전
- SMOTE는 특징 추출·선택 이후
- 세부 조건은 [[decisions]] D10

## 평가·보고
- 보고용 F₂는 임계값 선택에 쓰지 않은 바깥 fold(또는 중첩 교차검증)에서 계산
- F₂와 함께 미탐·오탐 파일 목록 기록
- 교차검증 점수와 val 제출 점수 비교

## 베이스라인 순서
1. 후보 피처 추출 → 150행 × N열, 정상 평균 스펙트럼으로 대역 경계 확인
2. 파일 단위 그룹 폴드 생성
3. RobustScaler + 폴드 내 피처 선택 + RBF-SVM(`class_weight='balanced'`) → OOF에서 F₂ 임계값
4. 같은 피처로 Elastic-Net 로지스틱 회귀 → TabPFN → MiniROCKET ([[decisions]] D12)
