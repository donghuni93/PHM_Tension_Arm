# %% 4-12 우리 모델(J-축소) ET 입력 피처 vs 팀원 피처 비교표: 설명 · 이유 · AUC · ET 중요도 (같은 150개 학습 파일, 라벨 없는 정의 그대로)
# 중요도 = ExtraTrees(500그루, balanced) 평균 불순도 감소, seed 5 평균. 통합 = 우리 14 + 팀원 16 = 30열을 한 모델에 넣은 값(서로 비교 가능), 자기 세트 = 각자 열만 넣은 값
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from src.exp2util import NS
ns = NS(); y, names = ns.y, list(ns.names)
ours = ["H1_nimp", "H2_ratio21", "H3_kurt_1-2k", "H5_kurt_4-8k", "H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k", "H12_share_1-2k", "H13_share_2-4k", "H14_share_4-8k", "H15_share_8-12.5k", "H18_hfl", "L1_S"]
A = ns.T[ours].copy(); A.index = names
P = os.path.join(_d, "3_단계", "4_피처", "아카이브", "4-3_1116_팀원피처·중복지도", "figs", "teammate_features.csv")
B = pd.read_csv(P, index_col=0).loc[names]; B.columns = [c for c in B.columns]
nrm = y == 0; B = B.fillna(B[nrm].median())                     # 충격 0개 파일: 정상 중앙값으로 채움 (4-3과 같은 처리)
def imp(X, seeds=range(5)):
    v = np.mean([ExtraTreesClassifier(500, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=s, n_jobs=-1).fit(X.values, y).feature_importances_ for s in seeds], 0)
    return pd.Series(v, X.columns)
U = pd.concat([A.add_prefix("우리:"), B.add_prefix("팀원:")], axis=1)
iu, ia, ib = imp(U), imp(A), imp(B)
auc = lambda s: roc_auc_score(y, s)
rows = []
D = {  # 이름: (출처, 설명, 보려는 것(이유), 현재 모델에서 상태)
 "H1_nimp": ("우리", "log(1 + 충격 수). 충격 = 1 kHz 고역통과 포락선이 100 ms 바탕의 4.5배 넘는 봉우리", "고장은 간헐 충격이 많다 (H1)", "ET 입력"),
 "H2_ratio21": ("우리", "충격 높이 2위 ÷ 1위", "같은 크기 충격이 반복되면 유격 (H1·H4)", "ET 입력"),
 "H3_kurt_1-2k": ("우리", "1–2 kHz 대역 첨도", "간헐 충격의 뾰족함 (H1)", "ET 입력 (H4 2–4k는 상관으로 제외)"),
 "H5_kurt_4-8k": ("우리", "4–8 kHz 대역 첨도", "고주파 충격 뾰족함 (H1)", "ET 입력 (H6 8–12.5k는 제외)"),
 "H7_ring_1-2k": ("우리", "충격 중 울림 주파수가 1–2 kHz인 비율 (충격 수로 축소)", "무엇이 어디를 쳤나 (H2)", "ET 입력, 축소 적용"),
 "H8_ring_2-4k": ("우리", "울림 주파수 2–4 kHz 비율", "연결부 유격 충격의 울림 (H2)", "ET 입력, 축소 적용"),
 "H9_ring_4-8k": ("우리", "울림 주파수 4–8 kHz 비율 (정상은 ≈1)", "노면 충격 = 정상 (H2)", "ET 입력, 축소 적용"),
 "H10_ring_8-12.5k": ("우리", "울림 주파수 8–12.5 kHz 비율", "8 kHz 이상 공진 (H2)", "ET 입력, 축소 적용"),
 "H12_share_1-2k": ("우리", "1 kHz 위 에너지 중 1–2 kHz 몫 (log)", "대역 모양, 게인에 둔감 (H3)", "ET 입력"),
 "H13_share_2-4k": ("우리", "2–4 kHz 몫 (log)", "같음", "ET 입력"),
 "H14_share_4-8k": ("우리", "4–8 kHz 몫 (log)", "같음", "ET 입력"),
 "H15_share_8-12.5k": ("우리", "8–12.5 kHz 몫 (log)", "같음", "ET 입력"),
 "H18_hfl": ("우리", "log10(고주파 포락선 중앙 ÷ 20 Hz 저역 RMS)", "하중 대비 고주파 (H3·H4)", "ET 입력 + E2 규칙 재료"),
 "L1_S": ("우리", "충격 시점이 저주파 주기에 잠긴 정도", "하중 반전 유격 (H4)", "ET 입력 + 박자 z"),
 "impact_n": ("신효섭", "충격 수 (1 kHz 고역통과 포락선 > 중앙값 × 6, 간격 5 ms)", "우리 H1_nimp과 같은 개념", "대응(H1) — 정의만 다름"),
 "ring_2_5k": ("신효섭", "충격 중 2–5 kHz로 울린 비율", "고장 충격 = 낮은 울림", "대응(H8) — 경계 2–5k vs 우리 2–4k"),
 "ring_5_8k": ("신효섭", "5–8 kHz로 울린 비율", "정상 노면 충격", "대응(H9)"),
 "ring_ge8k": ("신효섭", "8 kHz 이상 울린 비율", "8 kHz 이상 공진", "대응(H10)"),
 "ring_f_iqr": ("신효섭", "충격별 울림 주파수의 IQR", "울림이 퍼진 정도", "없음 (참고)"),
 "ring_f_med": ("신효섭", "울림 주파수 중앙값", "대표 울림", "없음 (H7–H10 비율로 대체)"),
 "seg_kurt_mean": ("신효섭", "0.1초 구간별 첨도의 평균", "충격이 여러 구간에 반복되는지", "**없음 (확인 안 함)**"),
 "SK_200_3k": ("신효섭", "200 Hz–3 kHz 스펙트럼 첨도 (주파수별 시간 첨도)", "충격성이 센 대역", "PHO-006(0.2–1k 변형) 기각"),
 "AR_res_ratio": ("신효섭", "AR(40) 백색화 잔차 에너지 비", "예측 불가능한 신호 = 정상 이탈", "EXP-019 스크리닝 기각 (AE와 겹침)"),
 "AR_res_kurt": ("신효섭", "AR 잔차 첨도", "예측 불가 충격의 뾰족함", "같음"),
 "so_ring25_over58": ("권소린", "큰 충격 20개 직후 12 ms의 대역 RMS 2–5k ÷ 5–8k", "대역 비율이라 게인에 둔감", "대응(울림 묶음, H8/H9 비)"),
 "so_ring811_over58": ("권소린", "같은 방식 8–11k ÷ 5–8k", "8 kHz 이상 울림", "**쓰지 않음** (오경보 055·059·116 상승)"),
 "so_ring_dom25": ("권소린", "2–5 kHz가 가장 센 큰 충격의 비율", "낮은 울림 충격 비율", "대응(H8)"),
 "so_bp25_5k_kurt": ("권소린", "2.5–5 kHz 대역통과 첨도", "고장 충격 대역 첨도", "중복 (ext150과 ρ 0.97)"),
 "so_bp25_5k_winkurt_max": ("권소린", "같은 대역 구간 첨도 최대", "가장 뾰족한 구간", "중복 (ρ 0.84)"),
 "so_811_over_58": ("권소린", "전체 신호 8–11k ÷ 5–8k", "고주파 비율", "**쓰지 않음** (8.3k 규칙과 같은 정보, 오경보 상승)"),
}
for c in ours:
    rows.append(("우리", c, D[c][1], D[c][2], D[c][3], auc(A[c]), iu["우리:" + c], ia[c]))
for c in B.columns:
    rows.append((D[c][0], c, D[c][1], D[c][2], D[c][3], auc(B[c]), iu["팀원:" + c], ib[c]))
T = pd.DataFrame(rows, columns=["출처", "피처", "설명", "이유", "현재 모델에서", "AUC", "통합중요도", "자기세트중요도"])
T["고장쪽"] = np.where(T.AUC >= .5, "큼", "작음"); T["분리도"] = (abs(T.AUC - .5) * 2).round(2)
T["통합순위"] = T.통합중요도.rank(ascending=False).astype(int)
T["통합중요도"] = T.통합중요도.round(3); T["자기세트중요도"] = T.자기세트중요도.round(3); T = T.drop(columns="AUC")
T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "표_피처비교.csv"), index=False, encoding="utf-8-sig")
md = lambda df: "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "표_피처비교.md"), "w", encoding="utf-8").write(md(T))
print(f"통합 ET 중요도 합: 우리 {sum(iu[k] for k in iu.index if k.startswith('우리')):.2f} · 팀원 {sum(iu[k] for k in iu.index if k.startswith('팀원')):.2f}")
print(md(T[["출처", "피처", "분리도", "고장쪽", "통합중요도", "통합순위", "자기세트중요도"]].sort_values("통합순위")))
