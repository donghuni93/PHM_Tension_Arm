# %% [markdown]
# 15-3. D+ 피처(81열)로 같은 그림: 지도 임베딩(ET 근접도 등) t-SNE + 방식별 OOF 이웃 순도 (train 150개만)
# run2.py와 같은 방식 4가지(S0 균등 · S1 AUC 가중 · S2 NCA · S3 ET 근접도)를 C(66) · D(76) · D+(81)에서 비교. D+ = D + 규칙 기반 5열(P83·W83·K1·K2·K3, src/dplus.py)
# 실행: python "휴대폰 분석/15_피처정리_tSNE/run3_Dplus.py"
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "run2.py"), encoding="utf-8").read().split("rows = []\nfor sn")[0]
G = {"__file__": os.path.join(HERE, "run2.py")}; exec(compile(src, "run2", "exec"), G)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from src.dplus import DPlus
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
y, names, mark, SETS, METHODS, oof, tsne_, purity = G["y"], G["names"], G["mark"], G["SETS"], G["METHODS"], G["oof"], G["tsne"], G["purity"]
dp = DPlus().fit(G["X"], G["F"]); Dp = dp.D_; assert Dp.shape == (150, 81)
SETS["D+ D+규칙5 (81)"] = (Dp, dp.names_)
rows = []
for sn, (Zs_, cn) in SETS.items():
    for mn, fn in METHODS.items():
        m_, se = oof(Zs_, fn); rows.append((sn, mn, m_, se)); print(f"{sn:20s} {mn:16s} OOF 순도 {m_:.3f} ± {se:.3f}", flush=True)
R = pd.DataFrame(rows, columns=["피처 집합", "방식", "OOF 이웃 순도", "표준오차"]).sort_values("OOF 이웃 순도", ascending=False)
R.round(3).to_csv(os.path.join(HERE, "supervised_oof_purity_Dplus.csv"), index=False, encoding="utf-8-sig"); print(R.round(3).to_string(index=False))
# 최종 그림: D+ 중 최고 방식
Rd = R[R["피처 집합"].str.startswith("D+")].iloc[0]; Zs_, cn = SETS["D+ D+규칙5 (81)"]; fn = METHODS[Rd["방식"]]
E = fn(Zs_, y, Zs_); nb = NearestNeighbors(n_neighbors=6).fit(E).kneighbors(E[y == 1], return_distance=False)[:, 1:]; pin = y[nb].mean()
fig, ax = plt.subplots(1, 2, figsize=(17, 7.2)); a = ax[0]; NOR, ABN = "#2a78d6", "#eb6834"
a.scatter(*E[y == 0].T, s=26, color=NOR, alpha=.6, lw=0, label="정상"); a.scatter(*E[y == 1].T, s=46, color=ABN, alpha=.9, lw=0, label="고장")
for i in mark: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), E[i], fontsize=8, xytext=(3, 3), textcoords="offset points")
a.set_title(f"D+ 81열 · {Rd['방식']}\n이웃 순도 in-sample {pin:.2f} · 폴드 밖(OOF) {Rd['OOF 이웃 순도']:.2f} (무작위 0.20)", fontsize=11); a.set_xticks([]); a.set_yticks([]); a.legend()
a = ax[1]; Rp = R.copy(); Rp["라벨"] = Rp["피처 집합"].str.split().str[0] + " " + Rp["방식"].str[:2]; Rp = Rp.sort_values("OOF 이웃 순도")
colr = ["#8E24AA" if s.startswith("D+") else "#CE93D8" for s in Rp["피처 집합"]]; a.barh(Rp["라벨"], Rp["OOF 이웃 순도"], xerr=Rp["표준오차"], color=colr); a.axvline(0.2, color="k", ls=":"); a.set_xlabel("OOF 이웃 순도"); a.set_title("방식 비교 (3회 × 5겹). 진한 보라 = D+")
fig.savefig(os.path.join(HERE, "figs", "tsne_supervised_Dplus.png"), dpi=125, bbox_inches="tight"); plt.close(fig)
# 경계 파일 OOF 이웃 고장 비율 (D vs D+, ET 근접도)
def per_file(Zs, fn, seeds=(0, 1, 2)):
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for sd in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(Zs, y):
            Em = fn(Zs[tr], y[tr], Zs); nb_ = NearestNeighbors(n_neighbors=5).fit(Em[tr]).kneighbors(Em[te], return_distance=False); acc[te] += y[tr][nb_].mean(1); cnt[te] += 1
    return acc / cnt
pf = {k: per_file(SETS[k][0], METHODS["S3 ET 근접도"]) for k in ("D C+물리10 (76)", "D+ D+규칙5 (81)")}
for k, p in pf.items(): print(f"{k}: AUC {roc_auc_score(y, p):.3f} | 고장 중 ≥0.5: {(p[y == 1] >= .5).sum()}/30 | 정상 중 ≥0.5: {(p[y == 0] >= .5).sum()}/120")
print("경계 파일 (이웃 5개 중 고장 비율, D → D+):")
for i in mark: print(f"  {names[i]}: {pf['D C+물리10 (76)'][i]:.2f} → {pf['D+ D+규칙5 (81)'][i]:.2f}")
