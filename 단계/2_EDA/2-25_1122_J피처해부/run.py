# 2-25 (11-22) J에 쓰인 피처 뜯어보기 (시각화). J = ET(H 15열) + 박자 z(L1_S) + AE z → 로지스틱 OR E2 OR 8.3k
# 그림 1: ET 중요도 (전체 학습 MDI · 5-fold OOF 순열 중요도 = 열을 섞었을 때 ROC-AUC 하락)
# 그림 2: 피처 16개(ET 입력 15 + 박자 S) 분포 — train 정상 · train 고장 · val, 정상 5–95% 띠. 제목 = 단독 AUC · val 띠 밖 비율
# H16 · H17(가장 솟은 좁은 봉우리 높이 · 폭)은 학습 정상 120개 기준(PeakRef)으로 계산 — J 안에서는 fold마다 다시 계산됨
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp import make_et
from src.exp2util import NS
from src.features2 import PeakRef, H_FOLD
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); INK, MUTED, GRID = "#1d2430", "#5b6575", "#c9ced6"; C_N, C_A, C_V = "#2a78d6", "#eb6834", "#8a4fc9"
ns = NS(); y = ns.y; nor = y == 0
pr = PeakRef(ns.df).fit(ns.L[nor]); pk, pkv = pr.transform(ns.L), pr.transform(ns.Lv)
COLS = ns.FIXED + ns.PEAK
H = np.column_stack([ns.T[ns.FIXED].values] + [pk[:, [H_FOLD.index(p) for p in ns.PEAK]]])
Hv = np.column_stack([ns.Tv[ns.FIXED].values] + [pkv[:, [H_FOLD.index(p) for p in ns.PEAK]]])
DESC = {"H1_nimp": "충격 수 (log1p)", "H2_ratio21": "2위/1위 충격 높이", "H3_kurt_1-2k": "첨도 1–2 kHz", "H5_kurt_4-8k": "첨도 4–8 kHz",
        "H7_ring_1-2k": "울림 몫 1–2 kHz", "H8_ring_2-4k": "울림 몫 2–4 kHz", "H9_ring_4-8k": "울림 몫 4–8 kHz", "H10_ring_8-12.5k": "울림 몫 8–12.5 kHz",
        "H12_share_1-2k": "PSD 몫 1–2 kHz", "H13_share_2-4k": "PSD 몫 2–4 kHz", "H14_share_4-8k": "PSD 몫 4–8 kHz", "H15_share_8-12.5k": "PSD 몫 8–12.5 kHz",
        "H16_peak_h": "가장 솟은 좁은 봉우리 높이", "H17_peak_w": "그 봉우리 폭", "H18_hfl": "고주파 포락선 ÷ 하중(<20 Hz) RMS", "L1_S": "박자 (주기 잠금 S)"}
# 그림 1: 중요도
et = make_et(n=300, seed=0)().fit(H, y); mdi = et.feature_importances_
perm = np.zeros(len(COLS)); rng = np.random.default_rng(0)
for s in range(3):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(H, y):
        m = make_et(n=300, seed=0)().fit(H[a], y[a]); base = roc_auc_score(y[b], m.predict_proba(H[b])[:, 1])
        for j in range(len(COLS)):
            Hb = H[b].copy(); Hb[:, j] = rng.permutation(Hb[:, j]); perm[j] += (base - roc_auc_score(y[b], m.predict_proba(Hb)[:, 1])) / 15
o = np.argsort(-perm)
fig, axs = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
for ax, v, t in ((axs[0], perm, "순열 중요도 (OOF ROC-AUC 하락, 5-fold × 3)"), (axs[1], mdi, "MDI (불순도 감소, 전체 학습)")):
    ax.barh(range(len(o))[::-1], v[o], color="#7a3db8", height=.65)
    for k, j in enumerate(o): ax.text(v[j], len(o) - 1 - k, f" {v[j]:.3f}", va="center", fontsize=9, color=INK)
    ax.set_title(t, fontsize=12, color=INK, loc="left"); ax.grid(axis="x", alpha=.3)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0)
axs[0].set_yticks(range(len(o))[::-1]); axs[0].set_yticklabels([f"{COLS[j]}  ({DESC[COLS[j]]})" for j in o], fontsize=9.5, color=INK)
fig.suptitle("ET 입력 15개의 중요도 — 위가 클수록 ET 판단에 많이 쓰임", fontsize=13, color=INK, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(os.path.join(HERE, "그림1_ET중요도.png"), dpi=130); plt.close(fig)
# 그림 2: 분포 (중요도 순 + 박자)
F = [(COLS[j], H[:, j], Hv[:, j]) for j in o] + [("L1_S", ns.T["L1_S"].values, ns.Tv["L1_S"].values)]
fig, axs = plt.subplots(4, 4, figsize=(20, 17)); stats = {}
for ax, (c, v, vv) in zip(axs.ravel(), F):
    q5, q95 = np.percentile(v[nor], [5, 95]); auc = roc_auc_score(y, v); out = float(((vv < q5) | (vv > q95)).mean())
    stats[c] = dict(auc=round(auc, 3), val_out=round(out, 3), val_med_pct=round(float((v[nor] < np.median(vv)).mean()), 2))
    ax.axhspan(q5, q95, color="#e6ecf5", zorder=0)
    for k, (vals, col, mk) in enumerate(((v[nor], C_N, "s"), (v[~nor], C_A, "o"), (vv, C_V, "^"))):
        ax.scatter(k + rng.uniform(-.2, .2, len(vals)), vals, s=14, color=col, marker=mk, alpha=.65, lw=0)
        ax.plot([k - .28, k + .28], [np.median(vals)] * 2, color=INK, lw=2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["정상", "고장", "val"], color=INK)
    ax.set_title(f"{c}\n{DESC[c]} · AUC {auc:.2f} · val 띠 밖 {out:.0%}", fontsize=10.5, color=INK, loc="left")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0); ax.grid(axis="y", alpha=.3, lw=.6)
fig.suptitle("J 피처 16개 분포 (ET 중요도 순, 마지막 = 박자). 회색 띠 = train 정상 5–95%, 검은 선 = 중앙값. AUC > 0.5 = 고장이 큼, < 0.5 = 고장이 작음. val 띠 밖 기대치 ≈ 10% + 고장 몫",
             fontsize=11, color=MUTED, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .965)); fig.savefig(os.path.join(HERE, "그림2_피처분포.png"), dpi=110); plt.close(fig)
json.dump(dict(perm=dict(zip(COLS, np.round(perm, 4).tolist())), mdi=dict(zip(COLS, np.round(mdi, 4).tolist())), stats=stats),
          open(os.path.join(HERE, "수치.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for j in o: print(f"{COLS[j]:18s} 순열 {perm[j]:.3f} · MDI {mdi[j]:.3f} · AUC {stats[COLS[j]]['auc']:.2f} · val 띠 밖 {stats[COLS[j]]['val_out']:.0%} · val 중앙 = 정상 {stats[COLS[j]]['val_med_pct']:.0%}")
print("L1_S", stats["L1_S"])
