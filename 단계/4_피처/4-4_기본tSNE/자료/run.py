# %% [markdown]
# 4-10 (11-22) t-SNE 공간 다듬기: 눈금 고침 + AE 열 추가 (방법 ①). 라벨은 색칠과 숫자 계산에만
# - S0 예전: 16열, robust_z (EXP-046 그림과 같음, H9 척도가 이상치 포함 여부에 따라 0.016 ↔ 0.6으로 바뀜)
# - S1 16열, map_scale (척도 하한 = 정상 sd × 0.5)          - S2 S1 + AE 평균 1열
# - S3 S1 + AE 5열 (평균 + 옥타브 4칸)                      - S4 S3에서 H10 · H15 → 8–10 · 10–12.5 kHz 4열 (방법 ② 미리보기)
# 지도 = 146개 (정상 이상치 059 · 065 · 116, 고장 022 제외, 사용자 11-22). 숫자는 146개와 150개 둘 다
# 실행: conda run -n qe-phm python "단계/4_피처/4-4_기본tSNE/자료/run.py"

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.exp import *
from src.exp2util import *
from src.features2 import PeakRef, robust_z, cached_hi_split, HI_SPLIT
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
ns = NS(); y, names = ns.y, np.array(ns.names); nor = y == 0
DROP = list(OUTLIER_NORMALS) + ["abn_022"]; keep = ~np.isin(names, DROP)
LAB = ["abn_009", "abn_018", "abn_013", "abn_014", "abn_023", "abn_028", "abn_017", "abn_002", "abn_029", "nor_073"]
out = []; P = lambda s="": (out.append(str(s)), print(s))

Tn = ns.T.copy(); pk = PeakRef(ns.df).fit(ns.L[nor & keep]).transform(ns.L); Tn["H16_peak_h"], Tn["H17_peak_w"] = pk[:, 0], pk[:, 1]
base = Tn[ns.cols].values                                        # 16열
SP = cached_hi_split("train", ns.X)
AE = ae_map_cols(ns)
cols_split = [c for c in ns.cols if c not in ("H10_ring_8-12.5k", "H15_share_8-12.5k")]
A_split = np.column_stack([Tn[cols_split].values, SP[HI_SPLIT].values])
ref = nor & keep
S = {"S0 예전 (16열 · robust_z)": np.column_stack([robust_z(base[ref, j])(base[:, j]) for j in range(base.shape[1])]),
     "S1 16열 · 눈금 고침": map_scale(base[ref], base)[0],
     "S2 S1 + AE 평균": map_scale(np.column_stack([base, AE[:, :1]])[ref], np.column_stack([base, AE[:, :1]]))[0],
     "S3 S1 + AE 5열": map_scale(np.column_stack([base, AE])[ref], np.column_stack([base, AE]))[0],
     "S4 S3 + 8–12.5k 쪼갬": map_scale(np.column_stack([A_split, AE])[ref], np.column_stack([A_split, AE]))[0]}
P("H9 척도: robust_z(이상치 제외 정상) vs map_scale → 최대 |z| " + f"{np.abs(S['S0 예전 (16열 · robust_z)'][:, ns.cols.index('H9_ring_4-8k')]).max():.1f} vs {np.abs(S['S1 16열 · 눈금 고침'][:, ns.cols.index('H9_ring_4-8k')]).max():.1f}")

# %% 분리 숫자 + t-SNE
rows, emb = [], {}
for k, Z in S.items():
    a1 = knn_report(Z[keep], y[keep]); a2 = knn_report(Z, y)
    kl, E, runs = tsne_best(Z[keep]); emb[k] = (kl, E, runs)
    rows.append({"공간": k, "열": Z.shape[1], "고장 이웃 중 정상 (146)": round(a1[0], 3), "정상 이웃 중 고장 (146)": round(a1[1], 3), "폴드 밖 5-NN AUC (146)": f"{a1[2]:.3f} ± {a1[3]:.3f}",
                 "고장 이웃 중 정상 (150)": round(a2[0], 3), "폴드 밖 5-NN AUC (150)": f"{a2[2]:.3f} ± {a2[3]:.3f}", "t-SNE KL": round(kl, 2)})
R = pd.DataFrame(rows); P(md_table(R))
# 섞인 고장 (S1 vs S3): 이웃 5개 중 정상 ≥ 3
from sklearn.neighbors import NearestNeighbors
for k in ("S1 16열 · 눈금 고침", "S3 S1 + AE 5열", "S4 S3 + 8–12.5k 쪼갬"):
    Zk, yk, nk = S[k][keep], y[keep], names[keep]; _, I = NearestNeighbors(n_neighbors=6).fit(Zk).kneighbors(Zk); I = I[:, 1:]
    fo = (yk[I] != yk[:, None]).mean(1)
    P(f"{k} 섞인 고장(정상 이웃 ≥ 3): " + " ".join(f"{nk[i][-3:]}({fo[i] * 5:.0f})" for i in np.flatnonzero((yk == 1) & (fo >= .6))) + " · 섞인 정상(고장 이웃 ≥ 2): " + " ".join(nk[i][-3:] for i in np.flatnonzero((yk == 0) & (fo >= .4))))


def draw(ax, E, yk, nk, title, gray=False):
    if gray:
        ax.scatter(*E.T, s=22, color="#666", alpha=.8, lw=0)
    else:
        ax.scatter(*E[yk == 0].T, s=22, color=NOR, alpha=.65, lw=0, label=f"정상 ({(yk == 0).sum()})")
        ax.scatter(*E[yk == 1].T, s=40, color=ABN, lw=0, label=f"고장 ({(yk == 1).sum()})")
        for f_ in LAB:
            if f_ in nk:
                i = list(nk).index(f_); ax.annotate(short(f_), E[i], xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=ABN if f_.startswith("abn") else NOR)
        ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])


# 그림 1: 다섯 공간 라벨 색 비교 (2 × 3, 마지막 칸 = 숫자 요약)
fig, axs = plt.subplots(2, 3, figsize=(17, 11)); yk, nk = y[keep], names[keep]
for ax, (k, (kl, E, _)) in zip(axs.ravel(), emb.items()):
    r = R.set_index("공간").loc[k]; draw(ax, E, yk, nk, f"{k}\n5-NN AUC {r['폴드 밖 5-NN AUC (146)']} · 고장 이웃 중 정상 {r['고장 이웃 중 정상 (146)']:.2f} · KL {kl:.2f}")
ax = axs.ravel()[-1]; ax.axis("off")
ax.text(0, .95, "읽는 법\n· 146개 (059 · 065 · 116 · 022 제외)\n· 라벨은 색칠만, 배치는 라벨 없이\n· 5-NN AUC = 폴드 밖 kNN 판별력\n· 고장 이웃 중 정상 = 고장의 가까운 5개 중\n  정상 비율 평균(작을수록 고장끼리 뭉침)\n· 이름표 = 4-10 진단의 섞인 고장 · 073", va="top", fontsize=10)
fig.suptitle("t-SNE 공간 비교 (perplexity 30, seed 3개 중 KL 최소)", fontsize=11); fig.tight_layout()
fig.savefig(os.path.join(FIG, "tsne_공간비교.png"), dpi=110); plt.close(fig)

# 그림 2: S3 라벨 없음 | 라벨 색 (사용자 형식)
for k, fn in (("S3 S1 + AE 5열", "tsne_S3_AE5"), ("S4 S3 + 8–12.5k 쪼갬", "tsne_S4_쪼갬")):
    kl, E, runs = emb[k]
    fig, axs = plt.subplots(1, 2, figsize=(17, 8))
    draw(axs[0], E, yk, nk, f"라벨 없음 — 146개 (정상 이상치 059 · 065 · 116, 고장 022 제외), {k} ({S[k].shape[1]}열)", gray=True)
    draw(axs[1], E, yk, nk, "같은 배치, 라벨 색 (비교용)")
    fig.suptitle(f"t-SNE (perplexity 30, KL {kl:.2f})", x=.01, ha="left", fontsize=11); fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{fn}.png"), dpi=110); plt.close(fig)

# 그림 3: S3 seed 안정성 (seed 3개 나란히)
kl, E, runs = emb["S3 S1 + AE 5열"]
fig, axs = plt.subplots(1, 3, figsize=(17, 6))
for s_, (ax, Es) in enumerate(zip(axs, runs)): draw(ax, Es, yk, nk, f"S3 seed {s_}")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "tsne_S3_seed.png"), dpi=100); plt.close(fig)

# AE 열이 섞인 고장을 가르는가 (열별 z, S3 눈금)
Z3 = S["S3 S1 + AE 5열"]; ae_j = list(range(len(ns.cols), len(ns.cols) + 5))
P(""); P("AE 열 z (S3 눈금): 섞인 고장 | 정상 중앙 · 정상 95위 · 고장 중앙")
for j, c in zip(ae_j, AE_MAP_COLS):
    P(f"  {c:10s} " + " ".join(f"{f_[-3:]} {Z3[list(names).index(f_), j]:.1f}" for f_ in LAB[:7]) + f" | {np.median(Z3[nor, j]):.1f} · {np.percentile(Z3[nor, j], 95):.1f} · {np.median(Z3[~nor, j]):.1f}")
open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8").write("\n".join(out))
print("done")
