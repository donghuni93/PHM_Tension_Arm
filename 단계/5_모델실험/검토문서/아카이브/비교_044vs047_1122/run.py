# %% [markdown]
# EXP-044(J) vs EXP-047(S) 비교 (11-22, 사용자 요청): t-SNE로 모델을 고를 수 있나 · 피처를 다듬어서 무엇이 달라졌나
# 1) 같은 출발점 t-SNE: 044 지도를 init으로 047 공간을 다시 펼쳐 겹침(화살표 = 파일 이동). 대조 = 044 공간을 같은 방식으로 다시 펼친 것(바뀐 피처 없이 생기는 흔들림)
# 2) 원래 공간(21열 vs 23열)의 이웃 변화: 10-NN 같은 라벨 비율 044 vs 047, 이웃이 많이 바뀐 파일
# 3) t-SNE seed 흔들림: 공간마다 seed 5개로 2차원 지도의 5-NN 숫자 → 두 공간 차이가 seed 차이보다 큰가
# 4) 바뀐 피처: H10 · H15 vs 쪼갠 4열 (정상 / 고장 분포)
# 5) 모델 판정 겹치기: OOF seed 0 결합 여유(점수 − 문턱) J vs S, 최종 판정이 갈린 파일
# 실행: conda run -n qe-phm python "단계/5_모델실험/검토문서/아카이브/비교_044vs047_1122/run.py"

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from src.exp import *
from src.exp2util import *
from src.models2 import TwoAE
from src.features2 import cached_hi_split, HI_SPLIT
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
out = []; P = lambda s="": (out.append(str(s)), print(s))
ns = NS(); y, names = ns.y, np.array(ns.names); nor = y == 0
DROPC = ["H10_ring_8-12.5k", "H15_share_8-12.5k"]
SP = cached_hi_split("train", ns.X); Hs = pd.concat([ns.T, SP], axis=1)
FX47 = [c for c in ns.FIXED if c not in DROPC] + HI_SPLIT
Z44, keep, c44 = tsne_space(ns)
Z47, _, c47 = tsne_space(ns, Hs, FX47 + list(ns.PEAK) + ["L1_S"])
yk, nk = y[keep], names[keep]; sk = np.array([short(s) for s in nk])
WATCH = ["abn_009", "abn_018", "abn_013", "abn_014", "abn_002", "abn_029", "abn_017", "abn_023", "abn_028", "nor_025", "nor_073", "nor_095"]

# %% 1) 같은 출발점 t-SNE + 겹치기
def tsne_from(Z, init, seed=0):
    I = init - init.mean(0); I = I / I[:, 0].std() * 1e-4                   # sklearn PCA init과 같은 크기로 맞춤
    return TSNE(2, perplexity=30, init=I, random_state=seed, learning_rate="auto").fit_transform(Z)


def align(E, ref):
    """E를 ref에 회전 · 반사 · 크기만 맞춤(모양은 그대로)"""
    a, b = E - E.mean(0), ref - ref.mean(0); a = a / np.linalg.norm(a) * np.linalg.norm(b)
    R, _ = orthogonal_procrustes(a, b); return a @ R + ref.mean(0)


kl44, E0, runs0 = tsne_best(Z44)
E44 = tsne_from(Z44, E0)                                                         # 같은 출발점(E0) · 같은 절차로 두 공간을 펼친다
E47 = align(tsne_from(Z47, E0), E44)
E44b = align(tsne_from(Z44, align(runs0[1] if runs0[1] is not E0 else runs0[2], E0)), E44)   # 대조: 044 공간 그대로, 출발점만 다른 seed 지도
scale = np.linalg.norm(E44 - E44.mean(0), axis=1).mean()
d47, dc = np.linalg.norm(E47 - E44, axis=1) / scale, np.linalg.norm(E44b - E44, axis=1) / scale
E44c = E44b
P(f"1) 지도 이동(평균 반지름 단위): 044 → 047 (같은 출발점) 중앙 {np.median(d47):.3f} · 상위 10 % {np.percentile(d47, 90):.3f} | 대조(044 공간, 출발점만 다름) 중앙 {np.median(dc):.3f} · 상위 10 % {np.percentile(dc, 90):.3f}")
top = np.argsort(d47)[::-1][:12]
P("   많이 움직인 파일: " + " ".join(f"{sk[i]}({d47[i]:.2f})" for i in top))


def scat(ax, E, title, lab=WATCH):
    ax.scatter(*E[yk == 0].T, s=22, color=NOR, alpha=.6, lw=0, label=f"정상 ({(yk == 0).sum()})")
    ax.scatter(*E[yk == 1].T, s=40, color=ABN, lw=0, label=f"고장 ({(yk == 1).sum()})")
    for f_ in lab:
        if f_ in nk: i = list(nk).index(f_); ax.annotate(short(f_), E[i], xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=ABN if f_.startswith("abn") else NOR)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8, loc="upper right")


lim = lambda ax: (ax.set_xlim(min(E44[:, 0].min(), E47[:, 0].min()) - 3, max(E44[:, 0].max(), E47[:, 0].max()) + 3), ax.set_ylim(min(E44[:, 1].min(), E47[:, 1].min()) - 3, max(E44[:, 1].max(), E47[:, 1].max()) + 3))
fig, axs = plt.subplots(2, 2, figsize=(16, 14))
scat(axs[0, 0], E44, f"A. EXP-044 공간 (21열) — 기본 t-SNE")
scat(axs[0, 1], E47, "B. EXP-047 공간 (23열) — A의 배치에서 출발해 다시 펼침")
for ax, (Eb, d, ttl) in zip(axs[1], ((E47, d47, "C. 겹침: 044 위치 → 047 위치 (화살표 = 파일 이동)"), (E44c, dc, "D. 대조: 044 공간 그대로, t-SNE 출발점만 다름 (피처 변화 없는 흔들림)"))):
    ax.scatter(*E44[yk == 0].T, s=14, color=NOR, alpha=.25, lw=0); ax.scatter(*E44[yk == 1].T, s=22, color=ABN, alpha=.25, lw=0)
    for i in range(len(yk)):
        if d[i] > np.percentile(d47, 75): ax.annotate("", Eb[i], E44[i], arrowprops=dict(arrowstyle="->", color=ABN if yk[i] else NOR, lw=1.1, alpha=.85))
    ax.scatter(*Eb[yk == 0].T, s=22, color=NOR, lw=0, label="정상 (새 위치)"); ax.scatter(*Eb[yk == 1].T, s=40, color=ABN, lw=0, label="고장 (새 위치)")
    for i in np.argsort(d)[::-1][:10]: ax.annotate(sk[i], Eb[i], xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=ABN if yk[i] else NOR)
    ax.set_title(ttl + f"\n이동 중앙 {np.median(d):.3f} · 화살표 = 044→047 이동 상위 25 %보다 크게 움직인 파일", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8, loc="upper right")
for ax in axs.ravel(): lim(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "tsne_겹침.png"), dpi=105); plt.close(fig)

# %% 2) 원래 공간의 이웃 변화 (t-SNE 아님)
K = 10
I44 = NearestNeighbors(n_neighbors=K + 1).fit(Z44).kneighbors(Z44)[1][:, 1:]; I47 = NearestNeighbors(n_neighbors=K + 1).fit(Z47).kneighbors(Z47)[1][:, 1:]
same44, same47 = (yk[I44] == yk[:, None]).mean(1), (yk[I47] == yk[:, None]).mean(1)
jac = np.array([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(I44, I47)])
P(f"\n2) 10-NN 같은 라벨 비율 (원래 공간): 정상 {same44[yk == 0].mean():.3f} → {same47[yk == 0].mean():.3f} · 고장 {same44[yk == 1].mean():.3f} → {same47[yk == 1].mean():.3f}")
P(f"   이웃 겹침(Jaccard) 중앙 {np.median(jac):.2f} · 이웃이 절반 넘게 바뀐 파일 {int((jac < .5).sum())}개")
ch = pd.DataFrame(dict(파일=sk, 라벨=np.where(yk, "고장", "정상"), 같은라벨_044=(same44 * K).astype(int), 같은라벨_047=(same47 * K).astype(int), 이웃겹침=jac.round(2)))
ch["변화"] = ch.같은라벨_047 - ch.같은라벨_044
P("   같은 라벨 이웃 수가 바뀐 파일 (10개 중):"); P(md_table(ch[ch.변화 != 0].sort_values(["변화", "라벨"])))
fig, ax = plt.subplots(figsize=(8, 7.5))
jit = lambda n: np.random.default_rng(0).uniform(-.25, .25, n)
for lab_, c, mk_ in ((0, NOR, "s"), (1, ABN, "o")):
    m = yk == lab_; ax.scatter(same44[m] * K + jit(m.sum()), same47[m] * K + jit(m.sum()), color=c, marker=mk_, s=30, alpha=.7, lw=0, label="정상" if lab_ == 0 else "고장")
ax.plot([0, K], [0, K], color="#888", lw=1, ls="--")
for i in np.flatnonzero(np.abs(ch.변화.values) >= 1): ax.annotate(sk[i], (same44[i] * K, same47[i] * K), xytext=(4, 4), textcoords="offset points", fontsize=8, color=ABN if yk[i] else NOR)
ax.set_xlabel("EXP-044 공간: 가까운 10개 중 같은 라벨 수"); ax.set_ylabel("EXP-047 공간: 가까운 10개 중 같은 라벨 수")
ax.set_title("원래 공간(t-SNE 전)의 이웃: 대각선 위 = 047에서 같은 무리 이웃이 늘어남", fontsize=10); ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(os.path.join(FIG, "이웃_변화.png"), dpi=110); plt.close(fig)

# %% 3) t-SNE seed 흔들림 vs 두 공간 차이 (2차원 지도에서 잰 5-NN 숫자)
P("\n3) 2차원 t-SNE 지도의 5-NN (seed 5개): 고장 이웃 중 정상 · 정상 이웃 중 고장 · 폴드 밖 5-NN AUC")
rows = []
for nm, Z in (("044", Z44), ("047", Z47)):
    for s_ in range(5):
        E = TSNE(2, perplexity=30, init="random", random_state=s_, learning_rate="auto").fit_transform(Z); r = knn_report(E, yk, seeds=3)
        rows.append(dict(공간=nm, seed=s_, 고장이웃정상=r[0], 정상이웃고장=r[1], AUC=r[2]))
S3 = pd.DataFrame(rows); g3 = S3.groupby("공간")[["고장이웃정상", "정상이웃고장", "AUC"]].agg(["mean", "std"]).round(3)
P(g3.to_string())
hd = pd.DataFrame([dict(공간=nm, **dict(zip(["고장이웃정상", "정상이웃고장", "AUC", "AUC_sd"], knn_report(Z, yk)))) for nm, Z in (("044", Z44), ("047", Z47))]).round(3)
P("   원래 공간(t-SNE 전) 같은 숫자:"); P(hd.to_string(index=False))
# 3b) 사용자 가설: "경계 고장 2개를 따로 처리하면 047이 더 잘 가른다" → 규칙이 맡는 고장을 빼고 같은 숫자 (원래 공간 + 2차원 지도 seed 5개)
spec_, res_ = rule83_flags(ns.X); R83 = (spec_ | res_)[keep]
for tag, ex_ in (("002 · 029 제외", np.isin(nk, ["abn_002", "abn_029"])), ("8.3k 규칙 고장 전부 + 017(E2) 제외", R83 | (nk == "abn_017"))):
    m = ~ex_; P(f"\n3b) {tag} ({int(m.sum())}개, 빠진 고장 {' '.join(sk[ex_])})")
    for nm, Z in (("044", Z44), ("047", Z47)):
        h = knn_report(Z[m], yk[m]); e2 = [knn_report(TSNE(2, perplexity=30, init="random", random_state=s_, learning_rate="auto").fit_transform(Z[m]), yk[m], seeds=3) for s_ in range(5)]
        e2 = np.array(e2)
        P(f"   {nm}: 원래 공간 고장이웃정상 {h[0]:.3f} · 정상이웃고장 {h[1]:.3f} · AUC {h[2]:.3f} | 2차원 지도(seed 5) 고장이웃정상 {e2[:, 0].mean():.3f} ± {e2[:, 0].std():.3f} · 정상이웃고장 {e2[:, 1].mean():.3f} ± {e2[:, 1].std():.3f} · AUC {e2[:, 2].mean():.3f} ± {e2[:, 2].std():.3f}")

# %% 4) 바뀐 피처
fig, axs = plt.subplots(1, 6, figsize=(18, 5.2), sharey=False)
FE = [("H10_ring_8-12.5k", "H10 울림 8–12.5k (044)"), ("H10a_ring_8-10k", "H10a 울림 8–10k (047)"), ("H10b_ring_10-12.5k", "H10b 울림 10–12.5k (047)"),
      ("H15_share_8-12.5k", "H15 몫 8–12.5k (044)"), ("H15a_share_8-10k", "H15a 몫 8–10k (047)"), ("H15b_share_10-12.5k", "H15b 몫 10–12.5k (047)")]
MARK = ["abn_009", "abn_018", "abn_013", "abn_002", "abn_029", "nor_025"]
rng = np.random.default_rng(0)
for ax, (c, ttl) in zip(axs, FE):
    v = Hs[c].values; au = roc_auc_score(y, v)
    for lab_, xpos, col in ((0, 0, NOR), (1, 1, ABN)):
        m = y == lab_; ax.scatter(xpos + rng.uniform(-.18, .18, m.sum()), v[m], s=14, color=col, alpha=.55, lw=0)
        ax.plot([xpos - .25, xpos + .25], [np.median(v[m])] * 2, color="#222", lw=2)
    for f_ in MARK:
        i = list(names).index(f_); ax.annotate(short(f_), (int(y[i]) + .2, v[i]), fontsize=7, color=ABN if y[i] else NOR)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["정상", "고장"]); ax.set_title(f"{ttl}\nAUC {max(au, 1 - au):.2f}", fontsize=9); ax.grid(axis="y", alpha=.3)
fig.suptitle("바뀐 피처: 044의 8–12.5k 두 열(왼쪽 각 묶음 첫 칸) → 047의 8–10k · 10–12.5k 네 열. 검은 막대 = 중앙값. 울림 비율은 충격 0개면 -1", fontsize=10); fig.tight_layout()
fig.savefig(os.path.join(FIG, "바뀐_피처.png"), dpi=110); plt.close(fig)

# %% 5) 모델 판정 겹치기 (OOF seed 0)
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float); idx = np.arange(ns.n, dtype=float)
M44 = np.column_stack([ns.M, idx, r83])
M47 = np.column_stack([np.hstack([Hs[FX47].values, ns.T[["L1_S"]].values, ns.L]), idx, r83])
mk44 = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
mk47 = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(FX47), nf=len(FX47), df=ns.df, peak=tuple(ns.PEAK)))
O = {}
for nm, M, mk in (("J", M44, mk44), ("S", M47, mk47)):
    mg, fin, a_ = np.empty(ns.n), np.zeros(ns.n, bool), np.empty(ns.n)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(ns.g, y, ns.g):
        m = mk().fit(M[tr], y[tr]); f = m.fires(M[te]); mg[te] = f["score"] - m.m_.tc_; fin[te] = f["main"] | f["E2"] | f["R83"]; a_[te] = f["a"]
    O[nm] = (mg, fin, a_)
rule = r83 > 0
diff = np.flatnonzero(O["J"][1] != O["S"][1])
P(f"\n5) OOF seed 0 최종 판정이 갈린 파일: " + " ".join(f"{short(names[i])}(J {'고장' if O['J'][1][i] else '정상'} → S {'고장' if O['S'][1][i] else '정상'})" for i in diff))
P(f"   J: 놓침 {int((~nor & ~O['J'][1]).sum())} · 오경보 {int((nor & O['J'][1]).sum())} | S: 놓침 {int((~nor & ~O['S'][1]).sum())} · 오경보 {int((nor & O['S'][1]).sum())}")
fig, ax = plt.subplots(figsize=(9, 8.5))
xj, ys = np.clip(O["J"][0], -8, 6), np.clip(O["S"][0], -8, 6)
ax.axhline(0, color="#555", lw=1); ax.axvline(0, color="#555", lw=1); ax.plot([-8, 6], [-8, 6], color="#aaa", lw=1, ls="--")
for lab_, c, mk_ in ((0, NOR, "s"), (1, ABN, "o")):
    m = (y == lab_) & ~rule; ax.scatter(xj[m], ys[m], color=c, marker=mk_, s=34, alpha=.7, lw=0, label=("정상" if lab_ == 0 else "고장") + " (결합으로 판정)")
ax.scatter(xj[rule], ys[rule], facecolor="none", edgecolor=ABN, s=60, lw=1.4, label="고장 · 8.3k 규칙이 켜짐 (결합 여유와 무관하게 고장)")
for i in set(diff) | {list(names).index(f_) for f_ in ("abn_009", "abn_018", "abn_013", "nor_025", "nor_116", "nor_059", "nor_065", "abn_017", "abn_022")}:
    ax.annotate(short(names[i]), (xj[i], ys[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR, weight="bold" if i in diff else "normal")
ax.text(5.8, -7.7, "오른쪽 아래 = J만 고장", ha="right", fontsize=8.5, color="#333"); ax.text(-7.8, 5.6, "왼쪽 위 = S만 고장", fontsize=8.5, color="#333")
ax.set_xlabel("J (EXP-044) 결합 여유 = 점수 - 문턱 (OOF seed 0)"); ax.set_ylabel("S (EXP-047) 결합 여유")
ax.set_title(f"두 모델의 판정 겹치기 — 굵은 이름 = 최종 판정이 갈린 파일 ({len(diff)}개). 점선 = 두 모델 같음", fontsize=10); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(os.path.join(FIG, "판정_겹침.png"), dpi=110); plt.close(fig)
P(f"   결합 여유 상관 {np.corrcoef(O['J'][0], O['S'][0])[0, 1]:.3f}")
open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8").write("\n".join(out))
print("done")
