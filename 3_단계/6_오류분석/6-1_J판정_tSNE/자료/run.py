# %% [markdown]
# 6-2 (11-22) 기본 t-SNE(4-10 S3) 위에 모델 판정을 덧입힌다 — 총정리용. J(EXP-044) · S(EXP-047, 8–12.5k 쪼갬)
# - 지도 = 그 실험의 tsne_default와 같은 공간 · 같은 seed (146개). 빠진 4개(059 · 065 · 116 · 022)는 원래 공간에서 가까운 146개 중 5개의 지도 위치 평균에 ◆로 올린다
# - 판정 = 그 실험 run.py와 같은 방식으로 OOF seed 0 (StratifiedGroupKFold 5겹). 켠 부품 = 결합(로지스틱) · E2 · 8.3k. 결합 여유 = 결합 점수 − 그 폴드의 결합 문턱
# - 중첩 5회 횟수 = 그 실험 result.json (다시 계산하지 않음)
# 실행: conda run -n qe-phm python "3_단계/6_오류분석/6-1_J판정_tSNE/자료/run.py"

# %% 설정
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from src.exp import *
from src.exp2util import *
from src.features2 import PeakRef, cached_hi_split, HI_SPLIT
from src.models2 import TwoAE
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
B5 = os.path.join(_d, "3_단계", "5_모델실험")
ns = NS(); y, g = ns.y, ns.g; names = np.array(ns.names); nor = y == 0
out = []; P = lambda s="": (out.append(str(s)), print(s))
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float); idx = np.arange(ns.n, dtype=float)

# 모델 두 개: (H 표, ET 고정 열, t-SNE 열, result.json 위치 · 키)
SP = cached_hi_split("train", ns.X); Hs = pd.concat([ns.T, SP], axis=1)
FX47 = [c for c in ns.FIXED if c not in ("H10_ring_8-12.5k", "H15_share_8-12.5k")] + HI_SPLIT
M47 = np.hstack([Hs[FX47].values, ns.T[["L1_S"]].values, ns.L])
CFG = {"J": dict(exp="EXP-044", T=ns.T, FX=list(ns.FIXED), M=ns.M, cols=list(ns.cols), json=("EXP-044_1122_AE추가", "중첩: J | W + AE + E2 + 8.3k (최종 후보)"), fig="tsne_J판정"),
       "S": dict(exp="EXP-047", T=Hs, FX=FX47, M=M47, cols=FX47 + list(ns.PEAK) + ["L1_S"], json=("EXP-047_1122_8-12k쪼개기", "중첩: S | J에서 H10 · H15 → 8–10 · 10–12.5k 4열 (이상치 포함 학습)"), fig="tsne_S판정_EXP047")}
EDGE = {"결합": "#e8590c", "E2": "#7a3db8", "8.3k": "#000000"}
keep = ~np.isin(names, list(TSNE_DROP))


def judge(c):
    """OOF seed 0: 판정 부품 · ET · 박자 z · AE z · 결합 여유 + 중첩 5회 고장 판정 횟수"""
    Mr = np.column_stack([c["M"], idx, r83])
    mk = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(c["FX"]), nf=len(c["FX"]), df=ns.df, peak=tuple(ns.PEAK)))
    fo = {k: np.empty(ns.n) for k in ("a", "z", "ae", "margin")}; fo.update({k: np.zeros(ns.n, bool) for k in ("main", "E2", "R83")})
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        m = mk().fit(Mr[tr], y[tr]); f = m.fires(Mr[te])
        for k in ("a", "z", "ae", "main", "E2", "R83"): fo[k][te] = f[k]
        fo["margin"][te] = f["score"] - m.m_.tc_
    fo["caught"] = fo["main"] | fo["E2"] | fo["R83"]
    fo["part"] = np.where(fo["R83"], "8.3k", np.where(fo["E2"], "E2", np.where(fo["main"], "결합", "—")))
    R = json.load(open(os.path.join(find_exp(c["json"][0]), "result.json"), encoding="utf-8"))["configs"][c["json"][1]]
    fo["n5"] = np.array([(5 - R["miss_counts"].get(n_, 0)) if y[i] else R["fp_counts"].get(n_, 0) for i, n_ in enumerate(names)])
    fo["res"] = R
    return fo


def tsne_map(c):
    """tsne_space와 같은 공간 → 지도 좌표 (150, 2; 빠진 4개는 가까운 5개 위치 평균), 이웃 고장 수, KL"""
    Tn = c["T"].copy(); pk = PeakRef(ns.df).fit(ns.L[nor & keep]).transform(ns.L); Tn["H16_peak_h"], Tn["H17_peak_w"] = pk[:, 0], pk[:, 1]
    A = np.column_stack([Tn[c["cols"]].values, ae_map_cols(ns)]); Z = map_scale(A[nor & keep], A)[0]
    assert np.allclose(Z[keep], tsne_space(ns, c["T"], c["cols"])[0])
    kl, Ek, _ = tsne_best(Z[keep])
    E = np.empty((ns.n, 2)); E[keep] = Ek
    _, I_ = NearestNeighbors(n_neighbors=6).fit(Z[keep]).kneighbors(Z)
    nb = np.where(keep[:, None], I_[:, 1:], I_[:, :5])
    E[~keep] = Ek[nb[~keep]].mean(1)
    return E, (y[np.flatnonzero(keep)][nb] == 1).sum(1), kl, Z.shape[1]


def draw(tag, c, fo, E, nb_abn, kl, ncol):
    part, caught = fo["part"], fo["caught"]
    mixed = np.where(y == 1, nb_abn <= 2, nb_abn >= 3)
    ann = sorted(set(TSNE_LABELS) | set(TSNE_DROP) | set(names[nor & caught]) | set(names[mixed]) | set(names[(fo["n5"] < 5) & ~nor]))
    fig, axs = plt.subplots(1, 2, figsize=(19, 8.6))
    ax = axs[0]
    for dk, mk_, sz in ((keep, "o", 1.0), (~keep, "D", 1.5)):
        m = dk & nor & ~caught; ax.scatter(*E[m].T, s=26 * sz, marker="s" if mk_ == "o" else "D", color=NOR, alpha=.6, lw=0, label="정상 → 정상" if mk_ == "o" else None)
        m = dk & nor & caught; ax.scatter(*E[m].T, s=90 * sz, marker="s" if mk_ == "o" else "D", color=NOR, edgecolor="#d62728", lw=2, label="정상 → 오경보" if mk_ == "o" else None)
        for k, ec in EDGE.items():
            m = dk & ~nor & (part == k)
            ax.scatter(*E[m].T, s=(70 if k == "결합" else 150) * sz, marker=mk_, color=ABN, edgecolor=ec, lw=.6 if k == "결합" else 2.4,
                       label=f"고장 → {k}이 잡음 ({int(((part == k) & ~nor).sum())})" if mk_ == "o" else None)
        m = dk & ~nor & ~caught; ax.scatter(*E[m].T, s=140 * sz, marker="X", color=ABN, edgecolor="#d62728", lw=1.5, label=f"고장 → 놓침 ({int((~nor & ~caught).sum())})" if mk_ == "o" else None)
    ax.scatter([], [], marker="D", s=60, facecolor="none", edgecolor="#444", label="◆ = 지도에서 뺀 4개 (가까운 5개 위치에 올림)")
    ax.set_title(f"색 = 정답 라벨, 테두리 · 모양 = {tag} 판정과 켠 부품 (OOF seed 0)", fontsize=10); ax.legend(fontsize=8.3, loc="upper right")
    ax = axs[1]
    mg = np.clip(fo["margin"], -6, 6); nrm = TwoSlopeNorm(0, -6, 6)
    for dk, mn, mf in ((keep, "s", "o"), (~keep, "D", "D")):
        ax.scatter(*E[dk & nor].T, c=mg[dk & nor], cmap="RdBu_r", norm=nrm, s=30 if mn == "s" else 70, marker=mn, lw=0)
        sc = ax.scatter(*E[dk & ~nor].T, c=mg[dk & ~nor], cmap="RdBu_r", norm=nrm, s=80 if mf == "o" else 120, marker=mf, edgecolor="#333", lw=.8)
    rule = ~nor & np.isin(part, ["E2", "8.3k"])
    ax.scatter(*E[rule].T, s=260, facecolor="none", edgecolor="#000", lw=1.2, ls="--", label=f"규칙(E2 · 8.3k)이 잡은 고장 ({int(rule.sum())})")
    ax.scatter([], [], marker="o", s=60, color="#bbb", edgecolor="#333", label="○ 고장"); ax.scatter([], [], marker="s", s=30, color="#bbb", label="□ 정상")
    cb = fig.colorbar(sc, ax=ax, shrink=.8); cb.set_label("결합 여유 = 로지스틱 점수 - 문턱 (+ = 고장 판정, ±6에서 자름)")
    ax.set_title(f"색 = {tag} 결합(ET + 박자 + AE z) 여유. 입력 공간의 어디서 결합이 고장 쪽으로 기우는가", fontsize=10); ax.legend(fontsize=8.3, loc="upper right")
    span = np.ptp(E, 0).max(); placed = []
    for f_ in ann:                                                      # 붙어 있는 점은 이름표를 아래로 내린다
        i = int(np.flatnonzero(names == f_)[0]); dy = 4 - 11 * sum(np.hypot(*(E[i] - E[j])) < .05 * span for j in placed); placed.append(i)
        for a_ in axs: a_.annotate(short(f_), E[i], xytext=(5, dy), textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR, weight="bold" if f_ in TSNE_DROP else None)
    for a_ in axs: a_.set_xticks([]); a_.set_yticks([])
    R = fo["res"]
    fig.suptitle(f"기본 t-SNE({c['exp']} 공간: H {ncol - 5}열 + AE 5열, 라벨 없이 배치) 위의 {tag} 판정 — KL {kl:.2f} · 중첩 F2 {R['f2']:.3f} ± {R['f2_sd']:.3f} · OOF seed 0: 고장 {int(caught[~nor].sum())}/30 · 오경보 {int(caught[nor].sum())}/120",
                 x=.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, c["fig"] + ".png"), dpi=130); plt.close(fig)
    rows = sorted([i for i in range(ns.n) if names[i] in ann], key=lambda i: (y[i] == 0, -fo["margin"][i]))
    return pd.DataFrame({"파일": [short(names[i]) for i in rows], "정답": ["고장" if y[i] else "정상" for i in rows],
                         "지도": ["◆ 뺌" if not keep[i] else ("섞임" if mixed[i] else "") for i in rows],
                         "이웃 고장(5)": [int(nb_abn[i]) for i in rows], "ET logit": [f"{fo['a'][i]:.2f}" for i in rows],
                         "박자 z": [f"{fo['z'][i]:.1f}" for i in rows], "AE z": [f"{fo['ae'][i]:.2f}" for i in rows],
                         "결합 여유": [f"{fo['margin'][i]:+.2f}" for i in rows], "켠 부품(seed 0)": [fo["part"][i] for i in rows],
                         "중첩 5회 고장 판정": [f"{fo['n5'][i]}/5" for i in rows]}), mixed


# %% 실행
md = ["# 6-2 섞인 점 · 지도에서 뺀 4개 · 오경보의 판정 (J = EXP-044, S = EXP-047)\n",
      "이웃 고장(5) = t-SNE 전 그 실험 공간에서 가까운 5개(146개 중) 중 고장 수. 결합 여유 = 로지스틱 점수 − 폴드 문턱(+ = 결합이 고장 판정). "
      "켠 부품 = OOF seed 0에서 고장 판정을 낸 부품(8.3k > E2 > 결합 순으로 표시). 중첩 5회 = 그 실험 result.json\n"]
for tag, c in CFG.items():
    fo = judge(c); E, nb_abn, kl, ncol = tsne_map(c); T, mixed = draw(tag, c, fo, E, nb_abn, kl, ncol)
    P(f"[{tag} {c['exp']}] OOF seed 0: 고장 {int(fo['caught'][~nor].sum())}/30 (결합 {int(((fo['part'] == '결합') & ~nor).sum())} · E2 {int(((fo['part'] == 'E2') & ~nor).sum())} · 8.3k {int(((fo['part'] == '8.3k') & ~nor).sum())}), "
      f"오경보 {int(fo['caught'][nor].sum())}/120: {' '.join(names[nor & fo['caught']])} · KL {kl:.2f}")
    P(f"  섞인 고장: {' '.join(short(n_) for n_ in names[mixed & ~nor])} · 섞인 정상: {' '.join(short(n_) for n_ in names[mixed & nor]) or '없음'}")
    P(T.to_string(index=False)); P()
    md += [f"\n## {tag} ({c['exp']})\n\n![{tag}](figs/{c['fig']}.png)\n\n", md_table(T), "\n"]
open(os.path.join(HERE, "표_J판정_섞인점.md"), "w", encoding="utf-8").write("".join(md))
open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8").write("\n".join(out))
print("done")
