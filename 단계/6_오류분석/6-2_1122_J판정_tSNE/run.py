# %% [markdown]
# 6-2 (11-22) 기본 t-SNE(4-10 S3, 입력 21열) 위에 J(EXP-044) 판정을 덧입힌다 — 총정리용
# - 지도 = tsne_default와 같은 공간 · 같은 seed (146개). 빠진 4개(059 · 065 · 116 · 022)는 원래 21열 공간에서 가까운 146개 중 5개의 지도 위치 평균에 ◆로 올린다
# - 판정 = J를 EXP-044와 같은 방식으로 OOF seed 0 (StratifiedGroupKFold 5겹). 켠 부품 = 결합(로지스틱) · E2 · 8.3k. 결합 여유 = 결합 점수 − 그 폴드의 결합 문턱
# - 중첩 5회 횟수 = EXP-044 result.json (다시 계산하지 않음)
# 실행: conda run -n qe-phm python "단계/6_오류분석/6-2_1122_J판정_tSNE/run.py"

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
from src.features2 import PeakRef
from src.models2 import TwoAE
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
EXP044 = os.path.join(_d, "단계", "5_모델비교", "EXP-044_1122_AE추가")
ns = NS(); y, g = ns.y, ns.g; names = np.array(ns.names); nor = y == 0
out = []; P = lambda s="": (out.append(str(s)), print(s))

# %% J OOF seed 0 (EXP-044 run.py의 oof_ae와 같은 계산) + 폴드별 결합 문턱
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float)
M_rule = np.column_stack([ns.M, np.arange(ns.n, dtype=float), r83])
mkJ = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
fo = {k: np.empty(ns.n) for k in ("a", "z", "ae", "margin")}; fo.update({k: np.zeros(ns.n, bool) for k in ("main", "E2", "R83")})
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    m = mkJ().fit(M_rule[tr], y[tr]); f = m.fires(M_rule[te])
    for k in ("a", "z", "ae", "main", "E2", "R83"): fo[k][te] = f[k]
    fo["margin"][te] = f["score"] - m.m_.tc_
caught = fo["main"] | fo["E2"] | fo["R83"]
part = np.where(fo["R83"], "8.3k", np.where(fo["E2"], "E2", np.where(fo["main"], "결합", "—")))
J = json.load(open(os.path.join(EXP044, "result.json"), encoding="utf-8"))["configs"]["중첩: J | W + AE + E2 + 8.3k (최종 후보)"]
n5 = np.array([(5 - J["miss_counts"].get(n_, 0)) if y[i] else J["fp_counts"].get(n_, 0) for i, n_ in enumerate(names)])   # 5회 중 고장 판정 횟수
P(f"OOF seed 0: 고장 {int(caught[~nor].sum())}/30 잡음 (결합 {int(((part == '결합') & ~nor).sum())} · E2 {int(((part == 'E2') & ~nor).sum())} · 8.3k {int(((part == '8.3k') & ~nor).sum())}), 오경보 {int(caught[nor].sum())}/120: {' '.join(names[nor & caught])}")

# %% 기본 t-SNE 공간 (tsne_space와 같은 계산, 빠진 4개 행도 같이 만든다)
keep = ~np.isin(names, list(TSNE_DROP))
Tn = ns.T.copy(); pk = PeakRef(ns.df).fit(ns.L[nor & keep]).transform(ns.L); Tn["H16_peak_h"], Tn["H17_peak_w"] = pk[:, 0], pk[:, 1]
A = np.column_stack([Tn[ns.cols].values, ae_map_cols(ns)]); Z = map_scale(A[nor & keep], A)[0]
assert np.allclose(Z[keep], tsne_space(ns)[0])
kl, Ek, _ = tsne_best(Z[keep])
E = np.empty((ns.n, 2)); E[keep] = Ek
nn = NearestNeighbors(n_neighbors=6).fit(Z[keep]); kidx = np.flatnonzero(keep)
d_, I_ = nn.kneighbors(Z)
nb = np.where(keep[:, None], I_[:, 1:], I_[:, :5])                     # 자기 자신 제외한 가까운 5개 (지도 안 번호)
E[~keep] = Ek[nb[~keep]].mean(1)
nb_abn = (y[kidx][nb] == 1).sum(1)                                     # 원래 21열 공간에서 가까운 5개 중 고장 수
mixed = np.where(y == 1, nb_abn <= 2, nb_abn >= 3)                     # 섞인 점: 고장인데 이웃 과반이 정상 / 정상인데 이웃 과반이 고장
P(f"t-SNE KL {kl:.2f}. 섞인 고장(이웃 5개 중 고장 ≤ 2): {' '.join(short(n_) for n_ in names[mixed & ~nor])}")
P(f"섞인 정상(이웃 5개 중 고장 ≥ 3): {' '.join(short(n_) for n_ in names[mixed & nor]) or '없음'}")

# %% 그림: 왼쪽 = 라벨 + J 판정(켠 부품), 오른쪽 = 결합 여유 색
EDGE = {"결합": "#e8590c", "E2": "#7a3db8", "8.3k": "#000000"}
ann = sorted(set(TSNE_LABELS) | set(TSNE_DROP) | set(names[nor & caught]) | set(names[mixed]))
fig, axs = plt.subplots(1, 2, figsize=(19, 8.6))
ax = axs[0]
for dk, mk_, sz in ((keep, "o", 1.0), (~keep, "D", 1.5)):
    m = dk & nor & ~caught; ax.scatter(*E[m].T, s=26 * sz, marker="s" if mk_ == "o" else "D", color=NOR, alpha=.6, lw=0, label="정상 → 정상" if mk_ == "o" else None)
    m = dk & nor & caught; ax.scatter(*E[m].T, s=90 * sz, marker="s" if mk_ == "o" else "D", color=NOR, edgecolor="#d62728", lw=2, label="정상 → 오경보" if mk_ == "o" else None)
    for k, c in EDGE.items():
        m = dk & ~nor & (part == k)
        ax.scatter(*E[m].T, s=(70 if k == "결합" else 150) * sz, marker=mk_, color=ABN, edgecolor=c, lw=.6 if k == "결합" else 2.4,
                   label=f"고장 → {k}이 잡음 ({int(((part == k) & ~nor).sum())})" if mk_ == "o" else None)
    m = dk & ~nor & ~caught; ax.scatter(*E[m].T, s=140 * sz, marker="X", color=ABN, edgecolor="#d62728", lw=1.5, label=f"고장 → 놓침 ({int((~nor & ~caught).sum())})" if mk_ == "o" else None)
ax.scatter([], [], marker="D", s=60, facecolor="none", edgecolor="#444", label="◆ = 지도에서 뺀 4개 (가까운 5개 위치에 올림)")
ax.set_title("색 = 정답 라벨, 테두리 · 모양 = J 판정과 켠 부품 (OOF seed 0)", fontsize=10)
ax.legend(fontsize=8.3, loc="upper right")
ax = axs[1]
mg = np.clip(fo["margin"], -6, 6); nrm = TwoSlopeNorm(0, -6, 6)
for dk, mn, mf in ((keep, "s", "o"), (~keep, "D", "D")):
    ax.scatter(*E[dk & nor].T, c=mg[dk & nor], cmap="RdBu_r", norm=nrm, s=30 if mn == "s" else 70, marker=mn, lw=0)
    sc = ax.scatter(*E[dk & ~nor].T, c=mg[dk & ~nor], cmap="RdBu_r", norm=nrm, s=80 if mf == "o" else 120, marker=mf, edgecolor="#333", lw=.8)
rule = ~nor & np.isin(part, ["E2", "8.3k"])
ax.scatter(*E[rule].T, s=260, facecolor="none", edgecolor="#000", lw=1.2, ls="--", label=f"규칙(E2 · 8.3k)이 잡은 고장 ({int(rule.sum())})")
ax.scatter([], [], marker="o", s=60, color="#bbb", edgecolor="#333", label="○ 고장"); ax.scatter([], [], marker="s", s=30, color="#bbb", label="□ 정상")
cb = fig.colorbar(sc, ax=ax, shrink=.8); cb.set_label("결합 여유 = 로지스틱 점수 − 문턱 (+ = 고장 판정, ±6에서 자름)")
ax.set_title("색 = J 결합(ET + 박자 + AE z) 여유. 입력 공간의 어디서 결합이 고장 쪽으로 기우는가", fontsize=10)
ax.legend(fontsize=8.3, loc="upper right")
for a_ in axs:
    for f_ in ann:
        i = int(np.flatnonzero(names == f_)[0]); a_.annotate(short(f_), E[i], xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR, weight="bold" if f_ in TSNE_DROP else None)
    a_.set_xticks([]); a_.set_yticks([])
fig.suptitle(f"기본 t-SNE(4-10 S3: H 16열 + AE 5열, 라벨 없이 배치) 위의 J 판정 — KL {kl:.2f} · OOF seed 0: 고장 {int(caught[~nor].sum())}/30 · 오경보 {int(caught[nor].sum())}/120", x=.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "tsne_J판정.png"), dpi=130); plt.close(fig)

# %% 표: 섞인 점 + 빠진 4개 + 오경보
rows = [i for i in range(ns.n) if names[i] in ann]
rows.sort(key=lambda i: (y[i] == 0, -fo["margin"][i]))
T = pd.DataFrame({"파일": [short(names[i]) for i in rows], "정답": ["고장" if y[i] else "정상" for i in rows],
                  "지도": ["◆ 뺌" if not keep[i] else ("섞임" if mixed[i] else "") for i in rows],
                  "이웃 고장(5)": [int(nb_abn[i]) for i in rows], "ET logit": [f"{fo['a'][i]:.2f}" for i in rows],
                  "박자 z": [f"{fo['z'][i]:.1f}" for i in rows], "AE z": [f"{fo['ae'][i]:.2f}" for i in rows],
                  "결합 여유": [f"{fo['margin'][i]:+.2f}" for i in rows], "켠 부품(seed 0)": [part[i] for i in rows],
                  "중첩 5회 고장 판정": [f"{n5[i]}/5" for i in rows]})
with open(os.path.join(HERE, "표_J판정_섞인점.md"), "w", encoding="utf-8") as fh:
    fh.write("# 6-2 섞인 점 · 지도에서 뺀 4개 · 오경보의 J 판정\n\n이웃 고장(5) = t-SNE 전 21열 공간에서 가까운 5개(146개 중) 중 고장 수. 결합 여유 = 로지스틱 점수 − 폴드 문턱(+ = 결합이 고장 판정). "
             "켠 부품 = OOF seed 0에서 고장 판정을 낸 부품(8.3k > E2 > 결합 순으로 표시). 중첩 5회 = EXP-044 result.json\n\n" + md_table(T) + "\n")
P(T.to_string(index=False))
open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8").write("\n".join(out))
print("done")
