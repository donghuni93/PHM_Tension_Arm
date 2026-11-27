# %% [markdown]
# V-1 보조 그림: 제출 모델(EXP-062 구성 3)의 ET logit과 경계 구간(-2 ≤ logit < 0)
# - 위: train 150 OOF logit(core.Z_[:, 0]) 벌떼 그림(점 안 겹침). ◆ = 이상치 정상 059 · 065 · 116
# - 아래: val 40 logit, 파일당 한 줄(점 안 겹침). 색 = val 그룹(다른 조 결과로 추론, 진단용)
# - 입력 캐시: 4_outputs/cache/val_eda_tabs.joblib (V-1 run.py가 만든 것)
# - 출력: 자료/figs/ET경계.png
#   conda run -n qe-phm python 5_VAL_단계/V-1_미탐원인진단/자료/plot_경계.py

# %% 설정
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import viz  # noqa: F401  (한글 글꼴 설정)
from src.exp import outlier_idx

SUB = os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062")
m = joblib.load(os.path.join(SUB, "모델", "final_model_062.joblib")); core = m.core_
Ttr, Tva, y, ntr, nva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))
Eva, _ = core._et_input(Tva)
oof = core.Z_[:, 0]
p = np.clip(core.et_.predict_proba(Eva)[:, 1], 1 / 600, 1 - 1 / 600); lv = np.log(p / (1 - p))

GROUPS = {"둘다고장": ["004", "013", "016", "027", "031", "035", "039"], "8조만": ["010", "034", "036"], "우리만": ["017", "024", "025"]}
tag = {f"val_{k}": g for g, ks in GROUPS.items() for k in ks}
grp = np.array([tag.get(n, "둘다정상") for n in nva])
NOR, ABN = "#2a78d6", "#eb6834"
GC = {"둘다고장": "#c0392b", "8조만": "#8e44ad", "우리만": "#f39c12", "둘다정상": "#7f8c8d"}
LO, HI = -2.0, 0.0

def swarm(x, sx=0.12, sy=0.09):
    """벌떼 배치: x는 그대로, 세로 오프셋을 0, ±sy, ±2sy … 순으로 시도해 다른 점과 (sx, sy) 타원 거리 1 이상이 되는 첫 자리."""
    off = np.zeros(len(x)); placed = []
    for i in np.argsort(x):
        near = [j for j in placed if abs(x[i] - x[j]) < sx]
        for k in range(400):
            c = (k + 1) // 2 * (1 if k % 2 else -1) * sy
            if all(((x[i] - x[j]) / sx) ** 2 + ((c - off[j]) / sy) ** 2 >= 1 for j in near):
                off[i] = c; break
        placed.append(i)
    return off

# %% 그림
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 14), sharex=True, gridspec_kw={"height_ratios": [1.3, 4]})
for a in (a1, a2):
    a.axvspan(LO, HI, color="#f7dc6f", alpha=.35, lw=0)
    for v in (LO, HI): a.axvline(v, color="k", lw=.8, ls="--")
    a.grid(axis="x", alpha=.3)

# 위: train OOF
out = np.zeros(len(ntr), bool); out[outlier_idx(ntr)] = True
for row, (lab, msk, col) in enumerate([("train 정상 (120)", y == 0, NOR), ("train 고장 (30)", y == 1, ABN)]):
    xs = oof[msk]; ys = row + swarm(xs); o = out[msk]
    a1.scatter(xs[~o], ys[~o], s=22, c=col, edgecolor="white", lw=.4)
    a1.scatter(xs[o], ys[o], s=50, c=col, marker="D", edgecolor="k", lw=.8)
a1.set_yticks([0, 1]); a1.set_yticklabels(["train 정상 (120)", "train 고장 (30)"]); a1.set_ylim(-.7, 1.7)
k = (oof >= LO) & (oof < HI)
a1.set_title(f"train OOF ET logit · 경계 구간 [-2, 0): 정상 {int((k & (y == 0)).sum())} · 고장 {int((k & (y == 1)).sum())}"
             f"  (◆ = 이상치 정상 059·065·116)", fontsize=11, loc="left")

# 아래: val 파일당 한 줄
o = np.argsort(lv)
for r, i in enumerate(o):
    a2.plot([lv[i]], [r], "o", ms=8, color=GC[grp[i]], mec="white", mew=.5)
    a2.text(lv[i] + .12, r, f"{p[i]:.2f}", va="center", fontsize=7, color="#555")
a2.set_yticks(range(len(o))); a2.set_yticklabels([nva[i] for i in o], fontsize=8); a2.set_ylim(-1, len(o) + 1)
for g, c in GC.items():
    a2.plot([], [], "o", color=c, label=f"{g} ({int((grp == g).sum())})")
a2.legend(loc="lower right", fontsize=9, title="val 그룹 (다른 조 결과로 추론, 진단용)", title_fontsize=8)
nb = int(((lv >= LO) & (lv < HI)).sum()); nf = int((lv >= HI).sum()); nn = int((lv < LO).sum())
for x, s in [((LO - 6.4) / 2 - 1, f"정상 유지 {nn}"), ((LO + HI) / 2, f"경계 {nb}"), ((HI + 6.4) / 2, f"지금 고장 {nf}")]:
    a2.text(x, len(o) + .1, s, ha="center", va="bottom", fontsize=11, weight="bold")
a2.set_title(f"val ET logit (파일당 한 줄, 숫자 = ET 확률) · 경계까지 고장으로 하면 {nf} + {nb} = {nf + nb}개",
             fontsize=11, loc="left", pad=22)
a2.set_xlabel("ET logit = ln(p / (1 - p))   (점선 -2 · 0, 노란 띠 = 경계 구간)")
a2.set_xlim(-6.8, 6.8)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "ET경계.png"), dpi=130)
print("done", nn, nb, nf)

# %% 그림 2 (11-27 추가): 상위 17번째에 선을 그으면 오른쪽은 어떤 파일인가 — 징후(H3 · H8) · 충격 모습
# - 17 = val 실제 고장 수(채점 결과). 선 위치만 정하고, 모델 · 문턱에는 쓰지 않는다(진단용)
# - 징후: train 정상 중앙 · MAD 기준 z, H3_kurt_1-2k > 3 · H8_ring_2-4k > 5 (V-1 F와 같음)
# - 충격: V-2 캐시(val_eda_imps.joblib)의 충격 수 · 최대 배수(포락선/바탕)
import pandas as pd
cols = list(core.cols) + list(core.peak)
Etr, _ = core._et_input(Ttr); nm = y == 0
med = np.median(Etr[nm], 0); mad = 1.4826 * np.median(np.abs(Etr[nm] - med), 0)
mad[mad < 1e-9] = Etr[nm].std(0)[mad < 1e-9] + 1e-9
Zva = (Eva - med) / mad
i3, i8 = cols.index("H3_kurt_1-2k"), cols.index("H8_ring_2-4k")
s3, s8 = Zva[:, i3] > 3, Zva[:, i8] > 5
sign = np.where(s3 & s8, "둘 다", np.where(s3, "H3만", np.where(s8, "H8만", "없음")))
_, Iva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_imps.joblib"))
nimp = np.array([len(im["height_x"]) for im in Iva]); hmax = np.array([max(im["height_x"]) if len(im["height_x"]) else 0 for im in Iva])

K = 17; o = np.argsort(-lv); cut = (lv[o[K - 1]] + lv[o[K]]) / 2
D = pd.DataFrame({"순위": np.arange(1, 41), "file": [nva[i] for i in o], "ET_p": p[o], "logit": lv[o], "징후": sign[o],
                  "H3z": Zva[o, i3], "H8z": Zva[o, i8], "충격수": nimp[o], "최대배수": hmax[o], "그룹": grp[o]})
txt = [f"17번째 선 = logit {cut:+.3f} (16·17·18번째 logit {lv[o[15]]:+.3f} {lv[o[16]]:+.3f} {lv[o[17]]:+.3f})",
       D.head(22).to_string(index=False, float_format="%+.2f"), "",
       "선 오른쪽 17개 징후: " + str(D.head(K).징후.value_counts().to_dict()),
       "그중 새로 들어온 7개(11–17위) 징후: " + str(D.iloc[10:K].징후.value_counts().to_dict()),
       "val 징후 하나만 9개 중 선 오른쪽: " + str([f for f, s in zip(D.head(K).file, D.head(K).징후) if s in ("H3만", "H8만")])]
open(os.path.join(HERE, "17선_표.txt"), "w", encoding="utf-8").write("\n".join(txt))

SM = {"둘 다": ("o", "#c0392b"), "H3만": ("s", "#27ae60"), "H8만": ("^", "#2980b9"), "없음": ("x", "#7f8c8d")}
fig, a = plt.subplots(figsize=(11, 12))
a.axvspan(LO, HI, color="#f7dc6f", alpha=.35, lw=0)
for v in (LO, HI): a.axvline(v, color="k", lw=.8, ls="--")
a.axvline(cut, color="#c0392b", lw=2)
a.text(cut - .08, 1.5, f"17번째 선\n(logit {cut:+.2f})", color="#c0392b", fontsize=10, ha="right", va="center", weight="bold")
oo = o[::-1]
for r, i in enumerate(oo):
    mk, c = SM[sign[i]]
    a.plot([lv[i]], [r], mk, ms=9, color=c, mec=c if mk == "x" else "white", mew=2 if mk == "x" else .5)
    a.text(lv[i] + .15, r, f"{p[i]:.2f} · 충격 {nimp[i]} · 최대 {hmax[i]:.0f}배", va="center", fontsize=7, color="#555")
a.set_yticks(range(len(oo)))
a.set_yticklabels([f"{nva[i]} ({grp[i]})" if grp[i] != "둘다정상" else nva[i] for i in oo], fontsize=8)
for t, i in zip(a.get_yticklabels(), oo): t.set_color(GC[grp[i]] if grp[i] != "둘다정상" else "k")
for s_, (mk, c) in SM.items():
    a.plot([], [], mk, color=c, ms=8, label=f"{s_} ({int((sign == s_).sum())})")
a.legend(loc="lower right", fontsize=9, title="징후: H3 z>3 · H8 z>5", title_fontsize=9)
a.set_ylim(-1, len(oo)); a.set_xlim(-6.8, 6.8); a.grid(axis="x", alpha=.3)
k17 = D.head(K).징후.value_counts()
a.set_title(f"val ET logit 상위 17개에 선을 그으면 · 오른쪽 17개 징후 = " + " · ".join(f"{k} {v}" for k, v in k17.items()),
            fontsize=11, loc="left")
a.set_xlabel("ET logit (점선 -2 · 0, 노란 띠 = 경계 구간, 빨간 선 = 상위 17번째)  · 파일 이름 색 = val 그룹(추론)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "ET_17선_징후.png"), dpi=130)
print("done2")
