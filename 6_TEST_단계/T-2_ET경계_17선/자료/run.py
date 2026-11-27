# %% [markdown]
# T-2 test ET logit 경계 · 상위 17번째 선 (V-1 plot_경계.py를 test에 적용, 11-27)
# - 제출 모델(EXP-062 구성 3)을 그대로 읽는다(학습 없음). test 라벨은 없다
# - 17 = val 실제 고장 비율(17/40)을 test에 그대로 가정한 값. 선 위치만 정하는 진단용이며 모델 · 문턱에 쓰지 않는다
# - 징후: train 정상 중앙 · MAD 기준 z, H3_kurt_1-2k > 3 · H8_ring_2-4k > 5 (V-1과 같음). 충격: src.impacts.all_impacts(fmax=11000) (V-2와 같음)
# - 점 색 = EXP-062 최종 판정, 이름 옆 * = AE 제외(EXP-070)에서 판정이 바뀐 파일
#   conda run -n qe-phm python 6_TEST_단계/T-2_ET경계_17선/자료/run.py
# %% 설정
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import viz  # noqa: F401  (한글 글꼴 설정)
from src.exp import outlier_idx
from src.data import load_folder
from src.impacts import all_impacts

SUB = os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062")
m = joblib.load(os.path.join(SUB, "모델", "final_model_062.joblib")); core = m.core_
Ttr, _, y, ntr, _ = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))
cache = os.path.join(ROOT, "4_outputs", "cache", "test_eda_tabs.joblib")
if os.path.exists(cache):
    Tte, nte, Ite = joblib.load(cache)
else:
    Xte, nte = load_folder(os.path.join(ROOT, "2_data", "TeamProject_test", "test"))
    Tte, Ite = m._tables(Xte), all_impacts(Xte, fmax=11000)
    joblib.dump((Tte, nte, Ite), cache)
nte = [str(n).replace(".csv", "") for n in nte]
Ete, _ = core._et_input(Tte)
oof = core.Z_[:, 0]
p = np.clip(core.et_.predict_proba(Ete)[:, 1], 1 / 600, 1 - 1 / 600); lv = np.log(p / (1 - p))
fin = core.parts(Tte).pred.values.astype(bool)
noae = pd.read_csv(os.path.join(ROOT, "6_TEST_단계", "실험", "EXP-070_AE제외_test", "자료", "test_여유비교.csv"))
chg = set(str(f).replace(".csv", "") for f in noae.file[noae["판정_AE있음"] != noae["판정_AE없음"]])
NOR, ABN = "#2a78d6", "#eb6834"
LO, HI = -2.0, 0.0


def swarm(x, sx=0.12, sy=0.09):
    off = np.zeros(len(x)); placed = []
    for i in np.argsort(x):
        near = [j for j in placed if abs(x[i] - x[j]) < sx]
        for k in range(400):
            c = (k + 1) // 2 * (1 if k % 2 else -1) * sy
            if all(((x[i] - x[j]) / sx) ** 2 + ((c - off[j]) / sy) ** 2 >= 1 for j in near):
                off[i] = c; break
        placed.append(i)
    return off


# %% 그림 1: train OOF · test ET logit 경계
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 14), sharex=True, gridspec_kw={"height_ratios": [1.3, 4]})
for a in (a1, a2):
    a.axvspan(LO, HI, color="#f7dc6f", alpha=.35, lw=0)
    for v in (LO, HI): a.axvline(v, color="k", lw=.8, ls="--")
    a.grid(axis="x", alpha=.3)
out = np.zeros(len(ntr), bool); out[outlier_idx(ntr)] = True
for row, (msk, col) in enumerate([(y == 0, NOR), (y == 1, ABN)]):
    xs = oof[msk]; ys = row + swarm(xs); o = out[msk]
    a1.scatter(xs[~o], ys[~o], s=22, c=col, edgecolor="white", lw=.4)
    a1.scatter(xs[o], ys[o], s=50, c=col, marker="D", edgecolor="k", lw=.8)
a1.set_yticks([0, 1]); a1.set_yticklabels(["train 정상 (120)", "train 고장 (30)"]); a1.set_ylim(-.7, 1.7)
k = (oof >= LO) & (oof < HI)
a1.set_title(f"train OOF ET logit · 경계 구간 [-2, 0): 정상 {int((k & (y == 0)).sum())} · 고장 {int((k & (y == 1)).sum())}  (◆ = 이상치 정상 059·065·116)", fontsize=11, loc="left")
o = np.argsort(lv)
for r, i in enumerate(o):
    a2.plot([lv[i]], [r], "o", ms=8, color=ABN if fin[i] else NOR, mec="white", mew=.5)
    a2.text(lv[i] + .12, r, f"{p[i]:.2f}", va="center", fontsize=7, color="#555")
a2.set_yticks(range(len(o))); a2.set_yticklabels([nte[i] + (" *" if nte[i] in chg else "") for i in o], fontsize=8); a2.set_ylim(-1, len(o) + 1)
a2.plot([], [], "o", color=ABN, label=f"EXP-062 고장 판정 ({int(fin.sum())})"); a2.plot([], [], "o", color=NOR, label=f"EXP-062 정상 판정 ({int((~fin).sum())})")
a2.legend(loc="lower right", fontsize=9, title="* = AE 제외(EXP-070)에서 판정이 바뀐 파일", title_fontsize=8)
nb = int(((lv >= LO) & (lv < HI)).sum()); nf = int((lv >= HI).sum()); nn = int((lv < LO).sum())
for x, s in [((LO - 6.4) / 2 - 1, f"정상 유지 {nn}"), ((LO + HI) / 2, f"경계 {nb}"), ((HI + 6.4) / 2, f"ET 고장 쪽 {nf}")]:
    a2.text(x, len(o) + .1, s, ha="center", va="bottom", fontsize=11, weight="bold")
a2.set_title(f"test ET logit (파일당 한 줄, 숫자 = ET 확률) · 경계까지 고장으로 하면 {nf} + {nb} = {nf + nb}개", fontsize=11, loc="left", pad=22)
a2.set_xlabel("ET logit = ln(p / (1 - p))   (점선 -2 · 0, 노란 띠 = 경계 구간)"); a2.set_xlim(-6.8, 6.8)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "01_ET경계.png"), dpi=130)

# %% 그림 2: 상위 17번째 선 · 징후 · 충격
cols = list(core.cols) + list(core.peak)
Etr, _ = core._et_input(Ttr); nm = y == 0
med = np.median(Etr[nm], 0); mad = 1.4826 * np.median(np.abs(Etr[nm] - med), 0)
mad[mad < 1e-9] = Etr[nm].std(0)[mad < 1e-9] + 1e-9
Z = (Ete - med) / mad
i3, i8 = cols.index("H3_kurt_1-2k"), cols.index("H8_ring_2-4k")
s3, s8 = Z[:, i3] > 3, Z[:, i8] > 5
sign = np.where(s3 & s8, "둘 다", np.where(s3, "H3만", np.where(s8, "H8만", "없음")))
nimp = np.array([len(im["height_x"]) for im in Ite]); hmax = np.array([max(im["height_x"]) if len(im["height_x"]) else 0 for im in Ite])
K = 17; o = np.argsort(-lv); cut = (lv[o[K - 1]] + lv[o[K]]) / 2
D = pd.DataFrame({"순위": np.arange(1, len(o) + 1), "file": [nte[i] for i in o], "ET_p": p[o], "logit": lv[o], "징후": sign[o],
                  "H3z": Z[o, i3], "H8z": Z[o, i8], "충격수": nimp[o], "최대배수": hmax[o],
                  "EXP062": np.where(fin[o], "고장", "정상"), "AE제외변화": ["*" if nte[i] in chg else "" for i in o]})
D.to_csv(os.path.join(HERE, "17선_표.csv"), index=False, encoding="utf-8-sig")
one = [f for f, s in zip(nte, sign) if s in ("H3만", "H8만")]
txt = [f"17번째 선 = logit {cut:+.3f} (16·17·18번째 logit {lv[o[15]]:+.3f} {lv[o[16]]:+.3f} {lv[o[17]]:+.3f}) · 17 = val 고장 비율 가정",
       D.head(25).to_string(index=False, float_format="%+.2f"), "",
       "선 오른쪽 17개 징후: " + str(D.head(K)["징후"].value_counts().to_dict()),
       "EXP-062 정상 판정인데 선 안: " + str([f for f, e in zip(D.head(K).file, D.head(K)["EXP062"]) if e == "정상"]),
       "EXP-062 고장 판정인데 선 밖: " + str([f for f, e in zip(D.file[K:], D["EXP062"][K:]) if e == "고장"]),
       f"test 징후 하나만 {len(one)}개: {one} · 그중 선 오른쪽: " + str([f for f, s in zip(D.head(K).file, D.head(K)["징후"]) if s in ("H3만", "H8만")]),
       "test 전체 징후: " + str(pd.Series(sign).value_counts().to_dict()),
       f"ET logit 구간: 정상 유지(< -2) {nn} · 경계 [-2, 0) {nb} · ET 고장 쪽(≥ 0) {nf}"]
open(os.path.join(HERE, "17선_표.txt"), "w", encoding="utf-8").write("\n".join(txt) + "\n")

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
a.set_yticklabels([nte[i] + (" (062 고장)" if fin[i] else "") + (" *" if nte[i] in chg else "") for i in oo], fontsize=8)
for t, i in zip(a.get_yticklabels(), oo): t.set_color(ABN if fin[i] else "k")
for s_, (mk, c) in SM.items():
    a.plot([], [], mk, color=c, ms=8, label=f"{s_} ({int((sign == s_).sum())})")
a.legend(loc="lower right", fontsize=9, title="징후: H3 z>3 · H8 z>5", title_fontsize=9)
a.set_ylim(-1, len(oo)); a.set_xlim(-6.8, 6.8); a.grid(axis="x", alpha=.3)
k17 = D.head(K)["징후"].value_counts()
a.set_title("test ET logit 상위 17개(val 고장 비율 가정)에 선을 그으면 · 오른쪽 17개 징후 = " + " · ".join(f"{k} {v}" for k, v in k17.items()), fontsize=11, loc="left")
a.set_xlabel("ET logit (점선 -2 · 0, 빨간 선 = 상위 17번째) · 이름 주황 = EXP-062 고장 판정 · * = AE 제외에서 판정 바뀜")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "02_ET_17선_징후.png"), dpi=130)
print("done")
