# %% [markdown]
# T-1 그림: EXP-062 구성 3의 ET 단독 점수로 본 test (11-27 사용자: 박자 · AE 없이 ET만, V-1 방식 그림, 17번째 선 없음)
# - 저장된 모델(제출_EXP-062/모델/final_model_062.joblib)만 읽는다. 다시 학습하지 않는다
# - ET logit = ln(p / (1 - p)), p = ET 고장 확률. 학습 150은 안쪽 OOF logit(core.Z_[:, 0]), val · test는 학습 전체 ET로 채점
# - ET 단독 경계 = 학습 OOF ET logit에서 F2 최대 문턱(best_f2_threshold, 제출_EXP-062/3_평면그림.py와 같음)
# - 그림은 V-1 plot_경계.py와 같은 형식: 노란 띠 = 경계 구간 -2 ≤ logit < 0, 파일당 한 줄. 17번째 선 대신 ET 단독 경계(굵은 선)
# - 출력: figs/01_ET경계.png (위 train OOF · 아래 test, 색 = ET 단독 판정) · figs/02_ET_징후.png (모양 = 징후 H3 · H8, 충격) · ET단독_판정표.csv
#   conda run -n qe-phm python 6_TEST_단계/T-1_EXP-062_예측결과/자료/run.py
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
from src.evaluate import best_f2_threshold
from src.exp import outlier_idx
from src.data import load_folder

m = joblib.load(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "모델", "final_model_062.joblib")); core = m.core_
Ttr, Tva, y, ntr, nva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))
cache = os.path.join(ROOT, "4_outputs", "cache", "test_eda_tabs.joblib")          # test 표 · 충격 캐시 (없으면 만든다)
if os.path.exists(cache):
    Tte, nte, Ite = joblib.load(cache)
else:
    from src.impacts import all_impacts
    Xte, nte = load_folder(os.path.join(ROOT, "2_data", "TeamProject_test", "test"))
    Tte, Ite = m._tables(Xte), all_impacts(Xte, fmax=11000); joblib.dump((Tte, nte, Ite), cache)
short = lambda s: str(s).replace(".csv", "")
nte, nva, ntr = [short(n) for n in nte], [short(n) for n in nva], [short(n) for n in ntr]


def et_logit(T):
    E, _ = core._et_input(T)
    p = np.clip(core.et_.predict_proba(E)[:, 1], 1 / 600, 1 - 1 / 600)
    return np.log(p / (1 - p)), p


ltr = core.Z_[:, 0]
lva, pva = et_logit(Tva)
lte, pte = et_logit(Tte)
t_et = float(best_f2_threshold(y, ltr))
fin_te = core.parts(Tte).pred.values.astype(bool)                                  # 최종(결합 OR 규칙) 판정, 비교용
NOR, ABN, BG_N, BG_A, INK = "#2a78d6", "#eb6834", "#dce9f8", "#fbe3d6", "#1d2430"

# %% 판정표 (test, ET 단독)
D = pd.DataFrame({"file": nte, "ET_p": pte, "ET_logit": lte, "경계까지": lte - t_et,
                  "ET단독": np.where(lte >= t_et, "고장", "정상"), "최종_EXP062": np.where(fin_te, "고장", "정상")}).sort_values("ET_logit", ascending=False)
D.to_csv(os.path.join(HERE, "ET단독_판정표.csv"), index=False, encoding="utf-8-sig", float_format="%.3f")
n_et = int((lte >= t_et).sum())
diff = [f for f, a, b in zip(D.file, D["ET단독"], D["최종_EXP062"]) if a != b]
near = D[D["경계까지"].abs() < 0.5]
S = [f"ET 단독 경계 = 학습 OOF ET logit의 F2 최대 문턱 {t_et:+.3f}",
     f"test ET 단독 고장 {n_et}/40 · 최종(EXP-062) 고장 {int(fin_te.sum())}/40 · 판정이 다른 파일: {diff or '없음'}",
     "경계 ±0.5 안: " + " ".join(f"{f}({d:+.2f})" for f, d in zip(near.file, near["경계까지"])),
     "", D.to_string(index=False, float_format="%+.2f")]
open(os.path.join(HERE, "ET단독_요약.txt"), "w", encoding="utf-8").write("\n".join(S) + "\n")


LO, HI = -2.0, 0.0                                                                  # 경계 구간 (V-1과 같음)


def swarm(x, sx=0.12, sy=0.09):
    off = np.zeros(len(x)); placed = []
    for i in np.argsort(x):
        near_ = [j for j in placed if abs(x[i] - x[j]) < sx]
        for k in range(400):
            c = (k + 1) // 2 * (1 if k % 2 else -1) * sy
            if all(((x[i] - x[j]) / sx) ** 2 + ((c - off[j]) / sy) ** 2 >= 1 for j in near_):
                off[i] = c; break
        placed.append(i)
    return off


def frame(a):
    a.axvspan(LO, HI, color="#f7dc6f", alpha=.35, lw=0)
    for v in (LO, HI): a.axvline(v, color="k", lw=.8, ls="--")
    a.axvline(t_et, color=INK, lw=2.4)
    a.grid(axis="x", alpha=.3)


# %% 그림 1: 위 train OOF · 아래 test 파일당 한 줄 (색 = ET 단독 판정)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 14), sharex=True, gridspec_kw={"height_ratios": [1.3, 4]})
for a in (a1, a2): frame(a)
out = np.zeros(len(ntr), bool); out[outlier_idx(ntr)] = True
for row, (msk, col) in enumerate([(y == 0, NOR), (y == 1, ABN)]):
    xs = ltr[msk]; ys = row + swarm(xs); o_ = out[msk]
    a1.scatter(xs[~o_], ys[~o_], s=22, c=col, edgecolor="white", lw=.4)
    a1.scatter(xs[o_], ys[o_], s=50, c=col, marker="D", edgecolor="k", lw=.8)
a1.set_yticks([0, 1]); a1.set_yticklabels(["train 정상 (120)", "train 고장 (30)"]); a1.set_ylim(-.7, 1.7)
k = (ltr >= LO) & (ltr < HI)
a1.set_title(f"train OOF ET logit · 경계 구간 [-2, 0): 정상 {int((k & (y == 0)).sum())} · 고장 {int((k & (y == 1)).sum())}  (◆ = 이상치 정상 059·065·116)", fontsize=11, loc="left")
o = np.argsort(lte)
for r, i in enumerate(o):
    a2.plot([lte[i]], [r], "o", ms=8, color=ABN if lte[i] >= t_et else NOR, mec="white", mew=.5)
    a2.text(lte[i] + .12, r, f"{pte[i]:.2f}", va="center", fontsize=7, color="#555")
a2.set_yticks(range(len(o))); a2.set_yticklabels([nte[i] for i in o], fontsize=8); a2.set_ylim(-1, len(o) + 1)
a2.plot([], [], "o", color=ABN, label=f"ET 단독 고장 ({n_et})"); a2.plot([], [], "o", color=NOR, label=f"ET 단독 정상 ({40 - n_et})")
a2.plot([], [], color=INK, lw=2.4, label=f"ET 단독 경계 (logit {t_et:+.2f})")
a2.legend(loc="lower right", fontsize=9)
nb = int(((lte >= LO) & (lte < HI)).sum()); nf = int((lte >= HI).sum()); nn = int((lte < LO).sum())
for x, s in [((LO - 6.4) / 2 - 1, f"정상 유지 {nn}"), ((LO + HI) / 2, f"경계 {nb}"), ((HI + 6.4) / 2, f"ET 고장 쪽 {nf}")]:
    a2.text(x, len(o) + .1, s, ha="center", va="bottom", fontsize=11, weight="bold")
a2.set_title(f"test ET logit (파일당 한 줄, 숫자 = ET 확률) · ET 단독 판정 고장 {n_et} · 경계까지 고장으로 하면 {nf} + {nb} = {nf + nb}개", fontsize=11, loc="left", pad=22)
a2.set_xlabel("ET logit = ln(p / (1 - p))   (점선 -2 · 0, 노란 띠 = 경계 구간, 굵은 선 = ET 단독 경계)"); a2.set_xlim(-6.8, 6.8)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "01_ET경계.png"), dpi=130)

# %% 그림 2: 징후(H3 · H8) · 충격 (V-1 그림 2와 같음, 17번째 선 없음)
cols = list(core.cols) + list(core.peak)
Etr, _ = core._et_input(Ttr); nm = y == 0
Ete, _ = core._et_input(Tte)
med = np.median(Etr[nm], 0); mad = 1.4826 * np.median(np.abs(Etr[nm] - med), 0)
mad[mad < 1e-9] = Etr[nm].std(0)[mad < 1e-9] + 1e-9
Z = (Ete - med) / mad
i3, i8 = cols.index("H3_kurt_1-2k"), cols.index("H8_ring_2-4k")
s3, s8 = Z[:, i3] > 3, Z[:, i8] > 5
sign = np.where(s3 & s8, "둘 다", np.where(s3, "H3만", np.where(s8, "H8만", "없음")))
nimp = np.array([len(im["height_x"]) for im in Ite]); hmax = np.array([max(im["height_x"]) if len(im["height_x"]) else 0 for im in Ite])
SM = {"둘 다": ("o", "#c0392b"), "H3만": ("s", "#27ae60"), "H8만": ("^", "#2980b9"), "없음": ("x", "#7f8c8d")}
fig, a = plt.subplots(figsize=(11, 12))
frame(a)
a.text(t_et + .1, 1.5, f"ET 단독 경계\n(logit {t_et:+.2f})", color=INK, fontsize=10, ha="left", va="center", weight="bold")
oo = np.argsort(lte)
for r, i in enumerate(oo):
    mk, c = SM[sign[i]]
    a.plot([lte[i]], [r], mk, ms=9, color=c, mec=c if mk == "x" else "white", mew=2 if mk == "x" else .5)
    a.text(lte[i] + .15, r, f"{pte[i]:.2f} · 충격 {nimp[i]} · 최대 {hmax[i]:.0f}배", va="center", fontsize=7, color="#555")
a.set_yticks(range(len(oo)))
a.set_yticklabels([nte[i] + (" (고장)" if lte[i] >= t_et else "") for i in oo], fontsize=8)
for t, i in zip(a.get_yticklabels(), oo): t.set_color(ABN if lte[i] >= t_et else "k")
for s_, (mk, c) in SM.items():
    a.plot([], [], mk, color=c, ms=8, label=f"{s_} ({int((sign == s_).sum())})")
a.legend(loc="lower right", fontsize=9, title="징후: H3 z>3 · H8 z>5", title_fontsize=9)
a.set_ylim(-1, len(oo)); a.set_xlim(-6.8, 6.8)
kh = pd.Series(sign[lte >= t_et]).value_counts()
a.set_title(f"test ET logit · ET 단독 고장 {n_et}개의 징후 = " + " · ".join(f"{k_} {v}" for k_, v in kh.items()), fontsize=11, loc="left")
a.set_xlabel("ET logit (점선 -2 · 0, 노란 띠 = 경계 구간, 굵은 선 = ET 단독 경계) · 이름 주황 = ET 단독 고장")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "02_ET_징후.png"), dpi=130)
print("done", t_et, n_et, diff, dict(kh))
