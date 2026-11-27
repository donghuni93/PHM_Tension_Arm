# %% [markdown]
# T-1 그림: EXP-062 구성 3의 ET 단독 점수로 본 학습 · val · test (11-27 사용자: 박자 · AE 없이 ET만, 경계가 잘 보이게)
# - 저장된 모델(제출_EXP-062/모델/final_model_062.joblib)만 읽는다. 다시 학습하지 않는다
# - ET logit = ln(p / (1 - p)), p = ET 고장 확률. 학습 150은 안쪽 OOF logit(core.Z_[:, 0]), val · test는 학습 전체 ET로 채점
# - ET 단독 경계 = 학습 OOF ET logit에서 F2 최대 문턱(best_f2_threshold, 제출_EXP-062/3_평면그림.py와 같음)
# - 출력: figs/01_test_ET단독.png (test 40, 파일당 한 줄) · figs/02_학습_val_test_ET단독.png (세 묶음, 경계 근처 확대) · ET단독_판정표.csv
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
cache = os.path.join(ROOT, "4_outputs", "cache", "test_eda_tabs.joblib")          # T-2가 만든 캐시 (없으면 만든다)
if os.path.exists(cache):
    Tte, nte, _ = joblib.load(cache)
else:
    from src.impacts import all_impacts
    Xte, nte = load_folder(os.path.join(ROOT, "2_data", "TeamProject_test", "test"))
    Tte = m._tables(Xte); joblib.dump((Tte, nte, all_impacts(Xte, fmax=11000)), cache)
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


def shade(a, lo, hi):
    a.axvspan(lo, t_et, color=BG_N, lw=0, zorder=0); a.axvspan(t_et, hi, color=BG_A, lw=0, zorder=0)
    a.axvline(t_et, color=INK, lw=2.4, zorder=3)
    a.set_xlim(lo, hi)


# %% 그림 1: test 40, 파일당 한 줄 (ET logit 순)
o = np.argsort(lte)
lo, hi = min(-5, lte.min() - .4), max(4, lte.max() + .9)
fig, a = plt.subplots(figsize=(10, 11.5))
shade(a, lo, hi)
for r, i in enumerate(o):
    hit = lte[i] >= t_et
    a.plot([lo, lte[i]], [r, r], color="#9aa5b1", lw=.6, zorder=1)
    a.plot(lte[i], r, "o", ms=9, color=ABN if hit else NOR, mec="white", mew=.8, zorder=4)
    a.text(lte[i] + .12, r, f"{pte[i]:.2f}  ({lte[i] - t_et:+.2f})", va="center", fontsize=7.5, color="#444", zorder=4)
a.set_yticks(range(len(o)))
a.set_yticklabels([nte[i] for i in o], fontsize=8)
for t, i in zip(a.get_yticklabels(), o):
    if abs(lte[i] - t_et) < .5: t.set_fontweight("bold")
a.set_ylim(-1, len(o) + 1.2)
a.text(t_et - .1, len(o) + .3, f"← 정상 {40 - n_et}", ha="right", va="bottom", fontsize=12, color=NOR, weight="bold")
a.text(t_et + .1, len(o) + .3, f"고장 {n_et} →", ha="left", va="bottom", fontsize=12, color=ABN, weight="bold")
a.text(t_et + .15, 3, f"◀ ET 단독 경계\n   logit {t_et:+.2f}\n   (학습 OOF F2 최대)", fontsize=10, color=INK, ha="left", va="center")   # 아래쪽 고장 칸은 비어 있음
a.grid(axis="x", alpha=.3)
a.set_xlabel("ET logit = ln(p / (1 - p))   · 굵은 세로선 = ET 단독 경계 · 숫자 = ET 확률 (경계까지 거리) · 굵은 이름 = 경계 ±0.5 안")
a.set_title(f"test 40 — ET 단독 판정: 고장 {n_et} · 정상 {40 - n_et}   (EXP-062 구성 3의 ET만, 박자 · AE 없음)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "01_test_ET단독.png"), dpi=130)


# %% 그림 2: 학습 · val · test 세 묶음, 경계 근처 확대
def swarm(x, sx=0.07, sy=0.11):
    off = np.zeros(len(x)); placed = []
    for i in np.argsort(x):
        near_ = [j for j in placed if abs(x[i] - x[j]) < sx]
        for k in range(600):
            c = (k + 1) // 2 * (1 if k % 2 else -1) * sy
            if all(((x[i] - x[j]) / sx) ** 2 + ((c - off[j]) / sy) ** 2 >= 1 for j in near_):
                off[i] = c; break
        placed.append(i)
    return off


LO2, HI2 = t_et - 2.5, t_et + 2.5                                                  # 경계 ±2.5 확대, 밖은 끝에 모아 화살표로 표시
out = np.zeros(len(ntr), bool); out[outlier_idx(ntr)] = True
rows = [("학습 정상 120\n(OOF)", ltr[y == 0], np.array(ntr)[y == 0], "s", NOR, out[y == 0]),
        ("학습 고장 30\n(OOF)", ltr[y == 1], np.array(ntr)[y == 1], "o", ABN, out[y == 1]),
        ("val 40", lva, np.array(nva), "^", None, None),
        ("test 40", lte, np.array(nte), "^", None, None)]
fig, axs = plt.subplots(len(rows), 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0, 1.6, 1.6]})
for a, (lab, x, nm, mk, col, ou) in zip(axs, rows):
    shade(a, LO2, HI2)
    inr = (x >= LO2) & (x <= HI2)                                                  # 범위 밖 점은 그리지 않고 개수만 적는다
    yy = np.zeros(len(x)); yy[inr] = swarm(x[inr])
    for i in np.where(inr)[0]:
        c = col or (ABN if x[i] >= t_et else NOR)
        m_ = "D" if (ou is not None and ou[i]) else mk
        a.plot(x[i], yy[i], m_, ms=7 if m_ != "D" else 8, color=c, mec="k" if m_ == "D" else "white", mew=.8 if m_ == "D" else .5, zorder=4)
    if col is None:                                                                # 경계 ±1 안 이름: x 순으로 4단 엇갈려 겹침 방지
        lab_i = [i for i in np.argsort(x) if abs(x[i] - t_et) < 1.0]
        for k_, i in enumerate(lab_i):
            dy = [14, 30, -16, -32][k_ % 4]
            a.annotate(nm[i].replace("val_", "V").replace("test_", "T"), (x[i], yy[i]), xytext=(0, dy), textcoords="offset points", ha="center",
                       va="center", fontsize=7.5, zorder=5, arrowprops=dict(arrowstyle="-", color="#888", lw=.5, shrinkA=0, shrinkB=3))
    nl, nr = int((x < LO2).sum()), int((x > HI2).sum())
    top = (yy[inr].max() if inr.any() else 0) + (1.0 if col is None else .3)
    bot = (yy[inr].min() if inr.any() else 0) - (1.0 if col is None else .3)
    if nl: a.text(LO2 + .05, top - .05, f"◀ {nl}개 더 왼쪽", fontsize=8.5, color="#555", va="top")
    if nr: a.text(HI2 - .05, top - .05, f"{nr}개 더 오른쪽 ▶", fontsize=8.5, color="#555", ha="right", va="top")
    nh = int((x >= t_et).sum())
    a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
    a.set_yticks([]); a.set_ylim(bot, top)
    a.text(1.003, .5, f"고장 쪽 {nh}\n정상 쪽 {len(x) - nh}", transform=a.transAxes, fontsize=9, va="center")
    a.grid(axis="x", alpha=.3)
axs[0].set_title(f"EXP-062 구성 3의 ET 단독 점수 — 굵은 세로선 = ET 단독 경계 (logit {t_et:+.2f}, 학습 OOF F2 최대) · 경계 ±2.5 확대 · 이름 = val · test 중 경계 ±1 안"
                 "\n◆ = 이상치 정상 059 · 065 · 116 · 범위 밖 점은 개수만 · 색 = ET 단독 판정(학습은 라벨)", loc="left", fontsize=10)
axs[-1].set_xlabel("ET logit = ln(p / (1 - p))")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "02_학습_val_test_ET단독.png"), dpi=130)
print("done", t_et, n_et, diff)
