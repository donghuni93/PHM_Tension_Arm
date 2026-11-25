# %% [markdown]
# PHO-007 (휴대폰 세션, 11-22) val 의사 정상으로 AE 기준 다시 맞추기 (비지도 도메인 적응, 라벨 안 씀)
# - 문제: val 배경 3–5 kHz가 통째로 +10.9 dB(2-19) → 학습 정상만 본 AE가 val 전체를 이상으로 읽음(AE z 중앙 +1.7)
# - 의사 정상 선택 (실행 전 고정, AE는 쓰지 않음 = 순환 방지): val 중
#     ET logit(학습 150 전체 모델) < 학습 정상 ET(안쪽 OOF) 중앙값  AND  박자 z ≤ 0  AND  E2 · 8.3k 규칙 꺼짐
# - 2-a 눈금만: AE 모델 그대로, AE z 중심만 의사 정상의 AE 원점수 중앙값으로 옮김 (척도 = 학습 정상 MAD 그대로)
# - 2-b 재학습: 학습 정상 120 + 의사 정상으로 AE 다시 학습. 의사 정상 자신의 점수는 4묶음 교차(자기 자신을 뺀 AE)로
# - 학습 150의 성능(J 0.977)은 바뀌지 않음(학습 쪽은 손대지 않음). 바뀌는 것은 val 판정뿐 → val 판정 수 · 경계 파일 이동을 본다
# - 결정.md 'val을 보고 모델을 고치지 않는다'와 충돌 → 제출 반영은 팀 결정(11/23 회의). 이 실험은 결과 확인용
# 실행: QE_FULL=1 python "단계/5_모델실험/아카이브/PHO-007_1122_val의사정상AE/run.py"   (EXP-044와 같은 ET 300)

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import FULL
from src.exp2util import NS, Add83, rule83_flags, short
from src.models2 import TwoAE
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")
assert FULL, "QE_FULL=1 로 실행 (EXP-044와 같은 ET 300)"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
ex = Experiment(__file__, "PHO-007", "val 의사 정상으로 AE 기준 다시 맞추기", ["H3"],
                member="이동훈", model="J + AE 기준 재보정 (2-a 눈금 · 2-b 재학습)",
                changes="val 중 ET로 고른 의사 정상으로 AE z 중심을 옮기거나(2-a) AE를 다시 학습(2-b). 라벨 안 씀, 학습 150 성능 불변",
                key_changes=["의사 정상 = val 중 ET < 학습 정상 ET 중앙값 · 박자 z ≤ 0 · 규칙 꺼짐 (AE 안 씀)", "2-a: AE z 중심만 의사 정상 중앙값으로", "2-b: 학습 정상 + 의사 정상으로 AE 재학습"])
ns = NS(); y = ns.y; nor = y == 0; nv = [n[-3:] for n in ns.nv]; NV = len(nv)
LOG = []
def say(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); LOG.append(s)

# %% J 학습 (EXP-044와 같음) · val 원점수
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float); sv, rv = rule83_flags(ns.Xv); r83v = (sv | rv).astype(float)
M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), r83]); Mv = np.column_stack([ns.Mv, np.zeros(NV), r83v])
mJ = Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK))).fit(M, y); mm = mJ.m_
_, LPv = segment_logpsd(ns.Xv)
ae0 = _fit_ae(LP_TR[nor]); raw0 = _ae_score(ae0, LPv)
iqr = np.subtract(*np.percentile(mm.c_.decision_function(mm.Z_)[nor], [75, 25]))

def judge(raw, aef=None):
    mm.ae_override_ = raw; keep = mm.aef_
    if aef is not None: mm.aef_ = aef
    f = mJ.fires(Mv); mm.aef_ = keep
    fin = f["main"] | f["E2"] | f["R83"]
    return dict(fin=fin, mar=(f["score"] - mm.tc_) / iqr, a=f["a"], ae=f["ae"], z=f["z"], E2=f["E2"], R83=f["R83"])

J0 = judge(raw0)

# %% 의사 정상 고르기 (AE 안 씀)
et_med = float(np.median(mm.Z_[nor, 0]))
pseudo = (J0["a"] < et_med) & (J0["z"] <= 0) & ~J0["E2"] & ~J0["R83"]
say(f"학습 정상 ET(OOF) 중앙값 {et_med:.2f} → 의사 정상 {int(pseudo.sum())}/40: {' '.join(np.array(nv)[pseudo])}")
say(f"(참고) 학습 고장 중 ET(OOF) < 중앙값: {int((mm.Z_[~nor, 0] < et_med).sum())}/30")

# %% 2-a 눈금만: 중심을 의사 정상 원점수 중앙값으로 (척도는 학습 정상 MAD 그대로)
sc_tr = 1.0 / (mm.aef_(np.array([1.0]))[0] - mm.aef_(np.array([0.0]))[0])         # robust_z의 척도 복원
c_ps = float(np.median(raw0[pseudo]))
aef_a = lambda v: (np.asarray(v) - c_ps) / sc_tr
A = judge(raw0, aef_a)

# %% 2-b 재학습: 학습 정상 + 의사 정상. 의사 정상 자신은 4묶음 교차로 점수
ip = np.flatnonzero(pseudo); raw_b = np.empty(NV)
aeb = _fit_ae(np.concatenate([LP_TR[nor], LPv[ip]])); raw_b[~pseudo] = _ae_score(aeb, LPv[~pseudo])
for k, (tr, te) in enumerate(KFold(4, shuffle=True, random_state=0).split(ip)):
    m_ = _fit_ae(np.concatenate([LP_TR[nor], LPv[ip[tr]]])); raw_b[ip[te]] = _ae_score(m_, LPv[ip[te]])
B = judge(raw_b)

# %% 결과 표
R = {"J (EXP-044)": J0, "2-a 눈금만": A, "2-b 재학습": B}
say("\n구성 | val 고장 판정 | AE z 중앙 (val 전체 · 의사 정상 · 나머지) | 고장 판정 파일")
for k, r in R.items():
    say(f"{k} | {int(r['fin'].sum())}/40 | {np.median(r['ae']):+.2f} · {np.median(r['ae'][pseudo]):+.2f} · {np.median(r['ae'][~pseudo]):+.2f} | {' '.join(np.array(nv)[r['fin']])}")
BORDER = ["027", "025", "018", "021", "012", "036", "009", "001", "031", "024"]
say("\n경계 파일 여유 (IQR 단위, + 고장 쪽): J → 2-a → 2-b")
rows = []
for t in BORDER:
    i = nv.index(t); lab = lambda r: ("고장" if r["fin"][i] else "정상") + ("(규칙)" if r["R83"][i] or r["E2"][i] else "")
    say(f"{t}: {J0['mar'][i]:+.2f} {lab(J0)} → {A['mar'][i]:+.2f} {lab(A)} → {B['mar'][i]:+.2f} {lab(B)} | AE z {J0['ae'][i]:+.2f} → {A['ae'][i]:+.2f} → {B['ae'][i]:+.2f} | 의사정상 {'O' if pseudo[i] else ''}")
    rows.append(dict(val=t, 의사정상="O" if pseudo[i] else "", J=f"{lab(J0)} {J0['mar'][i]:+.2f}", **{"2-a": f"{lab(A)} {A['mar'][i]:+.2f}", "2-b": f"{lab(B)} {B['mar'][i]:+.2f}"},
                    AE_z=f"{J0['ae'][i]:+.2f} → {A['ae'][i]:+.2f} → {B['ae'][i]:+.2f}"))
chg = {k: [nv[i] for i in np.flatnonzero(r["fin"] != J0["fin"])] for k, r in R.items() if k != "J (EXP-044)"}
say(f"\nJ 대비 판정이 바뀐 파일: {chg}")
ex.note("결과", {k: dict(n_fault=int(r["fin"].sum()), faults=list(np.array(nv)[r["fin"]]), ae_med=round(float(np.median(r["ae"])), 2)) for k, r in R.items()})
ex.note("의사정상", dict(rule="ET < 학습 정상 ET 중앙값 & 박자 z ≤ 0 & 규칙 꺼짐", et_med=round(et_med, 3), files=list(np.array(nv)[pseudo])))
ex.note("바뀐 파일", chg)
import pandas as pd
ex.save_md("경계파일_표", "# PHO-007 경계 파일 (여유 = 결합 점수 − 경계, 학습 정상 IQR 단위, + 고장 쪽)\n\n" + md_table(pd.DataFrame(rows)) + "\n")

# %% 핵심 그림: val 평면 세 장 (배경 = 최종 판단 영역, 박자 z = 0 단면)
BG = ListedColormap(["#d6e6fa", "#fbe0d0"]); C_N, C_A = "#2a78d6", "#e0622a"
oa = mm.Z_[:, 0]                                        # 학습 ET(안쪽 OOF), 축 범위용
fig, axs = plt.subplots(1, 3, figsize=(21, 7.2), sharey=True)
allae = np.concatenate([r["ae"] for r in R.values()]); x0, x1 = min(J0["a"].min(), oa.min()) - .3, max(J0["a"].max(), oa.max()) + .3; y0, y1 = min(-2.4, allae.min() - .3), max(3.8, allae.max() + .3)
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)); D = (mm._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape) >= mm.tc_).astype(int)
for ax, (k, r) in zip(axs, R.items()):
    ax.contourf(xx, yy, D, levels=[-.5, .5, 1.5], cmap=BG, zorder=0); ax.contour(xx, yy, D, levels=[.5], colors="#333", linewidths=1.1, linestyles="--", zorder=1)
    ax.scatter(r["a"][pseudo], r["ae"][pseudo], s=190, facecolor="none", edgecolor="#1f9d55", lw=1.6, zorder=3)
    ax.scatter(r["a"], r["ae"], marker="^", s=85, c=np.where(r["fin"], C_A, C_N), edgecolor="k", lw=.7, zorder=4)
    for t in BORDER:
        i = nv.index(t); ax.annotate(t, (r["a"][i], r["ae"][i]), xytext=(6, 5), textcoords="offset points", fontsize=9, weight="bold", zorder=5)
    ax.axhline(0, color="#666", lw=.8, ls=":"); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.25); ax.set_xlabel("ET 점수 (logit)")
    ax.set_title(f"{k} — val 고장 {int(r['fin'].sum())}/40 · AE z 중앙 {np.median(r['ae']):+.2f}", fontsize=11)
axs[0].set_ylabel("AE z (점선 = 학습 정상 중심 0)")
axs[0].legend(handles=[Patch(color="#d6e6fa", label="정상 판단 영역"), Patch(color="#fbe0d0", label="고장 판단 영역"), Line2D([], [], marker="^", ls="", mfc=C_N, mec="k", ms=9, label="val → 정상"),
                       Line2D([], [], marker="^", ls="", mfc=C_A, mec="k", ms=9, label="val → 고장"), Line2D([], [], marker="o", ls="", mfc="none", mec="#1f9d55", mew=1.6, ms=12, label=f"의사 정상 ({int(pseudo.sum())})")],
              fontsize=9, loc="lower left", framealpha=.95)
fig.tight_layout(); ex.save_fig(fig, "val_평면_3구성", "val 40개. 왼쪽 J, 가운데 2-a(AE z 중심만 의사 정상으로), 오른쪽 2-b(AE 재학습). 배경 = J 결합 경계(박자 z = 0 단면)", key=True)

# %% 보조 그림: AE z 분포 (학습 정상 OOF vs val, 세 구성)
fig, ax = plt.subplots(figsize=(10, 4.5)); bins = np.linspace(-3, 5, 41)
ax.hist(mm.Z_[nor, 2], bins, color="#8a8f99", alpha=.5, density=True, label="학습 정상 (OOF)")
for (k, r), c in zip(R.items(), ("#c0392b", "#e08e0b", "#1f9d55")):
    ax.hist(r["ae"], bins, histtype="step", lw=2, color=c, density=True, label=f"val · {k}")
ax.axvline(0, color="#333", lw=.8, ls=":"); ax.set_xlabel("AE z"); ax.legend(fontsize=9); ax.grid(alpha=.25)
fig.tight_layout(); ex.save_fig(fig, "AE_z_분포", "학습 정상(OOF) AE z와 val AE z 분포. 재보정 후 val이 0 근처로 오는지")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt"), "w", encoding="utf-8").write("\n".join(LOG))
print("done")
