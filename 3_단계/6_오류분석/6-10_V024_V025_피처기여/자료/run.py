# %% [markdown]
# 6-10 (11-27) val_024 · val_025 — 최종 모델(EXP-062 구성 3)은 왜 둘 다 고장으로 봤는가 (피처 기여)
# - 효섭 RF는 013 · 024 · 025를 정상으로 봤다(013은 고장 확정, 024 · 025 중 하나는 반드시 고장). 우리는 셋 다 고장 → 024 · 025의 근거를 본다
# - ① 결합 분해: 여유 = w_ET·ET + w_박자·박자 z + w_AE·AE z + 절편 − 문턱 (제출 모델, 학습 150개 전체)
# - ② ET 피처 기여: 그 열 하나를 학습 정상 중앙값으로 바꿨을 때 ET logit이 얼마나 내려가는가 × w_ET (여유 단위, + = 고장 쪽)
# - ③ AE 대역 오차: 8 구간 log PSD 복원 오차를 대역별로 (학습 정상 · val 정상 판정 파일과 비교)
# - 학습하지 않는다(모델 저장본만 읽음). 기준 파일 013 = 고장 확정, 비교 = val에서 정상으로 판정한 30개의 중앙값
# 실행: conda run -n qe-phm python run.py   (처음 한 번 학습 150개 표를 계산해 4_outputs/cache에 저장)
# %% 설정
import os, sys, pickle, warnings
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.data import load_dataset, load_folder
from src.final import _lg, _ae_score
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
INK, MUTED, GRID = "#1d2430", "#5b6575", "#d9dde3"
UP, DOWN = "#d9622b", "#3b7dd8"           # 고장 쪽(+) 주황, 정상 쪽(−) 파랑 (프로젝트 결합 그림과 같은 색)
TGT = ["val_013", "val_024", "val_025"]
M = joblib.load(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "모델", "final_model_062.joblib")); C = M.core_
w = C.c_.coef_[0]; b = float(C.c_.intercept_[0]); tc = C.tc_

# %% 학습 150 · val 40 표
cache = os.path.join(ROOT, "4_outputs", "cache", "val_attrib_062.pkl"); os.makedirs(os.path.dirname(cache), exist_ok=True)
Xv, nv = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val")); nv = [str(s) for s in nv]
if os.path.exists(cache):
    Xe_tr, y = pickle.load(open(cache, "rb"))
else:
    X, y, names = load_dataset(os.path.join(ROOT, "2_data", "TeamProject_train_변경"))
    Xe_tr = C._et_input(M._tables(X))[0]; pickle.dump((Xe_tr, y), open(cache, "wb"))
Tv = M._tables(Xv); Xe_v, _ = C._et_input(Tv); P = C.parts(Tv); P.insert(0, "file", nv)
cols = list(C.cols) + list(C.peak); med0 = np.median(Xe_tr[y == 0], 0)
pct = lambda j, v, lab: float((Xe_tr[y == lab, j] < v).mean() * 100)
assert (P.pred.values == pd.read_csv(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "출력", "근거표_submission_val_9조.csv"), encoding="utf-8-sig").pred.values).all(), "저장 출력과 판정이 다름"

# %% ① 결합 분해 · ② ET 피처 기여
rows = {}
for t in TGT:
    k = nv.index(t); r = P.iloc[k]; xe = Xe_v[k:k + 1]; a = float(r.et_logit)
    con = []
    for j, cn in enumerate(cols):
        h = xe.copy(); h[0, j] = med0[j]
        con.append(w[0] * (a - float(_lg(C.et_.predict_proba(h)[:, 1])[0])))
    rows[t] = dict(ET=w[0] * a, 박자=w[1] * r.beat_z, AE=w[2] * r.ae_z, 절편=b, 문턱=tc, 여유=float(r.margin), con=np.array(con), xe=xe[0], k=k)
    print(t, {q: round(v, 2) for q, v in rows[t].items() if q in ("ET", "박자", "AE", "절편", "문턱", "여유")}, flush=True)

# 결합 분해 그림
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for ax, t in zip(axs, TGT):
    d = rows[t]; steps = [("절편", d["절편"]), ("ET", d["ET"]), ("박자 z", d["박자"]), ("AE z", d["AE"]), ("- 문턱", -d["문턱"])]; cum = 0
    for i, (lab, v) in enumerate(steps):
        ax.bar(i, v, bottom=cum, color=UP if v > 0 else DOWN, width=.62); ax.text(i, cum + v + (0.06 if v >= 0 else -0.06), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
        cum += v
    ax.bar(len(steps), cum, color=INK, width=.62); ax.text(len(steps), cum + 0.06, f"{cum:+.2f}", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylim(0, 4.6); ax.axhline(0, color=INK, lw=1); ax.set_xticks(range(len(steps) + 1)); ax.set_xticklabels([s[0] for s in steps] + ["여유"], fontsize=9)
    ax.set_title(f"{t}  ({'고장 확정' if t == 'val_013' else '판정 고장'})", fontsize=11, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("결합 여유 (0 = 문턱, 위 = 고장)", color=MUTED)
fig.suptitle("결합 점수 분해 — 여유 = 절편 + w·ET + w·박자 + w·AE - 문턱", x=.01, ha="left", fontsize=12, color=INK); fig.tight_layout(); fig.savefig(os.path.join(FIG, "결합분해.png"), dpi=150); plt.close(fig)

# ET 피처 기여 그림 (024 · 025 나란히, 013 참고)
order = np.argsort(-np.abs(rows["val_024"]["con"]) - np.abs(rows["val_025"]["con"]))[:10][::-1]
fig, ax = plt.subplots(figsize=(10, 5.6)); h = .26
for i, (t, off, alpha) in enumerate(zip(["val_024", "val_025", "val_013"], [h, 0, -h], [1, 1, .35])):
    v = rows[t]["con"][order]
    ax.barh(np.arange(len(order)) + off, v, height=h * .92, color=[UP if q > 0 else DOWN for q in v], alpha=alpha, edgecolor=INK if t != "val_013" else "none", linewidth=.6, label=t)
    for yy, q in zip(np.arange(len(order)) + off, v):
        ax.text(q + (0.01 if q >= 0 else -0.01), yy, f"{q:+.2f}", va="center", ha="left" if q >= 0 else "right", fontsize=7.5, color=INK)
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([cols[j] for j in order], fontsize=9); ax.axvline(0, color=INK, lw=1)
ax.set_xlabel("ET 피처 기여 (여유 단위, + = 고장 쪽)  · 위 막대 val_024 · 가운데 val_025 · 아래(연하게) val_013", color=MUTED)
ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("ET가 고장 쪽으로 민 피처 (그 열만 학습 정상 중앙값으로 바꿨을 때의 변화)", loc="left", fontsize=12, color=INK); fig.tight_layout(); fig.savefig(os.path.join(FIG, "ET_피처기여.png"), dpi=150); plt.close(fig)

# %% ③ AE 대역 오차
mu, sd, ae = C.ae_; LPv = Tv["LP"]; Z = (LPv - mu) / sd
R = ae.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape); E = ((R - Z) ** 2).mean(1)      # (n, bin) 구간 평균 제곱 오차
fr = np.arange(E.shape[1]) * (25000 / 2048); BANDS = [(200, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 11000)]
BE = np.column_stack([np.log10(E[:, (fr >= lo) & (fr < hi)].mean(1)) for lo, hi in BANDS]); lab = [f"{lo / 1000:g}–{hi / 1000:g}k" for lo, hi in BANDS]
ok = (P.pred.values == 0); ref = np.median(BE[ok], 0)
fig, ax = plt.subplots(figsize=(9, 4.4)); xs = np.arange(len(BANDS))
ax.fill_between(xs, np.percentile(BE[ok], 10, 0), np.percentile(BE[ok], 90, 0), color=GRID, alpha=.8, label="val 정상 판정 30개 10–90%")
ax.plot(xs, ref, color=MUTED, lw=1.5, label="val 정상 판정 중앙값")
for t, c_, mk in zip(TGT, ["#8a63d2", UP, "#e0a30b"], ["s", "o", "^"]):
    ax.plot(xs, BE[nv.index(t)], color=c_, marker=mk, lw=1.8, label=f"{t} (AE z {P.ae_z[nv.index(t)]:+.2f})")
ax.set_xticks(xs); ax.set_xticklabels(lab); ax.set_ylabel("log10 복원 오차 (대역 평균)", color=MUTED); ax.grid(color=GRID, lw=.6); ax.legend(fontsize=8.5, frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("AE가 어느 대역에서 정상 모양을 벗어났다고 보는가", loc="left", fontsize=12, color=INK); fig.tight_layout(); fig.savefig(os.path.join(FIG, "AE_대역오차.png"), dpi=150); plt.close(fig)

# %% 표 (md)
L = ["# 표 — val_013 · 024 · 025 판정 근거 (최종 모델 EXP-062 구성 3)", "", "## 결합 분해 (여유 = 절편 + ET + 박자 + AE − 문턱, 문턱 %.3f)" % tc, "",
     "| 파일 | ET logit | 박자 z | AE z | w·ET | w·박자 | w·AE | 절편 | 여유 | n_imp |", "|---|---|---|---|---|---|---|---|---|---|"]
for t in TGT:
    r = P.iloc[rows[t]["k"]]; d = rows[t]
    L.append(f"| {t} | {r.et_logit:+.2f} | {r.beat_z:+.2f} | {r.ae_z:+.2f} | {d['ET']:+.2f} | {d['박자']:+.2f} | {d['AE']:+.2f} | {d['절편']:+.2f} | **{d['여유']:+.2f}** | {r.n_imp} |")
L += ["", f"계수 w = ET {w[0]:+.3f} · 박자 {w[1]:+.3f} · AE {w[2]:+.3f}, 절편 {b:+.3f}", "", "## ET 피처 기여와 값 (기여 = 여유 단위, 백분위 = 학습 정상 / 학습 고장 중 이 값보다 낮은 비율 %)", "",
     "| 피처 | " + " | ".join(f"{t} 기여 · 값 · 정상% · 고장%" for t in TGT) + " |", "|---|" + "---|" * len(TGT)]
for j in np.argsort(-np.abs(rows["val_024"]["con"]) - np.abs(rows["val_025"]["con"])):
    L.append(f"| {cols[j]} | " + " | ".join(f"{rows[t]['con'][j]:+.2f} · {rows[t]['xe'][j]:.3g} · {pct(j, rows[t]['xe'][j], 0):.0f} · {pct(j, rows[t]['xe'][j], 1):.0f}" for t in TGT) + " |")
L += ["", "## AE 대역 오차 (log10, val 정상 판정 30개 중앙값과의 차)", "", "| 파일 | " + " | ".join(lab) + " |", "|---|" + "---|" * len(lab)]
for t in TGT:
    L.append(f"| {t} | " + " | ".join(f"{q:+.2f}" for q in BE[nv.index(t)] - ref) + " |")
open(os.path.join(HERE, "표_V024_V025.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
