# %% [markdown]
# Q3 최종 모델(EXP-034 c, outputs/final/final_model.joblib) val 채점 분포 (사령탑 확인 요청, 11-21)
# - 모델은 고치지 않고 predict_table / core_.oof_z_ 를 읽기만 한다. 제출 파일은 만들지 않는다
# - train = 모델이 저장한 4등분 OOF z (자기가 학습에 안 쓰인 상태), val = 학습 전체로 다시 학습한 전문가의 점수
#   (둘은 학습 크기가 달라 눈금이 조금 다를 수 있음 → README 해석 주의)
# - z = (점수 - 정상 중앙값) / (정상 90% 선 - 정상 중앙값): 1 = 정상의 90% 선
# 실행: conda run -n qe-phm python run.py
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
p = HERE
while not os.path.isdir(os.path.join(p, "src")): p = os.path.dirname(p)
ROOT = p; sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.exp import load_train
from src.data import load_folder
from src.final import signal_tables
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
NOR, ABN, VAL = "#2a78d6", "#eb6834", "#2a9d5c"
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
m = joblib.load(os.path.join(ROOT, "outputs", "final", "final_model.joblib"))
X, y, names = load_train(); y = np.asarray(y); names = [str(n) for n in names]
XV, vn = load_folder(os.path.join(ROOT, "data", "TeamProject_val", "val"))
core = m.core_
out = [f"모델: ring={m.ring}, thr={m.thr_}, 학습 파일 {len(y)} (정상 {(y==0).sum()}, 고장 {(y==1).sum()})"]

# %% train: OOF z, 규칙 켜짐(최종 모델 규칙, 학습 신호)
Z_oof = core.oof_z_; lr_oof = core.lr_.decision_function(Z_oof)
Tt, _ = signal_tables(X, m.cols_); Dt = core.parts(Tt)       # 규칙 켜짐 판정용만 사용 (ae_z/et_z는 학습에 쓰인 점수라 안 씀)
rule_t = (Dt.rule83 | Dt.knock).values
oof_score = m.oof_.mean(0)                                    # 3 seed 평균 OOF 최종 점수
Dv = m.predict_table(XV, vn)
rule_v = (Dv.rule83 | Dv.knock).values
w = core.lr_.coef_[0]; b = float(core.lr_.intercept_[0])
out.append(f"로지스틱: lr = {w[0]:+.3f} * AE z + {w[1]:+.3f} * ET z + {b:+.3f}")

tbl = pd.DataFrame({"file": names, "y": y, "ae_z": Z_oof[:, 0], "et_z": Z_oof[:, 1], "lr": lr_oof, "rule": rule_t, "oof_score": oof_score})
tbl.to_csv(os.path.join(HERE, "train_oof.csv"), index=False, encoding="utf-8-sig")
Dv.to_csv(os.path.join(HERE, "val_table.csv"), index=False, encoding="utf-8-sig")

# %% 숫자
def q(a, p_): return float(np.percentile(a, p_))
nz = tbl[tbl.y == 0]
for c in ("ae_z", "et_z", "lr"):
    v = Dv[c].values
    out.append(f"[{c}] train 정상 OOF 중앙 {nz[c].median():+.2f} · 90% {q(nz[c],90):+.2f} · 99% {q(nz[c],99):+.2f} | 규칙 안 켜진 고장 OOF 중앙 {tbl[(tbl.y==1)&~tbl.rule][c].median():+.2f}"
               f" | val 평균 {v.mean():+.2f} · 중앙 {np.median(v):+.2f} · 정상 99% 초과 {(v > q(nz[c],99)).sum()}/40")
out.append(f"val 판정: 고장 {int(Dv.pred.sum())}/40 (임계값 {m.thr_}) · 규칙 고장 {int(rule_v.sum())} (8.3k 스펙트럼 {int(Dv.spec83.sum())}, 공진 {int(Dv.res83.sum())}, 유격 {int(Dv.knock.sum())}) · 본체 내림 {int(Dv.body.sum())} · 경계 {int(Dv.borderline.sum())}")
out.append(f"train OOF 정상 중 lr>=임계 {(nz.lr >= m.thr_).sum()}/120, 고장 중 규칙 없이 lr>=임계 {((tbl.y==1)&~tbl.rule&(tbl.lr>=m.thr_)).sum()}/{((tbl.y==1)&~tbl.rule).sum()}")
out.append("")
out.append("val 점수 상위 15:")
for r in Dv.sort_values("score", ascending=False).head(15).itertuples():
    out.append(f"  {r.file}  score {r.score:7.2f}  lr {r.lr:+.2f}  ae_z {r.ae_z:+.2f}  et_z {r.et_z:+.2f}  pred {r.pred}  8.3k {int(r.rule83)}  유격 {int(r.knock)}  내림 {int(r.body)}")
out.append("val lr 0.5–1.5 (규칙 없음): " + ", ".join(f"{r.file}({r.lr:+.2f})" for r in Dv.itertuples() if 0.5 <= r.lr <= 1.5 and not (r.rule83 or r.knock)))
Dvi = Dv.set_index("file")
out.append("관심 파일: " + "; ".join(f"{n} ae {Dvi.loc[n,'ae_z']:+.2f} et {Dvi.loc[n,'et_z']:+.2f} lr {Dvi.loc[n,'lr']:+.2f} pred {int(Dvi.loc[n,'pred'])}" for n in ("val_004", "val_031", "val_035", "val_016", "val_039", "val_022")))
out.append("val AE z 상위 8: " + ", ".join(f"{r.file}({r.ae_z:+.1f})" for r in Dv.sort_values("ae_z", ascending=False).head(8).itertuples()))
out.append("val ET z 상위 8: " + ", ".join(f"{r.file}({r.et_z:+.1f})" for r in Dv.sort_values("et_z", ascending=False).head(8).itertuples()))
open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8").write("\n".join(out))

# %% 그림 1: AE z x ET z 평면 (규칙으로 잡힌 train 고장은 뺀다, val은 모두 표시하고 규칙 켜짐은 X)
fig, ax = plt.subplots(figsize=(9, 8))
xs = np.linspace(-3, 12, 50)
sel = lambda k, r: (tbl.y == k) & (tbl.rule == r)
ax.scatter(tbl.ae_z[sel(0, False)], tbl.et_z[sel(0, False)], s=16, c=NOR, alpha=.5, label="train 정상 OOF")
ax.scatter(tbl.ae_z[sel(1, False)], tbl.et_z[sel(1, False)], s=40, c=ABN, label="train 고장 OOF (규칙 안 켜진 것)")
vv = ~rule_v
ax.scatter(Dv.ae_z[vv], Dv.et_z[vv], s=46, c=VAL, marker="^", label="val (규칙 안 켜짐)")
ax.scatter(Dv.ae_z[rule_v], Dv.et_z[rule_v], s=70, c=VAL, marker="X", label="val (규칙 켜짐)")
for r in Dv.itertuples():
    if r.lr > 0 or r.ae_z > 3 or r.et_z > 3: ax.annotate(r.file[-3:], (r.ae_z, r.et_z), fontsize=7, color=VAL, xytext=(3, 3), textcoords="offset points")
for lvl, ls_, lab_ in ((0.0, ":", "lr = 0"), (m.thr_, "-", f"lr = 임계값 {m.thr_:g}")):
    ax.plot(xs, (lvl - b - w[0] * xs) / w[1], color="k", ls=ls_, lw=1.3, label=lab_)
ax.axvline(1, color="gray", lw=.6, ls="--"); ax.axhline(1, color="gray", lw=.6, ls="--")
ax.set_xlim(-3, max(8, Dv.ae_z.max() + 1)); ax.set_ylim(-3, max(8, Dv.et_z.max() + 1))
ax.set_xlabel("AE z (스펙트럼 이상, 1 = 정상 90% 선)"); ax.set_ylabel("ET z (충격·피처 이상, 1 = 정상 90% 선)")
ax.set_title("최종 모델 z 평면: train OOF와 val (규칙으로 잡힌 train 고장은 제외)"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q3_plane.png"), dpi=90); plt.close(fig)

# %% 그림 2: AE z · ET z · lr 분포 띠
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 3, figsize=(16, 6))
for ax, (c, t) in zip(axs, (("ae_z", "AE z"), ("et_z", "ET z"), ("lr", "결합 점수 lr"))):
    nzv = tbl[c][tbl.y == 0]
    ax.axhspan(q(nzv, 5), q(nzv, 95), color=NOR, alpha=.15, label="정상 5-95%")
    ax.axhline(q(nzv, 99), color="k", ls="--", lw=1, label="정상 99%")
    if c == "lr": ax.axhline(m.thr_, color="r", lw=1, label=f"임계값 {m.thr_:g}")
    for k, v, col, mk in ((0, tbl[c][sel(0, False)].values, NOR, "o"), (1, tbl[c][sel(1, False)].values, ABN, "o"), (2, Dv[c].values, VAL, "^")):
        ax.scatter(k + rng.uniform(-.17, .17, len(v)), v, s=16 if k == 0 else 38, c=col, marker=mk, alpha=.55 if k == 0 else .95)
    for r, xj in zip(Dv.itertuples(), rng.uniform(-.17, .17, len(Dv))):
        if getattr(r, c) > q(nzv, 99): ax.annotate(r.file[-3:], (2 + xj, getattr(r, c)), fontsize=6.5, color=VAL, xytext=(2, 2), textcoords="offset points")
    ax.set_xticks([0, 1, 2], ["train 정상\nOOF", "train 고장 OOF\n(규칙 제외)", "val 40"]); ax.set_title(t); ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.suptitle("최종 모델 점수 분포 (train OOF vs val)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q3_strips.png"), dpi=90); plt.close(fig)

# %% 그림 3: val 40개 결합 점수 막대
D2 = Dv.sort_values("lr", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(13, 5))
cols = [("#d62728" if (r.rule83 or r.knock) else VAL) for r in D2.itertuples()]
ax.bar(range(len(D2)), D2.lr, color=cols)
ax.axhline(m.thr_, color="r", lw=1); ax.axhline(0, color="k", lw=.6)
ax.set_xticks(range(len(D2)), [f[-3:] for f in D2.file], rotation=90, fontsize=8)
ax.set_ylabel("결합 점수 lr (규칙 가산 제외)"); ax.set_title(f"val 40개 결합 점수 (빨강 = 규칙으로 고장, 빨간 선 = 임계값 {m.thr_:g})")
ax.set_ylim(top=min(D2.lr.max() + 1, 25)); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q3_val_bars.png"), dpi=90); plt.close(fig)
print("done")
