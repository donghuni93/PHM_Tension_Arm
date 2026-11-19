# %% [markdown]
# 18. 규칙을 피처로: 8.3 kHz 규칙(P83 · W83)과 유격 규칙(K1 · K2 · K3)이 쓰던 연속 수치를 ET 입력에 넣어, 규칙 없이 모델 안에서 002 · 014 · 017류를 가르는지 본다
# 동기: 규칙은 문턱을 라벨 본 뒤 정한 사후 값이라 과적합 위험이 있다. 피처로 주면 문턱은 트리가 폴드 안에서 학습한다.
# 비교(nested_cv seed 3 × 바깥 5 × 안쪽 4): P1 AE+ET(D) · P3 P1 + 규칙 · Q0 AE+ET(D+) 규칙 없음 · Q1 AE+ET(D+) + 규칙.  D+ = D 76열 + [P83, W83, K1, K2, K3] 5열(분위수 변환)
# 실행: python "휴대폰 분석/18_규칙을_피처로/run.py"
import os, sys, warnings, time
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "17_AE_ET_D", "run.py"), encoding="utf-8").read().split("g = np.arange(len(y))")[0]
G = {"__file__": os.path.join(HERE, "..", "17_AE_ET_D", "run.py")}; exec(compile(src, "run17", "exec"), G)
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from src.evaluate import nested_cv, best_f2_threshold
from src.final import signal_tables, _fit_ae, _ae_score
AEET, D, y, names, ae_mean = G["AEET"], G["D"], G["y"], G["names"], G["ae_mean"]
g16 = G["g16"]; Dv, nv, Xtr, Xv = g16["Dv"], g16["nv"], g16["X"], g16["Xv"]
Tt, cols = signal_tables(Xtr); Tv, _ = signal_tables(Xv, cols)
def rf(T): return np.c_[T["P83"], T["W83"], T["K"][:, 0], T["K"][:, 1], T["K"][:, 2]]
R, Rv = rf(Tt), rf(Tv); q = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit(R)
Dp, Dpv = np.hstack([D, q.transform(R)]), np.hstack([Dv, q.transform(Rv)]); print("D+", Dp.shape, flush=True)
g = np.arange(len(y)); CFG = {"P1 AE+ET(D)": (np.c_[g, D], False), "P3 AE+ET(D)+규칙": (np.c_[g, D], True), "Q0 AE+ET(D+) 규칙 없음": (np.c_[g, Dp], False), "Q1 AE+ET(D+)+규칙": (np.c_[g, Dp], True)}
rows = []
for nm, (F_, rl) in CFG.items():
    t0 = time.time(); r = nested_cv(F_, y, g, lambda rl=rl: AEET(rules=rl), seeds=range(3), n_jobs=-1)
    print(f"{nm:22s} F₂ {r['f2'].mean():.3f} ± {r['f2'].std():.3f} | 재현율 {r['recall'].mean():.3f} | 오경보율 {r['fpr'].mean():.3f} | 자주 놓침 {[names[i][-3:] for i in np.where(r['miss_counts'] >= 3)[0]]} | 놓침 횟수(002·003·006·010·014·017·022·029) { {names[i][-3:]: int(r['miss_counts'][i]) for i in range(len(y)) if names[i][-3:] in ('002','003','006','010','014','017','022','029') and names[i].startswith('abn')} } | 오경보 3회+ {[names[i][-3:] for i in np.where(r['fp_counts'] >= 3)[0]]} | {time.time() - t0:.0f}s", flush=True)
    rows.append((nm, r["f2"].mean(), r["f2"].std(), r["recall"].mean(), r["precision"].mean(), r["fpr"].mean()))
pd.DataFrame(rows, columns=["구성", "F2", "sd", "재현율", "정밀도", "오경보율"]).round(3).to_csv(os.path.join(HERE, "nested_cv.csv"), index=False, encoding="utf-8-sig")
# %% val: 규칙 없이 Q0가 규칙 켜진 val(001 · 027 · 035 · 039 등)을 잡는가
Fall = np.c_[g, Dp]; sc = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Fall, y): sc[te] = AEET(rules=False).fit(Fall[tr], y[tr]).decision_function(Fall[te])
thr = best_f2_threshold(y, sc); p = sc >= thr
print(f"Q0 train OOF: 임계값 {thr:.2f} 재현율 {int((p & (y == 1)).sum())}/30 오경보 {int((p & (y == 0)).sum())}/120 | 놓친 {[names[i][-3:] for i in np.where(~p & (y == 1))[0]]} | 오경보 {[names[i][-3:] for i in np.where(p & (y == 0))[0]]}")
fin = AEET(rules=False).fit(Fall, y); ae = _fit_ae(Tt["LP"][y == 0]); Sv = np.c_[_ae_score(ae, Tv["LP"]), fin._et(Dp, y, Dpv)]; sv = fin.st_.decision_function(fin.sc_.transform(Sv)); pv = sv >= thr
t25 = pd.read_csv("/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad/val_근거표_025.csv").set_index("file")
f25 = set(t25.index[t25.pred == 1]); fq = set(np.array(nv)[pv]); rl = set(t25.index[(t25.rule83 | t25.knock)])
print("val Q0(규칙 없음) 고장 판정:", sorted(f[-3:] for f in fq)); print("025 모델:", sorted(f[-3:] for f in f25), "| 025 규칙 켜진:", sorted(f[-3:] for f in rl))
print("Q0가 025 규칙 파일을 규칙 없이 잡음:", sorted(f[-3:] for f in rl & fq), "| 놓침:", sorted(f[-3:] for f in rl - fq), "| Q0만:", sorted(f[-3:] for f in fq - f25))
pd.DataFrame({"file": nv, "Q0 점수": sv, "Q0 판정": pv.astype(int), "025 판정": [int(f in f25) for f in nv], "025 규칙": [int(f in rl) for f in nv]}).sort_values("Q0 점수", ascending=False).round(3).to_csv(os.path.join(HERE, "val_scores.csv"), index=False, encoding="utf-8-sig")
