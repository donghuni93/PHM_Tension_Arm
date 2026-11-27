# %% [markdown]
# EXP-070 AE 제외 모델로 test 판정 (EXP-062 구성 3에서 AE만 뺌)
# - 가설: H1″ (EXP-069와 같음). 바꾼 것: FinalJ(fmax=11000, use_ae=False) 하나. 결합 = [ET logit, 박자 z], AE 학습 안 함. E2 · 8.3k 규칙 · 11 kHz 상한은 그대로
# - 학습 150개 전체로 한 번. 문턱은 학습 점수에서 정한다(test를 보고 고치지 않는다). 점수(F2)는 test 라벨이 없어 못 잰다 → EXP-062 판정과의 차이만 본다
# - 실행: conda run -n qe-phm python 6_TEST_단계/실험/EXP-070_AE제외_test/자료/run.py
# %%
import os, sys, json
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False   # 한글 · 마이너스 깨짐 방지
from src.data import load_dataset, load_folder
from src.final import FinalJ, write_submission

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
SUB62 = os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062")

# %% 학습 (AE 제외) · 예측
X, y, names = load_dataset(os.path.join(ROOT, "2_data", "TeamProject_train_변경"))
m = FinalJ(fmax=11000, use_ae=False).fit(X, y, names)
joblib.dump(m, os.path.join(HERE, "final_model_070_noAE.joblib"))
Xt, nt = load_folder(os.path.join(ROOT, "2_data", "TeamProject_test", "test"))
Xv, nv = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
Dt, Dv = m.predict_table(Xt, nt), m.predict_table(Xv, nv)
Dt.to_csv(os.path.join(HERE, "근거표_test_noAE.csv"), index=False, encoding="utf-8-sig")
write_submission(Dt, os.path.join(HERE, "submission_test_noAE.csv"), os.path.join(ROOT, "2_data", "TeamProject_test", "submission_test.csv"))

# %% EXP-062(AE 있음)와 비교
m62 = joblib.load(os.path.join(SUB62, "모델", "final_model_062.joblib"))
Dt62, Dv62 = m62.predict_table(Xt, nt), m62.predict_table(Xv, nv)
c = m.core_.c_.coef_[0]
tag = lambda D, p: sorted(str(f)[-3:] for f in D.file[D.pred == 1])
T0, T1, V0, V1 = tag(Dt62, 1), tag(Dt, 1), tag(Dv62, 1), tag(Dv, 1)
res = dict(coef_ET_beat_AE=[round(float(v), 3) for v in c], intercept=round(float(m.core_.c_.intercept_[0]), 3), threshold=round(float(m.core_.tc_), 3),
           test_with_AE=T0, test_no_AE=T1, test_added=sorted(set(T1) - set(T0)), test_dropped=sorted(set(T0) - set(T1)),
           val_with_AE=V0, val_no_AE=V1, val_added=sorted(set(V1) - set(V0)), val_dropped=sorted(set(V0) - set(V1)),
           rules_test=dict(spec83=int(Dt.spec83.sum()), res83=int(Dt.res83.sum()), E2=int(Dt.E2.sum())))
near = Dt[(Dt.margin.abs() < 0.5) & ~Dt.rule83 & ~Dt.E2].sort_values("margin")
res["test_near"] = [f"{str(r.file)[-3:]}({r.margin:+.2f})" for r in near.itertuples()]
json.dump(res, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# %% 파일별 표 (여유 비교)
M = pd.DataFrame({"file": Dt.file, "여유_AE있음": Dt62.margin.values, "여유_AE없음": Dt.margin.values,
                  "판정_AE있음": Dt62.pred.values, "판정_AE없음": Dt.pred.values, "AE_z": Dt62.ae_z.values}).sort_values("여유_AE없음", ascending=False)
M.to_csv(os.path.join(HERE, "test_여유비교.csv"), index=False, encoding="utf-8-sig")

# %% 그림: 여유 AE 있음 vs 없음 (test)
fig, ax = plt.subplots(figsize=(6.4, 6))
chg = M.판정_AE있음 != M.판정_AE없음
ax.axhline(0, c="gray", lw=.8); ax.axvline(0, c="gray", lw=.8)
ax.scatter(M.여유_AE있음[~chg], M.여유_AE없음[~chg], s=28, c="#7f8c8d", label="판정 같음")
ax.scatter(M.여유_AE있음[chg], M.여유_AE없음[chg], s=60, c="#e67e22", edgecolor="k", label="판정 바뀜")
for i, r in enumerate(M[chg].sort_values("여유_AE있음").itertuples()): ax.annotate("test_" + str(r.file)[-3:], (r.여유_AE있음, r.여유_AE없음), xytext=(-70 + 85 * i, 30 + 20 * i), textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="-", color="gray"))
lim = [min(M.여유_AE있음.min(), M.여유_AE없음.min()) - .5, max(M.여유_AE있음.max(), M.여유_AE없음.max()) + .5]
ax.plot(lim, lim, ls=":", c="gray"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("결합 여유 — AE 있음 (EXP-062)"); ax.set_ylabel("결합 여유 — AE 없음 (EXP-070)"); ax.set_title("test 40개: 문턱 대비 여유"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "여유_AE있음_없음.png"), dpi=130)
print(json.dumps(res, ensure_ascii=False, indent=1)); print(M.to_string())
