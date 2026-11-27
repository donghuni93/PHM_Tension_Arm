# %% [markdown]
# 최종 모델(EXP-062 구성 3) 판정: 폴더(val 또는 test) → 제출 CSV + 근거표 + 요약
# - 학습하지 않는다. 1_학습.py가 저장한 모델(이 폴더의 모델/final_model_062.joblib)만 읽는다
# - 인자 없이 실행하면 val: 2_data/TeamProject_val/val → 출력/submission_val_9조.csv (양식 2_data/TeamProject_val/submission_val.csv가 있으면 그 양식대로)
#     conda run -n qe-phm python "3_단계/7_최종·제출/제출_EXP-062/2_예측.py"
# - test: 폴더 · 파일 이름 · 양식을 준다 (경로는 프로젝트 루트 기준 상대 경로 또는 절대 경로)
#     conda run -n qe-phm python "3_단계/7_최종·제출/제출_EXP-062/2_예측.py" 2_data/TeamProject_test/test submission_test_9조.csv 2_data/TeamProject_test/submission_test.csv
# - 출력: 이 폴더의 출력/{제출 CSV, 근거표_*.csv, 요약_*.txt}. 제출은 사용자가 직접 한다(파일 확정은 승인)
# %%
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import joblib
from src.data import load_folder
from src.final import write_submission

rel = lambda p: p if p is None or os.path.isabs(p) else os.path.join(ROOT, p)
folder = rel(sys.argv[1] if len(sys.argv) > 1 else os.path.join("2_data", "TeamProject_val", "val"))
out_name = sys.argv[2] if len(sys.argv) > 2 else "submission_val_9조.csv"
template = rel(sys.argv[3]) if len(sys.argv) > 3 else rel(os.path.join("2_data", "TeamProject_val", "submission_val.csv"))
if template and not os.path.exists(template):
    template = None
model_path = os.path.join(HERE, "모델", "final_model_062.joblib")
if not os.path.exists(model_path):
    raise FileNotFoundError("모델이 없다. 먼저 1_학습.py를 실행한다: " + model_path)
OUT = os.path.join(HERE, "출력"); os.makedirs(OUT, exist_ok=True)

m = joblib.load(model_path)
X, names = load_folder(folder)
D = m.predict_table(X, names)
tag = os.path.splitext(os.path.basename(out_name))[0]
D.to_csv(os.path.join(OUT, f"근거표_{tag}.csv"), index=False, encoding="utf-8-sig")
write_submission(D, os.path.join(OUT, os.path.basename(out_name)), template)

# %% 요약 + (val일 때) 실험 결과와 같은지 확인
n1 = int(D.pred.sum()); comb_only = (D.pred == 1) & ~D.rule83 & ~D.E2
near = D[(D.margin.abs() < 0.5) & ~D.rule83 & ~D.E2].sort_values("margin")
S = [f"모델 EXP-062 구성 3 (옵션 fmax={getattr(m, 'fmax', None)}) · 읽은 파일 {len(D)}개 ({D.file.iloc[0]} … {D.file.iloc[-1]}) · 결합 문턱 {m.core_.tc_:.3f}",
     f"양식: {template or '없음(file,label 두 열)'}",
     f"고장 판정 {n1}개 ({n1 / len(D):.0%}) · 정상 판정 {len(D) - n1}개",
     f"  규칙으로 고장: 8.3 kHz 스펙트럼 {int(D.spec83.sum())} · 공진 {int(D.res83.sum())} · E2 {int(D.E2.sum())} · 결합 점수만으로 고장 {int(comb_only.sum())}",
     "  [관찰] 결합 여유 |점수 − 문턱| < 0.5 (규칙 안 켜짐): " + (" ".join(f"{r.file}({r.margin:+.2f})" for r in near.itertuples()) or "없음")]
EXPECT_VAL = {"004", "013", "016", "017", "024", "025", "027", "031", "035", "039"}   # EXP-062 구성 3 val_판정.md (학습 150 전체 모델)
if all(str(f).startswith("val_") for f in D.file):
    got = {str(f)[4:7] for f in D.file[D.pred == 1]}
    S.append(f"[확인] EXP-062 구성 3 실험의 val 판정과 같은가: {'같음' if got == EXPECT_VAL else '다름 → 차이 ' + str(sorted(got ^ EXPECT_VAL))}")
S += ["", "고장 판정 파일과 근거"] + [f"  {r.file}: {r.reason}" for r in D[D.pred == 1].itertuples()]
open(os.path.join(OUT, f"요약_{tag}.txt"), "w", encoding="utf-8").write("\n".join(S) + "\n")
print("done")
