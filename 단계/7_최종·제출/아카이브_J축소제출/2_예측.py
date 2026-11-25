# %% [markdown]
# 7단계 ② 새 파일 판정: 폴더(val 또는 test) → 제출 CSV + 근거표 (구성 J-축소, 11-22)
# - 학습은 하지 않는다. ①에서 저장한 모델(outputs/final/final_model.joblib)을 읽어 쓰기만 한다
# - 실행 예 (val 폴더가 data/TeamProject_val/val 일 때)
#     conda run -n qe-phm python "단계/7_최종·제출/아카이브_J축소제출/2_예측.py" data/TeamProject_val/val submission_val_9조.csv
#   제출 양식 CSV가 있으면 세 번째 인자로 준다 (열 이름·행 순서를 양식 그대로 맞춘다)
#     conda run -n qe-phm python "단계/7_최종·제출/아카이브_J축소제출/2_예측.py" data/TeamProject_val/val submission_val_9조.csv 양식.csv
# - 출력: 단계/7_최종·제출/출력/{제출 CSV, 근거표_*.csv, 요약_*.txt}
# %%
import os, sys
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import joblib
from src.data import load_folder
from src.final import write_submission
folder, out_name = sys.argv[1], sys.argv[2]
template = sys.argv[3] if len(sys.argv) > 3 else None
model_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(_d, "outputs", "final", "final_model.joblib")
OUT = os.path.join(HERE, "출력"); os.makedirs(OUT, exist_ok=True)
m = joblib.load(model_path)
X, names = load_folder(folder if os.path.isabs(folder) else os.path.join(_d, folder))
D = m.predict_table(X, names)
tag = os.path.splitext(os.path.basename(out_name))[0]
D.to_csv(os.path.join(OUT, f"근거표_{tag}.csv"), index=False, encoding="utf-8-sig")     # 엑셀에서 한글이 보이게 BOM. 양식 채우기가 실패해도 근거표는 남는다
write_submission(D, os.path.join(OUT, os.path.basename(out_name)), template)

# %% 요약: 제출 전에 사람이 볼 것
n1 = int(D.pred.sum()); comb_only = (D.pred == 1) & ~D.rule83 & ~D.E2
near = D[(D.margin.abs() < 0.5) & ~D.rule83 & ~D.E2].sort_values("margin")
S = [f"읽은 파일 {len(D)}개 ({D.file.iloc[0]} … {D.file.iloc[-1]}) · 구성 J-축소 · 결합 문턱 {m.core_.tc_:.3f}",
     f"고장 판정 {n1}개 ({n1 / len(D):.0%}) · 정상 판정 {len(D) - n1}개",
     f"  규칙으로 고장: 8.3 kHz 스펙트럼 {int(D.spec83.sum())} · 8.3 kHz 공진 {int(D.res83.sum())} · E2 {int(D.E2.sum())} (겹침 포함, 규칙 고장 합계 {int((D.rule83 | D.E2).sum())})",
     f"  결합 점수만으로 고장: {int(comb_only.sum())}개",
     f"  [관찰] 결합 여유 |점수 − 문턱| < 0.5 (규칙 안 켜짐, 판정이 흔들리기 쉬운 파일): "
     + (" ".join(f"{r.file}({r.margin:+.2f})" for r in near.itertuples()) or "없음"),
     "", "고장 판정 파일과 근거"]
S += [f"  {r.file}: {r.reason}" for r in D[D.pred == 1].itertuples()]
open(os.path.join(OUT, f"요약_{tag}.txt"), "w", encoding="utf-8").write("\n".join(S) + "\n")
