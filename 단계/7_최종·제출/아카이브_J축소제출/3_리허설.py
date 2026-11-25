# %% [markdown]
# 7단계 ③ 리허설 (구성 J-축소, 11-22): 학습 파일 40개를 '가짜 val'로 떼어 놓고, val 날에 할 일을 처음부터 끝까지 해 본다
# - 목적: 11/19에 처음 돌리다가 막히지 않게 한다 (폴더 읽기 → 판정 → 제출 CSV → 채점). 점수 자체는 중첩 CV의 한 조각과 같다
# - 방법: 고장 8 + 정상 32개(val과 같은 40개, 고장 20% 가정)를 뽑아 이름을 val_001… 로 바꿔 outputs/rehearsal/val/ 에 복사,
#         남은 110개로 학습 → 2_예측.py 와 같은 함수로 판정 → 숨겨 둔 정답으로 F2
#   뽑는 seed를 3개로 바꿔 3번 한다 (40개 표본 하나의 점수가 얼마나 흔들리는지도 같이 본다)
# 실행: conda run -n qe-phm python "단계/7_최종·제출/아카이브_J축소제출/3_리허설.py"

# %%
import os, sys, shutil
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from src.data import load_dataset, load_folder
from src.final import FinalJ, write_submission

DATA = os.path.join(_d, "data", "TeamProject_train_변경")
X, y, names = load_dataset(DATA)
lines = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    hold = np.concatenate([rng.choice(np.flatnonzero(y == 1), 8, replace=False), rng.choice(np.flatnonzero(y == 0), 32, replace=False)])
    hold = rng.permutation(hold)                               # val 번호가 라벨 순서를 드러내지 않게 섞는다
    keep = np.setdiff1d(np.arange(len(y)), hold)
    root = os.path.join(_d, "outputs", "rehearsal", f"seed{seed}")
    shutil.rmtree(root, ignore_errors=True); os.makedirs(os.path.join(root, "val"))
    key = []
    for k, i in enumerate(hold, 1):
        sub = "abnormal" if y[i] else "normal"
        shutil.copyfile(os.path.join(DATA, sub, names[i] + ".csv"), os.path.join(root, "val", f"val_{k:03d}.csv"))
        key.append((f"val_{k:03d}", names[i], int(y[i])))
    key = pd.DataFrame(key, columns=["file", "원래 이름", "정답"])
    key.to_csv(os.path.join(root, "정답.csv"), index=False, encoding="utf-8-sig")
    # 양식 흉내: 파일 이름에 .csv가 붙고 순서가 뒤집힌 양식이어도 맞게 채워지는지
    pd.DataFrame({"filename": [f + ".csv" for f in key.file[::-1]], "label": 0}).to_csv(os.path.join(root, "양식.csv"), index=False)

    m = FinalJ().fit(X[keep], y[keep], [names[i] for i in keep])
    Xv, nv = load_folder(os.path.join(root, "val"))
    D = m.predict_table(Xv, nv)
    sub = write_submission(D, os.path.join(root, "submission_val_리허설.csv"), os.path.join(root, "양식.csv"))
    D.merge(key, on="file").to_csv(os.path.join(root, "근거표.csv"), index=False, encoding="utf-8-sig")
    # 채점은 제출 파일을 다시 읽어서 한다 (양식 채우기까지 검증)
    got = pd.read_csv(os.path.join(root, "submission_val_리허설.csv"))
    got["file"] = got["filename"].str.replace(".csv", "", regex=False)
    J = got.merge(key, on="file")
    assert len(J) == 40 and (J.label.values == D.set_index("file").loc[J.file, "pred"].values).all()
    t, p = J["정답"].values, J["label"].values
    miss = J[(t == 1) & (p == 0)]["원래 이름"].tolist(); fa = J[(t == 0) & (p == 1)]["원래 이름"].tolist()
    lines.append(f"seed {seed}: F2 {fbeta_score(t, p, beta=2):.3f} · 고장 8개 중 {int(((t == 1) & (p == 1)).sum())}개 탐지 · 오경보 {len(fa)}/32 · 결합 문턱 {m.thr_:.2f} "
                 f"· 놓침 {miss or '없음'} · 오경보 {fa or '없음'} · 결합 여유 0.5 이내 {int((D.margin.abs() < 0.5).sum())}개")
open(os.path.join(HERE, "결과", "리허설_결과.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
