"""EXP-006 물리 가설 피처(H2 공진 날카로움, H3 스펙트럼 모양, H4 반전 충격) + 피처셋별 모델 결합"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from common import *
from sklearn.metrics import roc_auc_score

X, y, names = load_train()
E = cached_table("ext", feature_table_ext, X)
H = cached_table("phys", feature_table_phys, X)
log = open(os.path.join(OUT, "exp006.txt"), "w", encoding="utf-8")

# 1) 물리 피처 단변량 점검: AUC와 놓친 고장의 정상 대비 백분위
nor = H[y == 0]
rows = []
for c in H.columns:
    a = roc_auc_score(y, H[c])
    pct = {n: (nor[c] < H[c].iloc[names.index(n)]).mean() * 100 for n in ["abn_002", "abn_022", "abn_029"]}
    rows.append(dict(feat=c, auc=max(a, 1 - a), dir="+" if a >= .5 else "-", **pct))
A = pd.DataFrame(rows).sort_values("auc", ascending=False)
log.write("== 물리 피처 AUC 상위 15 (002/022/029 = 정상 대비 백분위)\n" + A.head(15).round(3).to_string(index=False) + "\n")
log.write("== H4·H1 피처\n" + A[A.feat.str.match("H[41]")].round(3).to_string(index=False) + "\n\n")
log.flush()

g = np.arange(len(y))
cE = list(range(E.shape[1]))
F_all = np.hstack([E.values, H.values])
off = E.shape[1]
idx = {k: [off + i for i, c in enumerate(H.columns) if c.startswith(k)] for k in ["H2", "H3", "H4"]}
idx["H14"] = idx["H4"] + [off + list(H.columns).index("H1_impact_rate")]
et = make_et()
col = lambda cols: (lambda: ColModel(cols, et))

configs = [
    ("기준 ext150", col(cE)),
    ("ext150 + H2 공진날카로움", col(cE + idx["H2"])),
    ("ext150 + H4 반전충격(+H1 충격수)", col(cE + idx["H14"])),
    ("ext150 + H2 + H3 + H4 (한 모델)", col(list(range(F_all.shape[1])))),
    ("물리만 H2+H3+H4", col(idx["H2"] + idx["H3"] + idx["H14"])),
    ("결합: ET(ext150) + ET(H3 모양)", lambda: RankAvg([col(cE), col(idx["H3"])])),
    ("결합: ET(ext150) + ET(H2+H3)", lambda: RankAvg([col(cE), col(idx["H2"] + idx["H3"])])),
    ("결합: ET(ext150+H4) + ET(H2+H3)", lambda: RankAvg([col(cE + idx["H14"]), col(idx["H2"] + idx["H3"])])),
]
for name, mk in configs:
    report(name, nested_cv(F_all, y, g, mk), names, log)
log.close()
