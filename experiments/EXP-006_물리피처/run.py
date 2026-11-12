# %% [markdown]
# EXP-006 물리 가설 피처 + 피처셋별 모델 결합
# - 가설: H2 공진 날카로움, H3 스펙트럼 모양(게인 무관), H4 하중 반전 충격
# - 바꾼 것: 확장 150(EXP-002)에 물리 피처(src/features.py phys)를 더하거나, 피처셋별 ET를 순위평균으로 결합
# - 이전: scripts/exp006.py (11-13). 구조 개편 때 이 폴더로 옮김

# %% 설정
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..")))
import warnings; warnings.filterwarnings("ignore")
from src.exp import *
from sklearn.metrics import roc_auc_score

ex = Experiment(__file__, "EXP-006", "물리 가설 피처와 피처셋별 모델 결합", ["H2", "H3", "H4"])
X, y, names = load_train()
E = cached_table("ext", feature_table_ext, X)
H = cached_table("phys", feature_table_phys, X)
g = np.arange(len(y))

# %% 그림: 놓친 고장 PSD vs 정상 5–95%
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_focus",
            "abn_002·022·029의 PSD가 정상 범위 밖으로 나가는 곳. 8.2–8.4 kHz 협대역 피크(H2), 저주파(H4)를 본다")
fig, auc_bin = viz.psd_auc(f, L, y)
ex.save_fig(fig, "psd_auc", "주파수 빈별 AUC. 0.5에서 멀수록 그 주파수의 파워가 정상·고장을 가른다")
ex.save_fig(viz.waves(X, names, ["nor_001", "nor_050"] + ex.focus), "wave_focus",
            "정상 2개와 놓친 고장 파형. 022는 저주파 흔들림 반전점의 충격(H4)을 본다")

# %% 물리 피처 단변량 점검: AUC와 놓친 고장의 정상 대비 백분위
nor = H[y == 0]
rows = []
for c in H.columns:
    a = roc_auc_score(y, H[c])
    pct = {n: (nor[c] < H[c].iloc[names.index(n)]).mean() * 100 for n in ex.focus}
    rows.append(dict(feat=c, auc=max(a, 1 - a), dir="+" if a >= .5 else "-", **pct))
A = pd.DataFrame(rows).sort_values("auc", ascending=False)
ex.note("물리 피처 AUC 상위 15 (abn 열 = 정상 대비 백분위)", A.head(15).round(3).to_string(index=False))
ex.save_fig(viz.feature_auc_bar(A), "phys_auc", "물리 피처(H1–H4) 단변량 AUC 상위")
h14 = [c for c in H.columns if c.startswith("H4")] + ["H1_impact_rate"]
ex.save_fig(viz.feature_percentile(H, y, names, ex.focus, h14), "h4_percentile",
            "H4 반전 충격 피처에서 놓친 고장이 정상 분포의 어디에 있나 (0·100이면 정상 밖)")

# %% 중첩 CV: 구성 비교
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
    ex.record(name, nested_cv(F_all, y, g, mk), names)
