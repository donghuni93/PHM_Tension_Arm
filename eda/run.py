# %% [markdown]
# 데이터 이해용 그림 모음 (실험과 무관한 데이터 자체의 그림). 이전 scripts/check_report.py
# 사용: VS Code 셀 실행, 또는  conda run -n qe-phm python eda/run.py abn_002 abn_022 abn_029
# 출력: eda/figs/*.png, eda/feature_auc.csv, eda/README.md (그림 + 무엇을 보는지)
# 새 실험에서 놓친 파일이 바뀌면 인자로 넣어 다시 돌린다.

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.path.abspath("eda")
sys.path.insert(0, os.path.dirname(HERE))
import warnings; warnings.filterwarnings("ignore")
from src.exp import *
from sklearn.metrics import roc_auc_score

ex = Experiment(os.path.join(HERE, "run.py"), "EDA", "데이터 이해 그림", ["H1", "H2", "H3", "H4"])
args = [a for a in sys.argv[1:] if a.startswith(("abn_", "nor_"))]
if args:
    ex.focus = args
X, y, names = load_train()
f, L = viz.log_psd(X)

# %% 1. 정상·고장 스펙트럼 비교 (H1·H2·H3)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_focus",
            "파란 띠 = 정상 120개 중 5–95%. 빨강 = 고장 중앙값. 지정 파일이 띠 밖으로 나가는 주파수가 그 파일을 잡을 단서")
ex.save_fig(viz.psd_band(f, L, y, names, [], ranges=((0, 12500),), title="정상 vs 고장 중앙값 (전체 대역)"),
            "psd_median", "고장이 평균적으로 어느 대역에서 정상과 다른가. 1–3 kHz(H1 충격 모드), 8.3 kHz(H2 공진)")

# %% 2. 주파수별 구분력
fig, auc_bin = viz.psd_auc(f, L, y)
ex.save_fig(fig, "psd_auc", f"주파수 빈별 AUC. 최고 {f[np.argmax(auc_bin)]:.0f} Hz ({auc_bin.max():.3f}), "
            f"최저 {f[np.argmin(auc_bin)]:.0f} Hz ({auc_bin.min():.3f}). 라벨을 본 그림이므로 대역 선택에 직접 쓰지 않는다")

# %% 3. 파형 (H1 충격, H4 저주파 흔들림)
show = ["nor_001", "nor_050"] + ex.focus
ex.save_fig(viz.waves(X, names, show), "wave_focus", "2초 전체 파형. 고장은 빨강. 짧은 스파이크(H1)와 느린 흔들림(H4)을 본다")
ex.save_fig(viz.waves(X, names, show, t_range=(0.5, 0.6)), "wave_zoom", "0.5–0.6초 확대. 충격 1개의 모양과 링다운(H5)")

# %% 4. 피처 단변량 AUC (부트스트랩 하위 5%)
F = pd.concat([cached_table("ext", feature_table_ext, X), cached_table("phys", feature_table_phys, X)], axis=1)
rng = np.random.default_rng(0)
boots = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(200)) if 0 < y[i].sum() < len(i)]
rows = []
for c in F.columns:
    v = F[c].values; a = roc_auc_score(y, v); s = 1 if a >= .5 else -1
    bs = [roc_auc_score(y[i], s * v[i]) for i in boots]
    rows.append(dict(feat=c, group=c.split("_")[0], auc=max(a, 1 - a), auc_p5=np.percentile(bs, 5), dir="+" if s > 0 else "-"))
A = pd.DataFrame(rows).sort_values("auc", ascending=False)
A.to_csv(os.path.join(HERE, "feature_auc.csv"), index=False, encoding="utf-8-sig")
ex.save_fig(viz.feature_auc_bar(A, 30), "feature_auc", "피처 단변량 AUC 상위 30 (전체 목록은 feature_auc.csv)")
ex.note("피처 묶음별 AUC (max·median·개수)",
        A.groupby("group").auc.agg(["max", "median", "count"]).sort_values("max", ascending=False).round(3).to_string())

# %% 5. 지정 파일이 정상 대비 어디에 있나
top = list(A.feat.head(30))
ex.save_fig(viz.feature_percentile(F, y, names, ex.focus, top), "focus_percentile",
            "AUC 상위 30 피처에서 지정 파일의 정상 대비 백분위. 전부 중간(흰색)이면 현재 피처로는 못 잡는 파일")
print("saved", ex.dir)
