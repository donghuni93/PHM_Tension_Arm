"""데이터·피처 점검 보고서. 실험 후 놓친 파일을 넣어 다시 돌린다.

사용: python check_report.py abn_002 abn_022 abn_029
출력: outputs/check/
  - psd.png        정상 5–95% 범위 + 고장 중앙값 + 지정 파일 (전체 / 0–300 Hz / 5–12.5 kHz)
  - psd_auc.png    PSD 빈별 AUC (어느 주파수에서 정상·고장이 갈리나)
  - wave.png       지정 파일 파형 (정상 2개와 비교)
  - feature_auc.csv  모든 피처의 단변량 AUC + 부트스트랩 하위 5%
  - report.md      피처 묶음별 AUC, 지정 파일의 정상 대비 가장 극단적인 피처
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
from common import OUT, load_train, cached_table
from src.features import feature_table_ext, feature_table_phys

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
D = os.path.join(OUT, "check"); os.makedirs(D, exist_ok=True)
X, y, names = load_train()
focus = sys.argv[1:] or ["abn_002", "abn_022", "abn_029"]

# --- PSD 그림 ---
Xc = X - X.mean(1, keepdims=True)
f, P = welch(Xc, fs=25000, nperseg=8192); L = np.log10(P)
nor, ab = L[y == 0], L[y == 1]
fig, axs = plt.subplots(3, 1, figsize=(13, 13))
for a, (lo, hi) in zip(axs, [(0, 12500), (0, 300), (5000, 12500)]):
    m = (f >= lo) & (f <= hi)
    a.fill_between(f[m], np.percentile(nor, 5, 0)[m], np.percentile(nor, 95, 0)[m], color="C0", alpha=.25, label="정상 5–95%")
    a.plot(f[m], np.median(nor, 0)[m], "C0", lw=1.5, label="정상 중앙값")
    a.plot(f[m], np.median(ab, 0)[m], "C3", lw=1.5, label="고장 중앙값")
    for i, n in enumerate(focus):
        a.plot(f[m], L[names.index(n)][m], lw=.8, color=f"C{i + 4}", label=n)
    a.set_xlim(lo, hi); a.set_xlabel("Hz"); a.set_ylabel("log10 PSD"); a.legend(fontsize=8, ncol=4); a.grid(alpha=.3)
axs[0].set_title("Welch PSD (df 3 Hz): 정상 범위 vs 지정 파일")
plt.tight_layout(); plt.savefig(os.path.join(D, "psd.png"), dpi=80); plt.close()

auc_bin = np.array([roc_auc_score(y, L[:, k]) for k in range(L.shape[1])])
fig, a = plt.subplots(figsize=(13, 3.5)); a.plot(f, auc_bin, lw=.8); a.axhline(.5, color="k", lw=.5)
a.set_title("PSD 빈별 AUC (>0.5 고장에서 큼, <0.5 고장에서 작음)"); a.set_xlabel("Hz"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "psd_auc.png"), dpi=80); plt.close()

show = ["nor_001", "nor_050"] + focus
fig, axs = plt.subplots(len(show), 1, figsize=(13, 2 * len(show)), sharex=True); t = np.arange(X.shape[1]) / 25000
for a, n in zip(axs, show):
    a.plot(t, Xc[names.index(n)], lw=.4); a.set_title(n, fontsize=9); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "wave.png"), dpi=70); plt.close()

# --- 피처 AUC ---
F = pd.concat([cached_table("ext", feature_table_ext, X), cached_table("phys", feature_table_phys, X)], axis=1)
rng = np.random.default_rng(0); rows = []
boots = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(200)) if 0 < y[i].sum() < len(i)]
for c in F.columns:
    v = F[c].values; a = roc_auc_score(y, v); s = 1 if a >= .5 else -1
    bs = [roc_auc_score(y[i], s * v[i]) for i in boots]
    rows.append(dict(feat=c, group=c.split("_")[0], auc=max(a, 1 - a), auc_p5=np.percentile(bs, 5), dir="+" if s > 0 else "-"))
A = pd.DataFrame(rows).sort_values("auc", ascending=False)
A.to_csv(os.path.join(D, "feature_auc.csv"), index=False, encoding="utf-8-sig")

def md(df, index=True):
    """tabulate 없이 마크다운 표"""
    df = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, df.columns)) + " |
|" + "---|" * len(df.columns) + "
"
    return head + "".join("| " + " | ".join(map(str, r)) + " |
" for r in df.itertuples(index=False))


nor = F[y == 0]
with open(os.path.join(D, "report.md"), "w", encoding="utf-8") as fh:
    fh.write(f"# 점검 보고서\n\n지정 파일: {', '.join(focus)}\n\n")
    fh.write(f"PSD 빈별 AUC 최고 {f[np.argmax(auc_bin)]:.0f} Hz ({auc_bin.max():.3f}), 최저 {f[np.argmin(auc_bin)]:.0f} Hz ({auc_bin.min():.3f})\n\n")
    fh.write("## 피처 묶음별 AUC\n\n" + A.groupby("group").auc.agg(["max", "median", "count"]).sort_values("max", ascending=False).round(3).pipe(md) + "\n\n")
    fh.write("## 단변량 AUC 상위 20\n\n" + md(A.head(20).round(3), index=False) + "\n\n")
    fh.write("## 지정 파일의 정상 대비 극단 피처 (백분위)\n\n| 파일 | 0·100 백분위 피처 수 | 가장 극단적인 5개 |\n|---|---|---|\n")
    for n in focus:
        v = F.iloc[names.index(n)]
        pct = pd.Series({c: (nor[c] < v[c]).mean() * 100 for c in F.columns})
        top = (pct - 50).abs().sort_values(ascending=False).index[:5]
        fh.write(f"| {n} | {int(((pct == 0) | (pct == 100)).sum())} | {', '.join(f'{c}={pct[c]:.0f}' for c in top)} |\n")
print("saved", D)
