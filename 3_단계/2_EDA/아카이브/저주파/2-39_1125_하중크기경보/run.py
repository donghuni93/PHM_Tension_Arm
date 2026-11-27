# %% 2-39 하중 크기(0–20 Hz 에너지) 위 꼬리 경보 EDA — abn_028을 저주파로 잡을 수 있나
# 규칙 후보: 0–20 Hz 에너지 > 학습 정상의 고정 분위수(폴드 안) → 고장(OR). 문턱은 정상 분포로만, 라벨로 고르지 않는다.
# 축은 abn_028을 보고 고른 것이라 [사후 축]. 그룹 CV(097·101·120 한 그룹), seed 0–4 = 최종 모델 평가와 같은 분할 방식
import glob, os
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "2_data")): ROOT = os.path.dirname(ROOT)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True); FS = 25000
fl, grp = [], []
for g, p in (("normal", "TeamProject_train_변경/normal"), ("abnormal", "TeamProject_train_변경/abnormal"), ("val", "TeamProject_val/val")):
    s = sorted(glob.glob(os.path.join(ROOT, "2_data", p, "*.csv"))); fl += s; grp += [g] * len(s)
names = np.array([os.path.basename(f)[:-4] for f in fl]); grp = np.array(grp)
X = np.stack([np.loadtxt(f) for f in fl]); Xc = X - X.mean(1, keepdims=True)
N, A, V, T = grp == "normal", grp == "abnormal", grp == "val", grp != "val"
sos = lambda kind, f: butter(4, f, kind, fs=FS, output="sos")
lo20 = sosfiltfilt(sos("lowpass", 20), Xc, axis=1)
cut = slice(2500, -2500)                                   # 양 끝 0.1 s(필터 가장자리) 제외 판
F = pd.DataFrame(index=names)
F["e020"] = np.log10(lo20.var(1))                         # 지금 정의(뷰어와 같은 0–20 Hz 에너지)
F["e020_cut"] = np.log10(lo20[:, cut].var(1))
F["e2_20"] = np.log10(sosfiltfilt(sos("bandpass", [2, 20]), Xc, axis=1)[:, cut].var(1))
h = X.shape[1] // 2
F["e020_h1"] = np.log10(lo20[:, 2500:h].var(1)); F["e020_h2"] = np.log10(lo20[:, h:-2500].var(1))
F["crest020"] = np.log10(np.abs(lo20[:, cut]).max(1) / lo20[:, cut].std(1))
auc = lambda a, b: mannwhitneyu(a, b).statistic / (len(a) * len(b))
L = []
L.append("① 분포 (log10 g²). AUC(고장>정상) · 정상 99 % / 최대 · abn_028 값(백분위) · 정상 최대를 넘는 고장 / val")
for c in ("e020", "e020_cut", "e2_20"):
    v = F[c].values; n = v[N]
    L.append(f"{c:9s} AUC {auc(v[A], n):.2f} | 정상 99 % {np.percentile(n,99):.2f} · 최대 {n.max():.2f} | 028 {F.loc['abn_028',c]:.2f} ({(n < F.loc['abn_028',c]).mean()*100:.0f}) | 초과 고장 {list(names[A & (v > n.max())])} · val {list(names[V & (v > n.max())])}")
    L.append(f"          정상 상위 4: {pd.Series(n, index=names[N]).sort_values(ascending=False).head(4).round(2).to_dict()} | 고장 상위 6: {pd.Series(v[A], index=names[A]).sort_values(ascending=False).head(6).round(2).to_dict()}")
L.append(f"반쪽 안정성 ρ(앞 0.9 s, 뒤 0.9 s) 정상 {spearmanr(F.e020_h1[N], F.e020_h2[N])[0]:+.2f} | 028 앞/뒤 {F.loc['abn_028','e020_h1']:.2f}/{F.loc['abn_028','e020_h2']:.2f} (정상 최대 앞/뒤 {F.e020_h1[N].max():.2f}/{F.e020_h2[N].max():.2f})")
L.append(f"하중 파고율(최대/RMS, log): 028 {F.loc['abn_028','crest020']:.2f} (정상 {(F.crest020[N] < F.loc['abn_028','crest020']).mean()*100:.0f} 백분위) — 한두 번 크게 튄 것인지, 계속 큰 것인지")
L.append(f"val 이동 AUC(val>정상) {auc(F.e020[V], F.e020[N]):.2f}")

# %% ② 폴드 안 규칙 모의: 학습 쪽 정상으로 문턱 → 시험 쪽 파일 판정 (최종 모델과 같은 그룹 분할 방식, seed 0–4)
tr_names = names[T]; y = A[T].astype(int); e = F.e020.values[T]
gid = np.array([0 if n in ("nor_097", "nor_101", "nor_120") else i + 1 for i, n in enumerate(tr_names)])
L.append("\n② 폴드 안 규칙: 문턱 = 학습 폴드 정상의 분위수 (5 seed × 5 폴드). 파일별 '켜진 횟수/5'")
for q in (99.0, 99.5, 100.0):
    cnt = np.zeros(len(tr_names))
    for s in range(5):
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(e, y, gid):
            t = np.percentile(e[a][y[a] == 0], q); cnt[b] += e[b] > t
    fa = {tr_names[i]: int(cnt[i]) for i in np.where((y == 0) & (cnt > 0))[0]}; hit = {tr_names[i]: int(cnt[i]) for i in np.where((y == 1) & (cnt > 0))[0]}
    L.append(f"분위 {q}: 고장 켜짐 {hit} | 정상 오경보 {fa} (seed당 평균 {sum(fa.values())/5:.1f}개)")
# 여유: 문턱(정상 최대)을 RMS ±20 %로 흔들 때
n = F.e020[N].values; t0 = n.max()
for k in (0.8, 0.9, 1.1, 1.2):
    t = t0 + 2 * np.log10(k)
    L.append(f"문턱 = 정상 최대 × RMS {k:.1f}: 고장 {list(names[A & (F.e020.values > t)])} | 정상 {list(names[N & (F.e020.values > t)])} | val {list(names[V & (F.e020.values > t)])}")
open(os.path.join(FIG, "수치.txt"), "w", encoding="utf-8").write("\n".join(L)); F.assign(group=grp).to_csv(os.path.join(FIG, "files.csv"))

# %% 그림: (a) 분포 (b) 028 · 하중이 가장 큰 정상 2개의 0–20 Hz 파형
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"; C = {"normal": "#2a78d6", "abnormal": "#eb6834", "val": "#1baf7a"}
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1.6]))
rng = np.random.default_rng(0)
for a_ in ax:
    a_.set_facecolor(SURF); a_.grid(True, color=GRID); a_.set_axisbelow(True)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
for k, g in enumerate(("normal", "abnormal", "val")):
    v = F.e020[grp == g].values; ax[0].scatter(k + rng.uniform(-.25, .25, len(v)), v, s=14, color=C[g], alpha=.6, lw=0)
for f in ("abn_028", "abn_010", "abn_015", "abn_016", "val_027"):
    k = {"abn": 1, "val": 2}[f[:3]]; ax[0].annotate(f[-3:], (k, F.loc[f, "e020"]), xytext=(k + .3, F.loc[f, "e020"]), fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2))
ax[0].axhline(n.max(), color=C["normal"], ls="--", lw=1); ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["정상 120", "고장 30", "val 40"])
ax[0].set_ylabel("0–20 Hz 에너지 (log10 g²)", color=INK2); ax[0].set_title("(a) 하중 크기. 점선 = 정상 최대", loc="left", color=INK)
tt = np.arange(X.shape[1]) / FS
top2 = pd.Series(n, index=names[N]).sort_values(ascending=False).index[:2]
for f, col in (("abn_028", C["abnormal"]), (top2[0], C["normal"]), (top2[1], "#7fb2ea")):
    ax[1].plot(tt, lo20[list(names).index(f)], color=col, lw=1.2, label=f)
ax[1].set_xlabel("시간 (s)", color=INK2); ax[1].set_ylabel("0–20 Hz 성분 (g)", color=INK2); ax[1].legend(frameon=False)
ax[1].set_title("(b) abn_028 vs 하중이 가장 큰 정상 2개", loc="left", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "하중크기.png"), dpi=140, facecolor=SURF); print("done")
