# %% 4-19 대역별 최대 충격 배수 EDA (피처 후보 P0)
# 검출기 = 최종 모델(EXP-062 구성 3)과 같은 설정: src.impacts.all_impacts(fmax=11000), 검출 하나씩, 250 Hz 사전
# 열 = 그 대역으로 운 충격 중 최대 '포락선 ÷ 바탕' 배수, log10(1 + x) (충격 없으면 0). 대역은 옥타브(2–4 kHz) · 8–11 kHz로 실행 전 고정
import glob, os, sys
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.impacts import all_impacts
from src.final import j_tables, J_FIXED
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True); FS = 25000
fl, grp = [], []
for g, p in (("normal", "TeamProject_train_변경/normal"), ("abnormal", "TeamProject_train_변경/abnormal"), ("val", "TeamProject_val/val")):
    s = sorted(glob.glob(os.path.join(ROOT, "data", p, "*.csv"))); fl += s; grp += [g] * len(s)
names = np.array([os.path.basename(f)[:-4] for f in fl]); grp = np.array(grp)
X = np.stack([np.loadtxt(f) for f in fl])
N, A, V = grp == "normal", grp == "abnormal", grp == "val"
imps = all_impacts(X, fmax=11000)
BANDS = {"max_all": (0, 99999), "max_2_4k": (2000, 4000), "max_8_11k": (8000, 11000), "(참고) max_1.6_4.5k": (1600, 4500), "(참고) max_body_5.5_8k": (5500, 8000)}
F = pd.DataFrame(index=names)
for c, (a, b) in BANDS.items():
    v = []
    for im in imps:
        f = im["rings"][:, 0] if len(im["t"]) else np.array([]); hx = np.asarray(im["height_x"])
        m = (f >= a) & (f < b); v.append(np.log10(1 + hx[m].max()) if m.any() else 0.0)
    F[c] = v
F["n_imp"] = [len(im["t"]) for im in imps]
# J 입력(최종 모델과 같은 fmax=11000)과의 중복
T = j_tables(X, fmax=11000); H = pd.DataFrame(T["H"], columns=J_FIXED, index=names)
load = sosfiltfilt(butter(4, 20, "lowpass", fs=FS, output="sos"), X - X.mean(1, keepdims=True), axis=1).std(1)
rng = np.random.default_rng(0)
auc = lambda a, b: mannwhitneyu(a, b).statistic / (len(a) * len(b))
def ci(a, b, B=2000):
    s = [auc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(B)]; return np.percentile(s, [2.5, 97.5])
def mix_range(c, frac, B=1000):
    """학습 정상 32개 + 고장 섞음(frac)으로 val 크기(40)를 재추출했을 때 '정상 vs 재추출' AUC의 90 % 범위"""
    n, a = F[c][N].values, F[c][A].values; k = int(round(40 * frac)); out = []
    for _ in range(B):
        s = np.r_[rng.choice(n, 40 - k), rng.choice(a, k)] if k else rng.choice(n, 40); out.append(auc(s, n))
    return np.percentile(out, [5, 95])
L = ["열 | AUC(고장>정상) [95 %] | 고장 중 정상 99 % 초과 | 정상 중 고장 10 % 선 초과 | val 이동 AUC (고장 0 % · 20 % · 40 % 섞은 기준 범위) | ρ(하중) | 가장 겹치는 J 열"]
cols = list(BANDS)
for c in cols:
    v = F[c].values; n, a = v[N], v[A]; lo, hi = ci(a, n); q99 = np.percentile(n, 99); q10a = np.percentile(a, 10)
    rv = mannwhitneyu(v[V], n).statistic / (V.sum() * N.sum())
    r = {j: spearmanr(v[N | A], H[j].values[N | A])[0] for j in J_FIXED}; jb = max(r, key=lambda k: abs(r[k]))
    m0, m20, m40 = mix_range(c, 0), mix_range(c, .2), mix_range(c, .4)
    L.append(f"{c:22s} | {auc(a, n):.2f} [{lo:.2f}, {hi:.2f}] | {', '.join(names[A & (v > q99)]) or '-'} | {int((N & (v > q10a)).sum())}/120 | {rv:.2f} (0 %: {m0[0]:.2f}–{m0[1]:.2f}, 20 %: {m20[0]:.2f}–{m20[1]:.2f}, 40 %: {m40[0]:.2f}–{m40[1]:.2f}) | {spearmanr(v[N], load[N])[0]:+.2f} | {jb} {r[jb]:+.2f}")
L.append("\n후보끼리 ρ (Train 150): " + ", ".join(f"{a}~{b} {spearmanr(F[a][N|A], F[b][N|A])[0]:+.2f}" for i, a in enumerate(cols[:3]) for b in cols[i+1:3]))
L.append("충격이 없는 칸(값 0) 비율: " + " | ".join(f"{c}: 정상 {np.mean(F[c][N]==0)*100:.0f} % · 고장 {np.mean(F[c][A]==0)*100:.0f} % · val {np.mean(F[c][V]==0)*100:.0f} %" for c in cols[:3]))
pct = lambda c, f: (F[c].values[N] < F.loc[f, c]).mean() * 100
L.append("\n파일별 (원래 배수, 정상 내 백분위)")
for f in ["abn_021", "abn_028", "abn_014", "abn_022", "abn_009", "abn_016", "abn_002", "abn_017", "nor_059", "nor_065", "nor_116", "nor_025", "nor_032", "nor_046", "nor_054", "nor_060", "nor_095", "nor_110"]:
    L.append(f"{f}: " + " | ".join(f"{c} {10**F.loc[f, c]-1:6.1f}({pct(c, f):3.0f})" for c in cols[:3]) + f" | 충격 {F.loc[f,'n_imp']}")
for c in cols[:3]:
    q = np.percentile(F[c][N], 99)
    L.append(f"val 중 {c} > 정상 99 %({10**q-1:.1f}배): " + (", ".join(names[V & (F[c].values > q)]) or "없음"))
open(os.path.join(FIG, "수치.txt"), "w", encoding="utf-8").write("\n".join(L))
F.assign(group=grp).to_csv(os.path.join(FIG, "files.csv"))

# %% 그림: 세 열의 분포 (정상 · 고장 · val), 강타형 정상 · 대상 고장 표시
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"; C = {"normal": "#2a78d6", "abnormal": "#eb6834", "val": "#1baf7a"}
LAB = {"normal": "정상 120", "abnormal": "고장 30", "val": "val 40"}
fig, ax = plt.subplots(1, 3, figsize=(15, 5), facecolor=SURF)
for a_, c, ttl in zip(ax, cols[:3], ["파일 전체 최대 충격", "2–4 kHz로 운 충격 중 최대", "8–11 kHz로 운 충격 중 최대"]):
    a_.set_facecolor(SURF); a_.grid(True, axis="y", color=GRID); a_.set_axisbelow(True)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
    for k, g in enumerate(("normal", "abnormal", "val")):
        v = 10 ** F[c][grp == g].values - 1 + 1
        a_.scatter(k + rng.uniform(-.25, .25, len(v)), v, s=14, color=C[g], alpha=.6, lw=0)
        a_.hlines(np.median(v), k - .32, k + .32, color=INK, lw=2)
    for f, lab in (("abn_021", "021"), ("abn_028", "028")):
        a_.annotate(lab, (1, 10 ** F.loc[f, c]), xytext=(1.35, 10 ** F.loc[f, c]), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2))
    a_.axhline(10 ** np.percentile(F[c][N], 99), color=C["normal"], ls="--", lw=1)
    a_.set_yscale("log"); a_.set_xticks(range(3)); a_.set_xticklabels([LAB[g] for g in ("normal", "abnormal", "val")])
    a_.set_title(ttl, loc="left", color=INK, fontsize=11); a_.set_ylabel("1 + 최대 배수(포락선 ÷ 바탕)", color=INK2)
fig.text(0.01, 0.01, "검은 선 = 중앙값, 파란 점선 = 정상 99 %. 충격이 없으면 1(값 0)에 놓인다. 검출기 = 최종 모델(1–11 kHz, 250 Hz 사전)", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(FIG, "최대충격배수.png"), dpi=140, facecolor=SURF); print("done")
