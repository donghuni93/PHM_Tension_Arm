# %% 2-5 계산 ① 하단 대역 상한 4.5 vs 5.5 kHz  ② 오경보 정상 vs 고장의 크기 무관 이벤트 지표
# 입력: 2-2의 events.csv (이벤트 = 25 ms 묶음, 대표 울림 = 50 Hz 격자). 새 학습 · 모델 없음. 라벨을 본 탐색이다.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.path.dirname(ROOT)
EV = os.path.join(ROOT, "단계", "아카이브", "new_단계", "2_EDA", "2-2_충격이벤트", "figs")
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
E = pd.read_csv(os.path.join(EV, "events.csv")); F = pd.read_csv(os.path.join(EV, "files.csv"))
files = F[["file", "group"]].copy()
E["baseline_g"] = E.h / E.x                       # 이벤트 순간의 바탕 포락선(g). 진단용, 피처로 쓰지 않음
HIGH = (8000, 11000)
rng = np.random.default_rng(0)
L = []

def counts(U):
    d = files.copy()
    low = E[(E.f >= 1600) & (E.f < U)].groupby("file").size(); high = E[(E.f >= HIGH[0]) & (E.f < HIGH[1])].groupby("file").size()
    d["low"] = d.file.map(low).fillna(0).astype(int); d["high"] = d.file.map(high).fillna(0).astype(int); d["off"] = d.low + d.high
    return d
f2 = lambda tp, fn, fp: 5 * tp / (5 * tp + 4 * fn + fp)

# %% 계산 ①
L.append("계산 ① 하단 비정상 대역 상한 4.5 vs 5.5 kHz (off = 1.6~상한 + 8~11 kHz에서 운 이벤트 수)")
C = {4500: counts(4500), 5500: counts(5500)}
for U, d in C.items():
    n, a = d[d.group == "normal"], d[d.group == "abnormal"]
    for k in (1, 2):
        tp, fp = int((a.off >= k).sum()), int((n.off >= k).sum())
        L.append(f"상한 {U/1000:.1f} kHz | off >= {k}: 고장 {tp}/30, 정상 {fp}/120 ({fp/120*100:.1f} %) | 같은 150개 F2 {f2(tp, 30-tp, fp):.3f}")
    L.append(f"상한 {U/1000:.1f} kHz | low 대역만 >= 2: 고장 {int((a.low >= 2).sum())}/30, 정상 {int((n.low >= 2).sum())}/120")
L.append("\n4.5 -> 5.5 kHz로 넓혔을 때 달라지는 파일")
for k in (1, 2):
    a0, a1 = C[4500], C[5500]
    gain = a1[(a1.off >= k) & ~(a0.off >= k)]
    L.append(f"off >= {k}: 새로 켜지는 고장 " + (", ".join(gain[gain.group == 'abnormal'].file) or "없음") + " | 새로 켜지는 정상 " + (", ".join(gain[gain.group == 'normal'].file) or "없음"))
e45 = E[(E.f >= 4500) & (E.f < 5500)]
L.append(f"4.5-5.5 kHz 대표 울림 이벤트: 정상 {int((e45.group=='normal').sum())}/{int((E.group=='normal').sum())} ({(e45.group=='normal').sum()/(E.group=='normal').sum()*100:.1f} %), 고장 {int((e45.group=='abnormal').sum())}/{int((E.group=='abnormal').sum())} ({(e45.group=='abnormal').sum()/(E.group=='abnormal').sum()*100:.1f} %), 파일 수 정상 {e45[e45.group=='normal'].file.nunique()}, 고장 {e45[e45.group=='abnormal'].file.nunique()}")
L.append("abn_023: off(4.5 상한) = %d, off(5.5 상한) = %d" % (C[4500].set_index('file').loc['abn_023', 'off'], C[5500].set_index('file').loc['abn_023', 'off']))

# %% 계산 ② 지표 (off = 1.6–4.5 + 8–11 kHz, Gemini 기본안)
d4 = C[4500].set_index("file")
off_ev = E[((E.f >= 1600) & (E.f < 4500)) | ((E.f >= HIGH[0]) & (E.f < HIGH[1]))]
g = off_ev.groupby("file")
M = files.set_index("file").copy()
M["off_cnt"] = d4.off
M["off_max_rel_amp"] = g.x.max(); M["off_med_rel_amp"] = g.x.median(); M["off_mean_fit"] = g.fit.mean()
M[["off_max_rel_amp", "off_med_rel_amp", "off_mean_fit"]] = M[["off_max_rel_amp", "off_med_rel_amp", "off_mean_fit"]].fillna(0)
# 이벤트 간 시간 간격 (전체 이벤트): 2개 이상일 때만
def gaps(t):
    t = np.sort(t.values); return np.diff(t)
gp = E.groupby("file").t.apply(gaps)
M["gap_med_s"] = [np.median(gp[f]) if f in gp.index and len(gp[f]) else np.nan for f in M.index]
M["gap_min_s"] = [np.min(gp[f]) if f in gp.index and len(gp[f]) else np.nan for f in M.index]
M["gap_cv"] = [(np.std(gp[f]) / np.mean(gp[f])) if f in gp.index and len(gp[f]) >= 2 else np.nan for f in M.index]
M["n_total"] = F.set_index("file").n_ev25
M["base_med_g"] = E.groupby("file").baseline_g.median()              # 진단용
M["off_base_med_g"] = off_ev.groupby("file").baseline_g.median()    # 진단용
M.to_csv(os.path.join(HERE, "figs", "file_metrics.csv"))

abn = M[M.group == "abnormal"]; nor = M[M.group == "normal"]
fl = nor[nor.off_cnt >= 2]; oth = nor[nor.off_cnt < 2]
L.append(f"\n계산 ② 기본안(상한 4.5 kHz)에서 off >= 2에 걸린 정상 {len(fl)}개: " + ", ".join(fl.index))

def auc_ci(a, b, B=2000):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3: return np.nan, np.nan, np.nan, len(a), len(b)
    u = mannwhitneyu(a, b).statistic / (len(a) * len(b))
    bs = [mannwhitneyu(rng.choice(a, len(a)), rng.choice(b, len(b))).statistic / (len(a) * len(b)) for _ in range(B)]
    return u, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(a), len(b)
cols = ["off_cnt", "off_max_rel_amp", "off_med_rel_amp", "off_mean_fit", "n_total", "gap_med_s", "gap_min_s", "gap_cv"]
groups = {"고장 30": abn, "정상 오경보 18(off>=2)": fl, "정상 나머지 102": oth, "정상 전체 120": nor}
L.append("\n지표별 중앙값 [IQR] (대상 수)")
for c in cols:
    row = []
    for nme, d in groups.items():
        v = d[c].dropna()
        row.append(f"{nme}: {v.median():.2f} [{v.quantile(.25):.2f}, {v.quantile(.75):.2f}] (n={len(v)})")
    L.append(f"{c:16s} " + " | ".join(row))
L.append("\n단독 AUC (고장 > 비교 집단, 95 % 부트스트랩 구간) · 방향 반대면 0.5보다 작게 나옴")
for c in cols:
    for nme in ("정상 오경보 18(off>=2)", "정상 전체 120"):
        u, lo, hi, na, nb = auc_ci(abn[c], groups[nme][c])
        L.append(f"{c:16s} vs {nme:22s} AUC {u:.2f} [{lo:.2f}, {hi:.2f}] (고장 {na}, 비교 {nb})")
L.append("\n오경보 가설 점검 (진단용, g 단위라 피처로 쓰지 않음): 조용한 바탕 때문에 오경보가 났는가?")
for c in ("base_med_g", "off_base_med_g"):
    row = []
    for nme, d in groups.items():
        v = d[c].dropna(); row.append(f"{nme}: {v.median():.4f} [{v.quantile(.25):.4f}, {v.quantile(.75):.4f}] (n={len(v)})")
    L.append(f"{c:15s} " + " | ".join(row))
u, lo, hi, na, nb = auc_ci(fl.off_base_med_g, abn.off_base_med_g)
L.append(f"오경보 18의 off 이벤트 바탕 < 고장의 것 (AUC {u:.2f} [{lo:.2f}, {hi:.2f}])")
L.append("\n오경보 정상 18개 개별: file off수 최대배수 중앙배수 fit평균 총이벤트 바탕(g)")
for f in fl.sort_values("off_cnt", ascending=False).index:
    r = fl.loc[f]; L.append(f"{f} {int(r.off_cnt):3d} {r.off_max_rel_amp:7.1f} {r.off_med_rel_amp:6.1f} {r.off_mean_fit:5.2f} {int(r.n_total):3d} {r.off_base_med_g:.4f}")
L.append("\n고장 중 최대배수가 낮은 파일(오름차순 5개): " + ", ".join(f"{f}({abn.off_max_rel_amp[f]:.1f})" for f in abn.sort_values("off_max_rel_amp").index[:5]))
open(os.path.join(HERE, "figs", "계산12_수치.txt"), "w", encoding="utf-8").write("\n".join(L))

# %% 그림 (계산 ②): 세 지표를 세 집단에서
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_N, C_A, C_F = "#2a78d6", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(1, 4, figsize=(17, 5), facecolor=SURF)
spec = [("off_max_rel_amp", "off 대역 이벤트 배수 최댓값", True), ("off_med_rel_amp", "off 대역 이벤트 배수 중앙값", True),
        ("off_mean_fit", "off 대역 이벤트 fit 평균", False), ("off_base_med_g", "off 이벤트 순간의 바탕 (g, 진단용)", True)]
order = [("정상 나머지 102", oth, C_N), ("정상 오경보 18", fl, C_F), ("고장 30", abn, C_A)]
for a, (c, ttl, lg) in zip(ax, spec):
    labs = []
    a.set_facecolor(SURF); a.grid(True, axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)
    for k, (nm, d, col) in enumerate(order):
        v = d[c].values; v = v[v > 0]            # off 이벤트가 없는 파일(값 0)은 모든 그림에서 제외
        a.scatter(k + rng.uniform(-.25, .25, len(v)), v, s=14, color=col, alpha=.6, lw=0)
        if len(v): a.hlines(np.median(v), k - .34, k + .34, color=INK, lw=2)
        labs.append(nm.replace(" ", "\n", 1) + f"\n(그림 {len(v)}개)")
    if lg:
        a.set_yscale("log"); a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    a.set_xticks(range(3)); a.set_xticklabels(labs, fontsize=9)
    a.set_title(ttl, loc="left", color=INK, fontsize=10.5)
fig.text(0.01, 0.005, "상한 4.5 kHz 기본안. off = 1.6–4.5 + 8–11 kHz 대표 울림 이벤트. 검은 선 = 중앙값. off 이벤트가 없는 파일(값 0)은 모든 그림에서 제외했다('그림 n개' = 실제로 그려진 파일 수).", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(HERE, "figs", "계산2_지표.png"), dpi=140, facecolor=SURF)
print("done")
