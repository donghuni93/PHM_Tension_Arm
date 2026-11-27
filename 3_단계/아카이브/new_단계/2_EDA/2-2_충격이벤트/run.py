# %% 2-2 충격 이벤트: Train 150개(정상 120 · 고장 30)
# 검출기 = src/impacts.py detect_impacts (사양은 README). 라벨은 색칠 · 비교에만 쓴다.
# 보강 두 가지(검출기 사양은 바꾸지 않음): ① 울림 주파수를 50 Hz 격자로 다시 맞춤 ② 가까운 검출을 이벤트로 묶음
import glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.impacts import detect_impacts, highpass, _atoms50, FS
from src.final import _SEG

D = os.path.join(ROOT, "2_data", "TeamProject_train_변경")
ZS = (0.01, 0.02, 0.04, 0.08)
GAPS = (10, 25, 50)           # 이벤트로 묶는 간격(ms) 민감도
GAP_MAIN = 25
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

# %% 로드 + 검출
def load(group):
    fl = sorted(glob.glob(os.path.join(D, group, "*.csv")))
    return [os.path.basename(f)[:-4] for f in fl], [np.loadtxt(f) for f in fl]

A50, F50 = _atoms50()

def fine_ring(xh, idx):
    """충격 뒤 8 ms를 50 Hz 격자 사전에 맞춰 (f, ζ, fit)"""
    S = np.stack([xh[p:p + _SEG] for p in idx])
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, A50), axis=2)
    b = fit.argmax(1)
    return np.column_stack([F50[b], np.array(ZS)[b % 4], fit.max(1)])

def clusters(idx, gap_ms):
    """검출 → 이벤트 번호. 앞 검출과의 간격이 gap_ms 이하면 같은 이벤트"""
    if len(idx) == 0:
        return np.array([], int)
    return np.concatenate([[0], np.cumsum(np.diff(idx) > gap_ms * FS / 1000)]).astype(int)

rows, events = [], []
for grp in ("normal", "abnormal"):
    names, Xs = load(grp)
    for n, x in zip(names, Xs):
        xh = highpass(x)
        imp = detect_impacts(x, _hp=xh)
        idx, hg = imp["idx"], imp["height_g"]
        r = dict(file=n, group=grp, n_det=len(idx))
        for g in GAPS:
            r[f"n_ev{g}"] = int(clusters(idx, g).max() + 1) if len(idx) else 0
        rows.append(r)
        if len(idx):
            ring = fine_ring(xh, idx)
            cl = clusters(idx, GAP_MAIN)
            for e in range(cl.max() + 1):
                m = np.where(cl == e)[0]; j = m[np.argmax(hg[m])]
                events.append(dict(file=n, group=grp, t=idx[j] / FS, f=ring[j, 0], z=ring[j, 1], fit=ring[j, 2],
                                   h=hg[j], x=imp["height_x"][j], n_in=len(m)))
R = pd.DataFrame(rows); E = pd.DataFrame(events)
R.to_csv(os.path.join(HERE, "figs", "files.csv"), index=False)
E.to_csv(os.path.join(HERE, "figs", "events.csv"), index=False)

# %% 수치
L = []
auc = lambda a, n: mannwhitneyu(a, n).statistic / (len(a) * len(n))
L.append("1) 파일당 검출 · 이벤트 수 (정상 120 · 고장 30)")
for c in ["n_det"] + [f"n_ev{g}" for g in GAPS]:
    n = R[R.group == "normal"][c].values; a = R[R.group == "abnormal"][c].values
    L.append(f"{c:8s} 정상 중앙 {np.median(n):4.1f} (p10 {np.percentile(n,10):.0f}, p90 {np.percentile(n,90):.0f}, 0개 {np.mean(n==0)*100:3.0f}%, ≤2개 {np.mean(n<=2)*100:3.0f}%)"
             f" | 고장 중앙 {np.median(a):4.1f} (p10 {np.percentile(a,10):.0f}, p90 {np.percentile(a,90):.0f}, 0개 {np.mean(a==0)*100:3.0f}%, ≤2개 {np.mean(a<=2)*100:3.0f}%) | AUC {auc(a,n):.2f}")
L.append("\n2) 이벤트 묶음 확인: 055 · 116 (사용자 점검: 055 = 1회 1.20 s, 116 = 2회 0.62 s · 1.54 s)")
for f in ("nor_055", "nor_116"):
    r = R[R.file == f].iloc[0]; e = E[E.file == f]
    L.append(f"{f}: 검출 {r.n_det}개, 이벤트(10/25/50 ms) {r.n_ev10}/{r.n_ev25}/{r.n_ev50}, 이벤트 시점 {np.round(e.t.values,3).tolist()}, 울림 f(50Hz) {e.f.values.astype(int).tolist()}")
L.append("\n3) 이벤트 대표 울림 주파수 분포 (대역별 비율, 이벤트 수 기준)")
bands = [(1000, 1600), (1600, 4500), (4500, 5500), (5500, 8000), (8000, 12001)]
for g in ("normal", "abnormal"):
    e = E[E.group == g]
    L.append(f"{g}: 이벤트 {len(e)}개 / 파일 {(R.group==g).sum()}개 | " + ", ".join(
        f"{a/1000:.1f}-{min(b,12500)/1000:.1f}k {np.mean((e.f>=a)&(e.f<b))*100:4.1f}%" for a, b in bands))
L.append("\n4) 9.2–9.7 kHz로 울린 이벤트 (50 Hz 격자)")
for g in ("normal", "abnormal"):
    e = E[(E.group == g) & (E.f >= 9200) & (E.f < 9700)]
    L.append(f"{g}: {len(e)}개 / 이벤트 {int((E.group==g).sum())}개, 파일 {e.file.nunique()}개 → " + ", ".join(
        f"{r.file}:{int(r.f)}Hz(z{r.z},fit{r.fit:.2f})" for r in e.itertuples()))
    if len(e):
        L.append(f"   주파수 중앙 {np.median(e.f):.0f} Hz (범위 {e.f.min():.0f}-{e.f.max():.0f})")
L.append("\n5) 감쇠비 ζ 분포 (이벤트 비율, 사전 4값) · fit")
for g in ("normal", "abnormal"):
    e = E[E.group == g]
    L.append(f"{g}: " + ", ".join(f"z{z}: {np.mean(np.isclose(e.z,z))*100:4.1f}%" for z in ZS) + f" | fit 중앙 {e.fit.median():.2f} (p10 {e.fit.quantile(.1):.2f}, p90 {e.fit.quantile(.9):.2f})")
L.append(f"fit >= 0.7 이벤트 비율: 정상 {np.mean(E[E.group=='normal'].fit>=.7)*100:.0f}%, 고장 {np.mean(E[E.group=='abnormal'].fit>=.7)*100:.0f}%")
open(os.path.join(HERE, "figs", "충격이벤트_수치.txt"), "w", encoding="utf-8").write("\n".join(L))

# %% 그림
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_N, C_A = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), facecolor=SURF)
for a in ax.ravel():
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=10)
rng = np.random.default_rng(0)
a = ax[0, 0]
for k, (g, c) in enumerate((("normal", C_N), ("abnormal", C_A))):
    v = R[R.group == g][f"n_ev{GAP_MAIN}"].values
    a.scatter(k + rng.uniform(-.28, .28, len(v)), v + rng.uniform(-.12, .12, len(v)), s=16, color=c, alpha=.6, lw=0)
    a.hlines(np.median(v), k - .35, k + .35, color=INK, lw=2)
a.set_xticks([0, 1]); a.set_xticklabels(["정상 120", "고장 30"]); a.set_xlim(-.6, 1.6)
a.set_ylabel("2초당 이벤트 수 (25 ms 이내 묶음)", color=INK2); a.set_title("(a) 파일당 충격 이벤트 수 (검은 선 = 중앙값)", loc="left", color=INK, fontsize=12)
a = ax[0, 1]; edges = np.arange(1000, 12501, 500)
for g, c in (("normal", C_N), ("abnormal", C_A)):
    e = E[E.group == g]; h, _ = np.histogram(e.f, bins=edges); a.step(edges[:-1] / 1000, h / len(e), where="post", color=c, lw=1.8)
a.set_xlabel("대표 울림 주파수 (kHz)", color=INK2); a.set_ylabel("이벤트 비율", color=INK2)
a.set_title("(b) 이벤트 울림 주파수 (50 Hz 격자 맞춤)", loc="left", color=INK, fontsize=12)
a = ax[1, 0]; w = .38
for k, (g, c) in enumerate((("normal", C_N), ("abnormal", C_A))):
    e = E[E.group == g]; v = [np.mean(np.isclose(e.z, z)) for z in ZS]
    a.bar(np.arange(4) + (k - .5) * w, v, width=w - .04, color=c, edgecolor=SURF, lw=1.5)
a.set_xticks(range(4)); a.set_xticklabels([f"ζ {z}" for z in ZS]); a.set_ylabel("이벤트 비율", color=INK2)
a.set_title("(c) 감쇠비 ζ (사전 4값 중 가장 닮은 값)", loc="left", color=INK, fontsize=12)
a = ax[1, 1]
for g, c in (("normal", C_N), ("abnormal", C_A)):
    v = np.sort(E[E.group == g].fit.values); a.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=1.8)
a.set_xlabel("맞춤도 fit (1 = 감쇠 사인파와 같은 모양)", color=INK2); a.set_ylabel("누적 비율", color=INK2)
a.set_title("(d) 울림 맞춤도 분포", loc="left", color=INK, fontsize=12)
for aa, y0 in ((ax[0, 1], 0.93), (ax[1, 1], 0.28), (ax[1, 0], 0.93)):
    aa.annotate("정상 120", xy=(0.97, y0), xycoords="axes fraction", ha="right", color=C_N, fontsize=10)
    aa.annotate("고장 30", xy=(0.97, y0 - 0.07), xycoords="axes fraction", ha="right", color=C_A, fontsize=10)
fig.text(0.01, 0.005, "Train 150개. 검출기 = 1 kHz 고역 → |Hilbert| → 바탕(100 ms 이동 중앙값) 대비 ≥ 4.5배 봉우리, 간격 ≥ 6.75 ms. 울림 = 충격 뒤 8 ms를 감쇠 사인파 사전(1–12 kHz, 50 Hz × ζ 4값)에 투영.",
         color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(os.path.join(HERE, "figs", "충격이벤트.png"), dpi=140, facecolor=SURF)
print("done")
