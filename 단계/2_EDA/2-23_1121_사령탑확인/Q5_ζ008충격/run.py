# %% [markdown]
# Q5 ζ = 0.08 충격의 정체 (사령탑 확인 요청, 11-21)
# 질문: 사전 맨 끝 칸 ζ = 0.08은 "0.08 이상" 또는 "깨끗한 울림이 아님"을 한꺼번에 담고 있나
# 방법 (src/final.impact_rings와 같은 충격 찾기·맞춤, 사전만 넓힘. src는 고치지 않음):
#   충격 = 평균 제거 → 1 kHz 고역통과 → Hilbert 포락선 → 500 Hz 저역 → 포락선 > 6 x 중앙값 봉우리(5 ms 간격)
#   맞춤 = 충격 뒤 8 ms를 사전 기저(주파수 1–12 kHz 250 Hz 간격 x ζ)에 투영한 길이 fit (1 = 완전히 같은 모양)
#   사전 A = ζ {0.01, 0.02, 0.04, 0.08} (현재 최종 모델), 사전 B = A + {0.16, 0.32}
#   꺼짐 시간 tau = 1/(ζ 2π f) ms  (울림 진폭이 1/e로 줄어드는 시간)
#   잡음 기준 = 충격이 아닌 8 ms 구간(충격에서 10 ms 이상 떨어진 임의 위치)을 같은 사전으로 맞춘 fit
# 라벨은 색칠·비교에만 쓴다. 실행: conda run -n qe-phm python run.py
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
p = HERE
while not os.path.isdir(os.path.join(p, "src")): p = os.path.dirname(p)
ROOT = p; sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from sklearn.metrics import roc_auc_score
from src.exp import load_train
from src.data import load_folder
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FS = 25000; SEG = int(0.008 * FS); NOR, ABN = "#2a78d6", "#eb6834"
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
ZS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)

def atoms():
    t = np.arange(SEG) / FS; A, par = [], []
    for f in np.arange(1000, 12001, 250):
        for z in ZS:
            w = 2 * np.pi * f
            Q, _ = np.linalg.qr(np.stack([np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z ** 2) * t), np.exp(-z * w * t) * np.cos(w * np.sqrt(1 - z ** 2) * t)], 1))
            A.append(Q); par.append((f, z))
    return np.stack(A), np.array(par)
A_, PAR = atoms()
m4 = PAR[:, 1] <= 0.08 + 1e-9

def assign(S):
    """S (k, SEG) -> 사전 A·B 각각의 (f, ζ, fit)"""
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, A_), axis=2)       # (k, 원자)
    out = []
    for mk in (m4, np.ones(len(PAR), bool)):
        idx = np.flatnonzero(mk); b = idx[fit[:, idx].argmax(1)]
        out.append((PAR[b, 0], PAR[b, 1], fit[np.arange(len(S)), b]))
    return out

lp500 = butter(4, 500, btype="lowpass", fs=FS, output="sos")
hp1k = butter(4, 1000, btype="highpass", fs=FS, output="sos")
rng = np.random.default_rng(0)
def impacts(X, names, null=False):
    rows, nulls = [], []
    X1k = sosfiltfilt(hp1k, X - X.mean(1, keepdims=True), axis=-1)
    for i, x in enumerate(X1k):
        env = sosfiltfilt(lp500, np.abs(hilbert(x)))
        pk, _ = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS)); pk = pk[pk + SEG <= len(x)]
        if len(pk):
            S = np.stack([x[q:q + SEG] for q in pk]); a, b = assign(S)
            for k in range(len(pk)):
                rows.append((i, names[i], pk[k], a[0][k], a[1][k], a[2][k], b[0][k], b[1][k], b[2][k]))
        if null:
            c = [q for q in rng.integers(0, len(x) - SEG, 60) if len(pk) == 0 or np.abs(pk - q).min() > int(0.010 * FS)][:3]
            if c:
                a, b = assign(np.stack([x[q:q + SEG] for q in c]))
                nulls += [(a[0][k], a[1][k], a[2][k]) for k in range(len(c))]
    cols = ["file", "name", "pos", "fA", "zA", "fitA", "fB", "zB", "fitB"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(nulls, columns=["fA", "zA", "fitA"])

X, y, names = load_train(); y = np.asarray(y); names = [str(n) for n in names]
D, N0 = impacts(X, names, null=True); D["y"] = y[D.file.values]
XV, vn = load_folder(os.path.join(ROOT, "data", "TeamProject_val", "val"))
DV, _ = impacts(XV, vn)
D["tauA"] = 1000 / (D.zA * 2 * np.pi * D.fA); D["tauB"] = 1000 / (D.zB * 2 * np.pi * D.fB)
D.to_csv(os.path.join(HERE, "impacts.csv"), index=False, encoding="utf-8-sig")
out = [f"충격: train {len(D)}개 (정상 {(D.y==0).sum()}, 고장 {(D.y==1).sum()}), val {len(DV)}개. 잡음 기준 구간 {len(N0)}개", ""]

# %% 1. fit 분포: 사전 A에서 ζ별
out.append("[1] 사전 A(ζ 0.01~0.08)에서 뽑힌 ζ별 충격 수와 fit 중앙 (정상 / 고장 / val)")
for z in ZS[:4]:
    d = D[np.isclose(D.zA, z)]; dv = DV[np.isclose(DV.zA, z)]
    out.append(f"  ζ {z}: 전체 {len(d)} ({len(d)/len(D):.0%}) 정상 {int((d.y==0).sum())} 고장 {int((d.y==1).sum())} val {len(dv)} ({len(dv)/len(DV):.0%}) | fit 중앙 {d.fitA.median():.3f} (정상 {d[d.y==0].fitA.median():.3f}, 고장 {d[d.y==1].fitA.median():.3f})")
out.append(f"  잡음 기준 구간: fit 중앙 {N0.fitA.median():.3f}, 95% {N0.fitA.quantile(.95):.3f}, ζ=0.08로 뽑힌 비율 {(np.isclose(N0.zA,.08)).mean():.0%}")
d8 = D[np.isclose(D.zA, .08)]
out.append(f"  ζ=0.08 충격 중 fit이 잡음 기준 95%({N0.fitA.quantile(.95):.3f})보다 낮은 비율: {(d8.fitA < N0.fitA.quantile(.95)).mean():.0%}")
out.append("  ζ=0.08 충격의 주파수 대역: " + ", ".join(f"{lo}-{hi}k {((d8.fA>=lo*1000)&(d8.fA<hi*1000)).mean():.0%}" for lo, hi in ((1,2.5),(2.5,4.5),(4.5,6.5),(6.5,8),(8,12.25))))
out.append("  ζ=0.08 충격의 주파수 대역 (고장만): " + ", ".join(f"{lo}-{hi}k {((d8[d8.y==1].fA>=lo*1000)&(d8[d8.y==1].fA<hi*1000)).mean():.0%}" for lo, hi in ((1,2.5),(2.5,4.5),(4.5,6.5),(6.5,8),(8,12.25))))

# %% 2. 사전 B(0.16·0.32 추가)로 다시 뽑기
out.append("\n[2] 사전 B에서 다시 뽑힌 ζ (행 = 사전 A에서 뽑힌 ζ, 열 = 사전 B에서 뽑힌 ζ), 충격 수")
M = pd.crosstab(D.zA, D.zB).reindex(index=ZS[:4], columns=ZS, fill_value=0)
out.append(M.to_string())
r8 = D[np.isclose(D.zA, .08)]
out.append(f"  ζ=0.08이던 충격 {len(r8)}개 중 → 0.16: {np.isclose(r8.zB,.16).mean():.0%}, 0.32: {np.isclose(r8.zB,.32).mean():.0%}, 그대로 0.08: {np.isclose(r8.zB,.08).mean():.0%}, 더 낮은 ζ로: {(r8.zB<.08-1e-9).mean():.0%}")
out.append(f"  ζ=0.08이던 충격의 fit 변화(B-A) 중앙 {(r8.fitB-r8.fitA).median():+.3f}; 새로 0.16/0.32가 된 것의 중앙 {(r8[r8.zB>.08+1e-9].fitB-r8[r8.zB>.08+1e-9].fitA).median():+.3f}")
for z in (0.01, 0.02, 0.04):
    r = D[np.isclose(D.zA, z)]; out.append(f"  (참고) ζ={z}이던 충격 중 사전 B에서도 같은 값 유지: {np.isclose(r.zB, z).mean():.0%}")
out.append("  사전 B에서 뽑힌 ζ 분포 (정상 / 고장 비율): " + ", ".join(f"{z}: {((D[D.y==0].zB.round(3)==z).mean()):.0%}/{((D[D.y==1].zB.round(3)==z).mean()):.0%}" for z in ZS))

# %% 3. 꺼짐 시간 tau = 1/(ζ 2π f) ms
out.append("\n[3] 꺼짐 시간 tau (ms), 사전 B 기준")
for k, g in ((0, "정상"), (1, "고장")):
    t = D[D.y == k].tauB
    out.append(f"  {g}: 중앙 {t.median():.2f} · 10% {t.quantile(.1):.2f} · 90% {t.quantile(.9):.2f} | tau < 0.3 ms {(t<.3).mean():.0%} · < 0.5 ms {(t<.5).mean():.0%} · < 1 ms {(t<1).mean():.0%} · > 3 ms {(t>3).mean():.0%}")
out.append("  ζ별 tau 중앙 (정상/고장) — ζ가 같아도 주파수가 높으면 tau가 짧아짐:")
for z in ZS:
    a = D[(D.y == 0) & np.isclose(D.zB, z)].tauB; b = D[(D.y == 1) & np.isclose(D.zB, z)].tauB
    out.append(f"   ζ {z}: 정상 {a.median():.2f} ({len(a)}) / 고장 {b.median():.2f} ({len(b)})")
# 파일별 후보 C5: 빨리 꺼지는 충격 비율
def frac(df, cond):
    g = df.assign(c=cond(df).astype(float)).groupby("file").c.mean(); return g
out.append("  파일별 '빨리 꺼지는 충격 비율' 후보 (충격 있는 파일, AUC는 사후):")
fy = pd.Series(y)
for tag, cond in (("ζ_A >= 0.08 (현재 사전 끝 칸)", lambda d: d.zA >= .08 - 1e-9), ("ζ_B >= 0.08", lambda d: d.zB >= .08 - 1e-9), ("ζ_B >= 0.16", lambda d: d.zB >= .16 - 1e-9),
                  ("tau_B < 0.5 ms", lambda d: d.tauB < .5), ("tau_B < 0.3 ms", lambda d: d.tauB < .3), ("tau_A < 0.5 ms", lambda d: d.tauA < .5)):
    g = frac(D, cond); yy = fy[g.index].values
    out.append(f"   {tag}: AUC {roc_auc_score(yy, g.values):.2f} | 파일 평균 정상 {g[yy==0].mean():.1%} 고장 {g[yy==1].mean():.1%} | 비율>0인 파일 정상 {(g[yy==0]>0).mean():.0%} 고장 {(g[yy==1]>0).mean():.0%} ({len(g)}개 파일)")
open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8").write("\n".join(out))

# %% 그림 1: fit 분포 (사전 A ζ별 + 잡음 기준)
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
bins = np.linspace(0.3, 1, 36)
ax = axs[0]
for z, c in zip(ZS[:4], ("#1b9e77", "#7570b3", "#e6ab02", "#d62728")):
    ax.hist(D[np.isclose(D.zA, z)].fitA, bins=bins, histtype="step", lw=1.6, color=c, label=f"ζ {z} ({np.isclose(D.zA, z).sum()}개)")
ax.hist(N0.fitA, bins=bins, color="gray", alpha=.35, label="잡음 기준 구간(충격 아님)")
ax.set_xlabel("fit (1 = 완전히 감쇠 사인파)"); ax.set_ylabel("충격 수"); ax.set_title("사전 A에서 뽑힌 ζ별 fit"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axs[1]
d8 = D[np.isclose(D.zA, .08)]
ax.hist(d8[d8.y == 0].fitA, bins=bins, histtype="step", lw=1.8, color=NOR, label=f"ζ 0.08 정상 {int((d8.y==0).sum())}개")
ax.hist(d8[d8.y == 1].fitA, bins=bins, histtype="step", lw=1.8, color=ABN, label=f"ζ 0.08 고장 {int((d8.y==1).sum())}개")
ax.hist(N0.fitA, bins=bins, color="gray", alpha=.35, label="잡음 기준")
ax.axvline(N0.fitA.quantile(.95), color="k", ls="--", lw=1, label="잡음 95%")
ax.set_xlabel("fit"); ax.set_title("ζ = 0.08 충격: fit이 잡음 기준과 얼마나 다른가"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axs[2]
for k, c, g in ((0, NOR, "정상"), (1, ABN, "고장")):
    d = D[(D.y == k) & np.isclose(D.zA, .08)]
    ax.scatter(d.fA / 1000, d.fitA, s=10, c=c, alpha=.5, label=f"ζ 0.08 {g}")
ax.set_xlabel("울림 주파수 (kHz)"); ax.set_ylabel("fit"); ax.set_title("ζ = 0.08 충격: 주파수 대 fit"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q5_fit.png"), dpi=90); plt.close(fig)

# %% 그림 2: 다시 뽑기 행렬 + fit 변화
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
ax = axs[0]
Mn = M.div(M.sum(1), axis=0).fillna(0)
im = ax.imshow(Mn.values, cmap="Blues", vmin=0, vmax=1)
for i in range(Mn.shape[0]):
    for j in range(Mn.shape[1]):
        ax.text(j, i, f"{int(M.values[i, j])}\n{Mn.values[i, j]:.0%}", ha="center", va="center", fontsize=8, color="w" if Mn.values[i, j] > .5 else "k")
ax.set_xticks(range(6), [str(z) for z in ZS]); ax.set_yticks(range(4), [str(z) for z in ZS[:4]])
ax.set_xlabel("사전 B(0.16 · 0.32 추가)에서 뽑힌 ζ"); ax.set_ylabel("사전 A에서 뽑힌 ζ"); ax.set_title("ζ 사전을 넓히면 다시 뽑히는 값 (수, 행 비율)")
ax = axs[1]
bb = np.linspace(-0.1, 0.3, 41)
for z, c in zip(ZS[:4], ("#1b9e77", "#7570b3", "#e6ab02", "#d62728")):
    d = D[np.isclose(D.zA, z)]; ax.hist(d.fitB - d.fitA, bins=bb, histtype="step", lw=1.6, color=c, label=f"A에서 ζ {z}")
ax.set_xlabel("fit 변화 (사전 B - 사전 A)"); ax.set_ylabel("충격 수"); ax.set_yscale("log"); ax.set_title("사전을 넓히면 맞춤이 얼마나 좋아지나"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q5_reassign.png"), dpi=90); plt.close(fig)

# %% 그림 3: 꺼짐 시간 tau 분포
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
ax = axs[0]; tb = np.logspace(-1.3, 1.2, 40)
for k, c, g in ((0, NOR, "정상"), (1, ABN, "고장")):
    t = D[D.y == k].tauB; ax.hist(t, bins=tb, histtype="step", lw=1.8, color=c, density=True, label=f"{g} {len(t)}개 (중앙 {t.median():.2f} ms)")
ax.set_xscale("log"); ax.set_xticks([.1, .2, .5, 1, 2, 5, 10]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
ax.set_xlabel("꺼짐 시간 tau = 1/(ζ 2π f) (ms)"); ax.set_ylabel("밀도"); ax.set_title("충격이 꺼지는 데 걸리는 시간 (사전 B 기준)"); ax.legend(); ax.grid(alpha=.3, which="both")
ax = axs[1]
for k, c, g in ((0, NOR, "정상"), (1, ABN, "고장")):
    d = D[D.y == k]; ax.scatter(d.fB / 1000, d.tauB, s=8, c=c, alpha=.4, label=g)
ax.set_yscale("log"); ax.set_yticks([.1, .2, .5, 1, 2, 5, 10]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
ax.set_xlabel("울림 주파수 (kHz)"); ax.set_ylabel("꺼짐 시간 tau (ms)"); ax.set_title("주파수 대 꺼짐 시간 (ζ가 같아도 주파수가 높으면 tau가 짧다)"); ax.legend(); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q5_tau.png"), dpi=90); plt.close(fig)
print("done")
