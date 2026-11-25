# 임시 (11-22 밤): 경계 묶음 ② 충격성이 강한 정상(H3_kurt_1-2k · 박자 S 높음). 그 충격은 고장 충격과 다르게 생겼나
# 그림3 = 파일별: 스펙트로그램 0–12.5 kHz(2초) + 충격 / 충격 직후 8 ms 평균 스펙트럼 vs 정상 · 고장 집단 중앙
# 그림4 = 라벨 없는 전수: 충격 직후 8 ms 에너지의 옥타브 몫(1–2 · 2–4 · 4–8 · 8–12.5 kHz, 기존 피처와 같은 칸) × H3_kurt_1-2k
#        옥타브 경계는 기존 H 피처 정의 그대로(새로 고르지 않음)
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from src.exp import *
from src.exp2util import NS, NOR, ABN, PUR, short
from src.impacts import detect_impacts, highpass
HERE = os.path.dirname(os.path.abspath(__file__)); FS = 25000; W = int(.008 * FS)
ns = NS(); y, names = ns.y, np.array(ns.names); nor = y == 0
OCT = [(1000, 2000), (2000, 4000), (4000, 8000), (8000, 12500)]; ON = ["1–2k", "2–4k", "4–8k", "8–12.5k"]
fr = np.fft.rfftfreq(W, 1 / FS); win = np.hanning(W)
out = []; P = lambda s="": (out.append(str(s)), print(s))


def imp_spec(x):
    """충격마다 직후 8 ms(1 kHz 고역) 파워 스펙트럼 → (충격 수, 주파수) dB, 충격 시점"""
    imp = detect_impacts(x); xh = highpass(x)
    S = [np.abs(np.fft.rfft(xh[j: j + W] * win)) ** 2 for j in imp["idx"] if j + W <= len(x)]
    return (10 * np.log10(np.array(S) + 1e-30) if S else np.zeros((0, len(fr)))), imp


rows, SPEC = [], {}
for X_, nm_, lab_ in ((ns.X, names, y), (ns.Xv, np.array(ns.nv), np.full(len(ns.nv), 2))):
    for k in range(len(X_)):
        S, imp = imp_spec(X_[k]); SPEC[nm_[k]] = S
        if len(S):
            p = 10 ** (S / 10); tot = p[:, fr >= 1000].sum(1)
            sh = [float(np.median(p[:, (fr >= a) & (fr < b)].sum(1) / tot)) for a, b in OCT]
        else:
            sh = [np.nan] * 4
        rows.append(dict(name=nm_[k], y=int(lab_[k]), nimp=len(imp["idx"]), **{f"몫_{o}": v for o, v in zip(ON, sh)}))
D = pd.DataFrame(rows); D["H3"] = list(ns.T["H3_kurt_1-2k"].values) + list(ns.Tv["H3_kurt_1-2k"].values)
D["S"] = list(ns.T["L1_S"].values) + list(ns.Tv["L1_S"].values)
D.to_csv(os.path.join(HERE, "충격성정상_전수.csv"), index=False, encoding="utf-8-sig")
# 집단 중앙 스펙트럼 (파일마다 충격 평균 → 파일들의 중앙)
med = lambda nms: np.median([SPEC[n].mean(0) for n in nms if len(SPEC[n])], 0)
REF_N, REF_A = med(names[nor]), med(names[~nor])

# %% 그림3
BN = ["nor_027", "nor_073", "nor_066", "nor_095", "nor_106", "nor_110"]
BA = ["abn_009", "abn_028", "abn_023"]
SHOW = BN + BA + ["nor_001"]
fig, axs = plt.subplots(len(SHOW), 2, figsize=(17, 2.5 * len(SHOW)), gridspec_kw=dict(width_ratios=[3.2, 1]))
for r, n in enumerate(SHOW):
    i = int(np.flatnonzero(names == n)[0]); x = ns.X[i] - ns.X[i].mean(); S, imp = SPEC[n], detect_impacts(ns.X[i])
    f, t, Sg = spectrogram(x, fs=FS, nperseg=512, noverlap=384); L = 10 * np.log10(Sg + 1e-30)
    ax = axs[r, 0]; ax.pcolormesh(t, f / 1000, L, shading="auto", cmap="magma", vmin=np.percentile(L, 5), vmax=np.percentile(L, 99.9))
    for ti in imp["t"]: ax.axvline(ti, color="cyan", lw=.7, alpha=.7)
    d = D.set_index("name").loc[n]
    ax.set_title(f"{short(n)} ({'고장' if y[i] else '정상'}) · 충격 {int(d.nimp)}개 · H3_kurt_1-2k {d.H3:.2f} · 박자 S {d.S:.2f} · 충격 몫 1–2k {d['몫_1–2k']:.2f} / 8–12.5k {d['몫_8–12.5k']:.2f}",
                 fontsize=9.5, loc="left", color=ABN if y[i] else NOR)
    ax.set_ylabel("kHz"); ax.set_ylim(0, 12.5)
    ax = axs[r, 1]
    ax.plot(fr / 1000, REF_N, color=NOR, lw=1.2, alpha=.7, label="정상 집단 중앙"); ax.plot(fr / 1000, REF_A, color=ABN, lw=1.2, alpha=.7, label="고장 집단 중앙")
    if len(S): ax.plot(fr / 1000, S.mean(0), color="#222", lw=1.6, label="이 파일 (충격 평균)")
    ax.set_xlim(1, 12.5); ax.grid(alpha=.3); ax.legend(fontsize=6.5, loc="lower left"); ax.set_title("충격 직후 8 ms 스펙트럼 (dB)", fontsize=8.5)
axs[-1, 0].set_xlabel("시간 (s) — 하늘색 선 = 충격"); axs[-1, 1].set_xlabel("kHz")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림3_충격성정상_파일별.png"), dpi=105); plt.close(fig)

# %% 그림4: 라벨 없는 전수
fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
for ax, col_ in zip(axs, ["몫_8–12.5k", "몫_1–2k"]):
    for lab_, c, mk, nm in ((0, NOR, "s", "train 정상"), (1, ABN, "o", "train 고장"), (2, PUR, "^", "val")):
        m = D.y == lab_; ax.scatter(D.H3[m], D[col_][m], s=30, marker=mk, color=c, alpha=.65, lw=0, label=f"{nm} ({int(m.sum())}, 충격 0개 {int((m & D[col_].isna()).sum())} 제외)")
    for n in BN + BA + ["val_036", "val_012", "val_025"]:
        r_ = D[D.name.str.endswith(n[-3:]) & (D.name.str.startswith(n[:3]) | (n.startswith("val") & (D.y == 2)))]
        r_ = r_[r_.y == (2 if n.startswith("val") else (1 if n.startswith("abn") else 0))].iloc[0]
        ax.annotate(("V" + n[-3:]) if n.startswith("val") else short(n), (r_.H3, r_[col_]), xytext=(4, 4), textcoords="offset points", fontsize=8,
                    color=PUR if r_.y == 2 else (ABN if r_.y else NOR), weight="bold")
    ax.set_xlabel("H3_kurt_1-2k (1–2 kHz 첨도, log10) — ET가 쓰는 충격성"); ax.set_ylabel(f"충격 직후 8 ms 에너지 중 {col_[2:]} 몫 (충격 중앙)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
axs[0].set_title("충격이 고주파(8–12.5k, 금속 모드 칸)를 때리는가", fontsize=9.5); axs[1].set_title("충격 에너지가 저주파(1–2k)에 몰려 있는가", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림4_충격몫_전수.png"), dpi=110); plt.close(fig)

# %% 숫자
T = D[D.y < 2]
for o in ON:
    a, b = T[f"몫_{o}"][T.y == 0].dropna(), T[f"몫_{o}"][T.y == 1].dropna()
    from sklearn.metrics import roc_auc_score
    P(f"충격 몫 {o}: 정상 중앙 {a.median():.3f} · 고장 중앙 {b.median():.3f} · AUC(고장↑) {roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]):.3f}")
P(); P(D[D.name.isin(BN + BA)].round(3).to_string(index=False))
P(); P(D[(D.y == 2) & D.name.str[-3:].isin(["036", "012", "025"])].round(3).to_string(index=False))
open(os.path.join(HERE, "충격성정상_log.txt"), "w", encoding="utf-8").write("\n".join(out))
