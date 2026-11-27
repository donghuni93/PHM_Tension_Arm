# 임시 (11-22): 059 · 065 · 116의 H10(8–12.5k 울림 비율)은 "한 번의 충격"에서 오는가, "지속음"이 충격 울림으로 잘못 맞춰진 것인가
# 그림1 = 파일별 스펙트로그램(6–12.5 kHz, 2초) + 충격 시점 · 충격마다 맞춘 울림 f / 가장 큰 충격 앞 50 ms vs 뒤 8 ms 스펙트럼
# 그림2 = 라벨 없는 전수: 충격 구간을 뺀 배경의 8–12.5 kHz 좁은 봉우리 돌출(dB) × H10. 지속음이 몇 개 파일에 있는가 (과적합 판단용)
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch, medfilt
from src.exp import *
from src.exp2util import NS, NOR, ABN, short
from src.impacts import detect_impacts, impact_rings_v2
HERE = os.path.dirname(os.path.abspath(__file__)); FS = 25000
ns = NS(); y, names = ns.y, np.array(ns.names); H10 = ns.T["H10_ring_8-12.5k"].values
out = []; P = lambda s="": (out.append(str(s)), print(s))


def bg_tone(x, imp_idx, guard=int(.03 * FS)):
    """충격 ±30 ms를 뺀 배경 구간의 PSD(4096점)에서 8–12.5 kHz 좁은 봉우리 돌출 = PSD − 이동중앙값(≈600 Hz), dB 최대 · 그 주파수"""
    m = np.ones(len(x), bool)
    for i in imp_idx: m[max(0, i - guard): i + guard] = False
    xs = (x - x.mean())[m]
    f, p = welch(xs, fs=FS, nperseg=4096, detrend=False)
    L = 10 * np.log10(p + 1e-30); base = medfilt(L, 101); pr = L - base; sel = (f >= 8000) & (f <= 12400)
    j = np.argmax(np.where(sel, pr, -np.inf)); return float(pr[j]), float(f[j]), float(m.mean())


# %% 전수 계산 (train 150 + val 40)
rows = []
for X_, nm_, lab_ in ((ns.X, names, y), (ns.Xv, np.array(ns.nv), np.full(len(ns.nv), 2))):
    for k in range(len(X_)):
        imp = detect_impacts(X_[k]); rg = impact_rings_v2(X_[k], imp=imp)
        tp, tf, keepfrac = bg_tone(X_[k], imp["idx"])
        rows.append(dict(name=nm_[k], y=int(lab_[k]), nimp=len(imp["idx"]), h10=float(np.mean((rg[:, 0] >= 8000) & (rg[:, 0] < 12500))) if len(rg) else -1,
                         tone_db=tp, tone_f=tf, bgfrac=keepfrac))
D = pd.DataFrame(rows); D.to_csv(os.path.join(HERE, "h10_지속음_전수.csv"), index=False, encoding="utf-8-sig")
tr = D[D.y < 2]; assert np.allclose(tr.h10.values, H10, atol=1e-6), "H10 재계산이 피처 표와 다름"

# %% 그림1: 파일별 상세
FOCUS = ["nor_059", "nor_065", "nor_116", "nor_055"]
hi_abn = tr[(tr.y == 1) & (tr.h10 >= .5)].sort_values("h10", ascending=False).name.tolist()
SHOW = FOCUS + list(dict.fromkeys(["abn_002"] + hi_abn))[:5] + ["nor_001"]
P("H10 ≥ 0.5 인 고장: " + ", ".join(f"{short(n)}(H10 {tr.set_index('name').h10[n]:.2f}, 충격 {tr.set_index('name').nimp[n]})" for n in hi_abn))
P("H10 ≥ 0.5 인 정상: " + ", ".join(f"{short(n)}(H10 {h:.2f}, 충격 {c})" for n, h, c in tr[(tr.y == 0) & (tr.h10 >= .5)][["name", "h10", "nimp"]].values))
fig, axs = plt.subplots(len(SHOW), 2, figsize=(17, 2.55 * len(SHOW)), gridspec_kw=dict(width_ratios=[3.2, 1]))
for r, n in enumerate(SHOW):
    i = int(np.flatnonzero(names == n)[0]); x = ns.X[i] - ns.X[i].mean(); imp = detect_impacts(ns.X[i]); rg = impact_rings_v2(ns.X[i], imp=imp)
    f, t, S = spectrogram(x, fs=FS, nperseg=512, noverlap=384); sel = f >= 6000
    ax = axs[r, 0]; ax.pcolormesh(t, f[sel] / 1000, 10 * np.log10(S[sel] + 1e-30), shading="auto", cmap="magma", vmin=np.percentile(10 * np.log10(S[sel] + 1e-30), 5), vmax=np.percentile(10 * np.log10(S[sel] + 1e-30), 99.9))
    for k, (ti, fr) in enumerate(zip(imp["t"], rg[:, 0] if len(rg) else [])):
        ax.axvline(ti, color="cyan", lw=.8, alpha=.8); ax.plot(ti, fr / 1000, "o", ms=6, mfc="none", mec="lime" if fr >= 8000 else "white", mew=1.6)
    ax.set_ylim(6, 12.5); ax.set_ylabel("kHz")
    d = tr.set_index("name").loc[n]
    ax.set_title(f"{short(n)} ({'고장' if y[i] else '정상'}) · 충격 {int(d.nimp)}개 · H10 {d.h10:.2f} · 배경 8–12.5k 봉우리 돌출 {d.tone_db:.1f} dB @ {d.tone_f / 1000:.2f} kHz", fontsize=9.5, loc="left", color=ABN if y[i] else NOR)
    ax = axs[r, 1]
    if len(imp["idx"]):
        j = imp["idx"][np.argmax(imp["height_g"])]
        for seg, lab_, c in ((x[max(0, j - int(.055 * FS)): j - int(.005 * FS)], "충격 앞 50 ms", "#888"), (x[j: j + int(.008 * FS)], "충격 뒤 8 ms", "#d62728")):
            ff, pp = welch(seg, fs=FS, nperseg=min(len(seg), 1024), detrend=False); ax.plot(ff / 1000, 10 * np.log10(pp + 1e-30), color=c, lw=1, label=lab_)
        ax.set_xlim(6, 12.5); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=.3)
    ax.set_title("가장 큰 충격: 앞 vs 뒤 스펙트럼 (dB)", fontsize=8.5)
axs[-1, 0].set_xlabel("시간 (s) — 하늘색 선 = 충격, 원 = 그 충격에 맞춘 울림 f (초록 = 8–12.5 kHz로 셈 → H10)"); axs[-1, 1].set_xlabel("kHz")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림1_파일별_스펙트로그램.png"), dpi=110); plt.close(fig)

# %% 그림2: 라벨 없는 전수 — 배경 지속음 돌출 × H10
fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
ax = axs[0]
for lab_, c, mk, nm in ((0, NOR, "s", "train 정상"), (1, ABN, "o", "train 고장"), (2, "#7B1FA2", "^", "val")):
    m = D.y == lab_; ax.scatter(D.tone_db[m], D.h10[m] + np.random.RandomState(lab_).uniform(-.015, .015, m.sum()), s=30, marker=mk, color=c, alpha=.7, lw=0, label=f"{nm} ({m.sum()})")
for n in FOCUS + ["abn_002"] + hi_abn[:6]:
    r_ = D[D.name == n].iloc[0]; ax.annotate(short(n), (r_.tone_db, r_.h10), xytext=(4, 4), textcoords="offset points", fontsize=8, color=ABN if r_.y else NOR, weight="bold")
ax.set_xlabel("배경(충격 ±30 ms 제외) 8–12.5 kHz 좁은 봉우리 돌출 (dB) — 클수록 지속음"); ax.set_ylabel("H10 = 8–12.5 kHz로 맞춰진 충격 비율 (충격 0개 = −1)")
ax.legend(fontsize=8.5); ax.grid(alpha=.3); ax.set_title("라벨 없이 계산. 오른쪽 위 = 지속음이 있고 충격 울림도 8–12.5k로 맞춰진 파일", fontsize=9.5)
ax = axs[1]
bins = np.linspace(D.tone_db.min(), D.tone_db.max(), 30)
for lab_, c, nm in ((0, NOR, "train 정상"), (1, ABN, "train 고장"), (2, "#7B1FA2", "val")):
    ax.hist(D.tone_db[D.y == lab_], bins=bins, color=c, alpha=.45, density=True, label=nm)
for n in FOCUS: ax.axvline(D[D.name == n].tone_db.iloc[0], color=NOR, ls=":", lw=1); ax.text(D[D.name == n].tone_db.iloc[0], ax.get_ylim()[1] * .9, short(n), fontsize=8, color=NOR, rotation=90)
ax.set_xlabel("배경 8–12.5 kHz 봉우리 돌출 (dB)"); ax.legend(fontsize=8.5); ax.set_title("돌출 분포 (밀도). 지속음은 몇 개 파일의 특징인가", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림2_지속음_전수.png"), dpi=110); plt.close(fig)

# %% 숫자
for thr in (6, 10, 15):
    P(f"돌출 ≥ {thr} dB: 정상 {int(((tr.y == 0) & (tr.tone_db >= thr)).sum())}/120 · 고장 {int(((tr.y == 1) & (tr.tone_db >= thr)).sum())}/30 · val {int((D[D.y == 2].tone_db >= thr).sum())}/40")
P(tr[tr.name.isin(FOCUS + ["abn_002"] + hi_abn)].round(2).to_string(index=False))
open(os.path.join(HERE, "h10_지속음_log.txt"), "w", encoding="utf-8").write("\n".join(out))
