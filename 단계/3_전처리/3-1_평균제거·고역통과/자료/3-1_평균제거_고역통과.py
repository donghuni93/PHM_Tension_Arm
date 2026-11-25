# %% [markdown]
# 3-1 평균 제거와 200 Hz 고역통과 — 전처리가 신호를 어떻게 바꾸나
# - EXP-012(스펙트럼 AE)의 전처리 두 단계를 하나씩 적용해 그림으로 본다
# - 라벨은 그림 색과 설명용 AUC에만 쓴다 (차단 주파수 선택에 쓰지 않음)
# 실행: conda run -n qe-phm python "단계/3_전처리/3-1_평균제거·고역통과/자료/3-1_평균제거_고역통과.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, next(_p for _p in (os.path.abspath(os.path.join(HERE, *[".."] * _k)) for _k in range(8)) if os.path.isdir(os.path.join(_p, "src"))))
from src.exp import load_train
from src import viz  # 한글 폰트 설정
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, sosfreqz, welch
from sklearn.metrics import roc_auc_score

FS, FC = 25000, 200
NOR, ABN = "#2a78d6", "#eb6834"
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train()
nor = y == 0
SOS = butter(4, FC, btype="highpass", fs=FS, output="sos")
t = np.arange(X.shape[1]) / FS
log = []

Xc = X - X.mean(1, keepdims=True)                 # 1단계: 평균 제거
H = sosfiltfilt(SOS, Xc, axis=-1)                 # 2단계: 200 Hz 고역통과 (양방향)
H_raw = sosfiltfilt(SOS, X, axis=-1)              # 비교: 평균 제거 없이 바로 고역통과

# %% 수치 1: 평균 제거
m = X.mean(1)
log.append("[평균 제거]")
log.append(f"파일 평균: 정상 중앙 {np.median(m[nor]):.4f} g, |평균|>0.1 g 파일: " +
           ", ".join(f"{names[i]}({m[i]:.3f})" for i in np.flatnonzero(np.abs(m) > 0.1)))
rms_raw, rms_c = np.sqrt((X ** 2).mean(1)), Xc.std(1)
i97 = names.index("nor_097")
log.append(f"nor_097: 원신호 RMS {rms_raw[i97]:.3f} g → 평균 제거 후 {rms_c[i97]:.3f} g "
           f"(평균 제거 없이 RMS를 쓰면 정상 중 {100*(rms_raw[nor] < rms_raw[i97]).mean():.0f}%ile, 제거 후 {100*(rms_c[nor] < rms_c[i97]).mean():.0f}%ile)")
edge = int(0.05 * FS)
d = np.abs(H_raw - H)
log.append(f"평균 제거 없이 고역통과한 결과와의 차이 (평균 -1 g 파일들): 양 끝 50 ms 최대 "
           + ", ".join(f"{n} {max(d[names.index(n), :edge].max(), d[names.index(n), -edge:].max()):.2e} g"
                       for n in ["nor_097", "abn_016"])
           + f", 가운데 최대 {d[:, edge:-edge].max():.2e} g")

# %% 수치 2: 필터 특성
w, h = sosfreqz(SOS, worN=2 ** 16, fs=FS)
gain = np.abs(h) ** 2                             # 양방향 → 진폭 이득이 제곱, 위상 0
log.append("\n[200 Hz 고역통과: 양방향 Butterworth 4차의 진폭 이득]")
for fq in [6, 20, 50, 100, 150, 200, 250, 300, 500, 1000]:
    log.append(f"  {fq:>5} Hz: ×{np.interp(fq, w, gain):.4f}")

# %% 수치 3: 200 Hz 아래에 에너지가 얼마나 있나 (파일별)
f, P = welch(Xc, fs=FS, nperseg=8192)             # 3.05 Hz 간격
tot = P.sum(1)
def share(lo, hi): return P[:, (f >= lo) & (f < hi)].sum(1) / tot
s20, s200 = share(0, 20), share(0, FC)
log.append("\n[파일 분산 중 200 Hz 아래 비율] 중앙값 [5–95%]")
for lab, msk in [("정상", nor), ("고장", ~nor)]:
    log.append(f"  {lab}: 0–20 Hz {np.median(s20[msk]):.2f} [{np.percentile(s20[msk],5):.2f}–{np.percentile(s20[msk],95):.2f}]"
               f" · 0–200 Hz {np.median(s200[msk]):.2f} [{np.percentile(s200[msk],5):.2f}–{np.percentile(s200[msk],95):.2f}]")
log.append(f"  0–200 Hz가 분산의 절반 이상인 파일: 정상 {(s200[nor] >= .5).sum()}/120, 고장 {(s200[~nor] >= .5).sum()}/30")
std_h = H.std(1)
log.append(f"  고역통과 후 표준편차 / 전: 정상 중앙 {np.median(std_h[nor] / rms_c[nor]):.2f}, 고장 중앙 {np.median(std_h[~nor] / rms_c[~nor]):.2f}")
for n in ["abn_017", "abn_022", "abn_005", "nor_021", "nor_097"]:
    i = names.index(n); log.append(f"  {n}: 0–20 Hz {s20[i]:.2f}, 0–200 Hz {s200[i]:.2f}, 표준편차 {rms_c[i]:.4f} → {std_h[i]:.4f}")

# 저주파 우세 주파수 (1–50 Hz 최대 피크)
lo = (f >= 1) & (f <= 50)
pk = f[lo][P[:, lo].argmax(1)]
vals, cnt = np.unique(np.round(pk, 1), return_counts=True)
top = np.argsort(cnt)[::-1][:4]
log.append("  1–50 Hz 최대 피크 주파수 (파일 수): " + ", ".join(f"{vals[k]} Hz({cnt[k]})" for k in top))

# 설명용 AUC (선택에 쓰지 않음): 대역별 log 에너지
log.append("\n[설명용 AUC: 대역 log 에너지, 0.5 = 구분 못 함]")
for a, b in [(0, 20), (20, 200), (200, 12500)]:
    v = np.log10(P[:, (f >= a) & (f < b)].sum(1))
    log.append(f"  {a}–{b} Hz: AUC {roc_auc_score(y, v):.3f}, 정상 5% 미만 고장 {(v[~nor] < np.percentile(v[nor], 5)).sum()}, 95% 초과 고장 {(v[~nor] > np.percentile(v[nor], 95)).sum()}")

open(os.path.join(FIG, "3-1_log.txt"), "w", encoding="utf-8").write("\n".join(log))

# %% 그림 1: 파형 — 원신호 → 평균 제거 → 고역통과
SHOW = [("nor_097", "평균 -1 g 정상"), ("nor_021", "느린 흔들림 정상"),
        ("abn_022", "느린 흔들림 + 작은 충격 고장"), ("abn_005", "충격 고장")]
fig, axs = plt.subplots(len(SHOW), 3, figsize=(17, 2.6 * len(SHOW)), sharex=True)
for r, (n, desc) in enumerate(SHOW):
    i = names.index(n); col = NOR if y[i] == 0 else ABN
    for c, (sig, ttl) in enumerate([(X[i], "원신호"), (Xc[i], "① 평균 제거"), (H[i], "② + 200 Hz 고역통과")]):
        a = axs[r, c]; a.plot(t, sig, color=col, lw=.4)
        a.axhline(0, color="k", lw=.5, ls=":")
        a.set_title(f"{n} ({desc}) — {ttl}\n평균 {sig.mean():+.3f} g · 표준편차 {sig.std():.4f} g", fontsize=9)
        if c > 0:
            lim = np.abs(Xc[i]).max() * 1.05; a.set_ylim(-lim, lim)   # ①·② 같은 y축 → 무엇이 빠졌는지 비교
        a.grid(alpha=.3)
    axs[r, 0].set_ylabel("g")
for a in axs[-1]: a.set_xlabel("초")
fig.suptitle("전처리 단계별 파형 (①·② 같은 y축)", fontsize=11); fig.tight_layout()
fig.savefig(os.path.join(FIG, "3-1_waves.png"), dpi=110); plt.close(fig)

# %% 그림 2: 필터 이득 + PSD 전후
_, Ph = welch(H, fs=FS, nperseg=8192)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
a1.semilogx(w[1:], gain[1:], color="k")
for fq in [20, 100, 200, 300]:
    g = np.interp(fq, w, gain); a1.plot(fq, g, "o", color=ABN); a1.annotate(f"{fq} Hz ×{g:.2f}", (fq, g), xytext=(5, -12), textcoords="offset points", fontsize=8)
a1.axvline(FC, color=ABN, ls="--", lw=.8)
a1.set_xlim(1, FS / 2); a1.set_xlabel("Hz (로그 축)"); a1.set_ylabel("에너지 이득 (1 = 그대로)")
a1.set_title("양방향 Butterworth 4차 200 Hz 고역통과: 주파수별로 얼마나 남기나", fontsize=10); a1.grid(alpha=.3, which="both")
k = f > 0
for PP, ls, lab in [(P, "-", "전"), (Ph, "--", "후")]:
    L = np.log10(PP[:, k] + 1e-20)
    a2.fill_between(f[k], *np.percentile(L[nor], [5, 95], 0), color=NOR, alpha=.15 if ls == "-" else .08)
    a2.semilogx(f[k], np.median(L[nor], 0), color=NOR, ls=ls, label=f"정상 중앙값 ({lab})")
    a2.semilogx(f[k], np.median(L[~nor], 0), color=ABN, ls=ls, label=f"고장 중앙값 ({lab})")
a2.axvline(FC, color=ABN, ls="--", lw=.8)
a2.set_xlim(1, FS / 2); a2.set_ylim(-13, -2); a2.set_xlabel("Hz (로그 축)"); a2.set_ylabel("log10 PSD")
a2.set_title("고역통과 전(실선)·후(점선) PSD. 띠 = 정상 5–95%", fontsize=10); a2.legend(fontsize=8); a2.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-1_filter_psd.png"), dpi=110); plt.close(fig)

# %% 그림 3: 파일별 저주파 비율
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
rj = np.random.default_rng(0)
for a, (v, ttl) in zip(axs, [(s20, "0–20 Hz 비율"), (s200, "0–200 Hz 비율 (= 고역통과가 지우는 몫)")]):
    for k_, (msk, col, lab) in enumerate([(nor, NOR, "정상"), (~nor, ABN, "고장")]):
        a.scatter(k_ + rj.uniform(-.2, .2, msk.sum()), v[msk], s=14, color=col, alpha=.7)
    for n in ["abn_017", "abn_022", "abn_005", "nor_021", "nor_097"]:
        i = names.index(n); a.annotate(n, (int(y[i]) + .22, v[i]), fontsize=7, va="center")
    a.set_xticks([0, 1]); a.set_xticklabels(["정상 120", "고장 30"]); a.set_ylim(-.03, 1.03)
    a.set_ylabel("파일 분산 중 비율"); a.set_title(ttl, fontsize=10); a.grid(alpha=.3, axis="y")
fig.suptitle("파일 에너지의 몇 %가 저주파에 있나", fontsize=11); fig.tight_layout()
fig.savefig(os.path.join(FIG, "3-1_lowfrac.png"), dpi=110); plt.close(fig)
print("done")
