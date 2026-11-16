# %% [markdown]
# 2-12 유격 반복 · 하중 대비 고주파 — 정상 중심에서 먼 파일이 고장인가
# - 출발: EXP-021이 늘 놓치는 014·017 (신호가 거의 없는 고장)
# - 두 축 (도메인 가설로 먼저 정의, 라벨 무관)
#   D1 유격 반복       = log10(두 번째로 큰 사건 높이 / σ)  — 유격이 있으면 하중이 걸릴 때마다 큰 타격이 반복 (H1)
#   D6 하중 대비 고주파 = log10(저주파(<20 Hz) RMS / σ)   — 암의 출렁임(하중)이 고주파 바탕의 몇 배인가
#   σ = 200 Hz 고역통과 신호의 1.4826·MAD (충격에 둔감한 바탕 수준), 사건 = |h| > 8σ, 10 ms 안은 한 사건
# - 보조 축 (r 바깥에서 정상·고장을 가를 후보, 역시 라벨 무관 정의)
#   D2 미세 틱 = 5σ 사건 수/초 · D3 배경 첨도 = 8σ 사건 ±10 ms 가린 h의 첨도 · D4 사건 중심 주파수 = 상위 8사건 10 ms 창
#   R  2위/1위 사건 높이 · D8 타격 지속 = 상위 3사건에서 포락선(2 ms 평균)이 그 사건 최고점의 20% 위에 머무는 시간(ms) 중앙값
# - 라벨은 색칠·설명에만. 여기서 본 차이는 EXP(중첩 CV)로 확인 (원칙 3)
# 실행: conda run -n qe-phm python "단계/2_EDA/유의미/2-12_1114_저에너지고장/run.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import *
from src.evaluate import quick_cv
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; y = np.asarray(y)
nor, abn = y == 0, y == 1
LOG = []
def log(*a): LOG.append(" ".join(str(v) for v in a)); print(*a, flush=True)
def save(fig, name): fig.savefig(os.path.join(FIG, name + ".png"), dpi=80); plt.close(fig)
NOR, ABN = "#2a78d6", "#eb6834"
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")

xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), xc, axis=-1)
LO = sosfiltfilt(butter(4, 20, btype="lowpass", fs=FS, output="sos"), xc, axis=-1)
SIG = 1.4826 * np.median(np.abs(H - np.median(H, 1, keepdims=True)), 1) + 1e-12


# %% 특징 (150개 전체, 라벨 무관)
W256 = np.hanning(256); FR256 = np.fft.rfftfreq(256, 1 / FS)
def feats(h, lo, sig):
    f = {}
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    gs = np.split(idx, np.flatnonzero(np.diff(idx) > 250) + 1) if len(idx) else []
    ev = np.array([g[np.argmax(np.abs(h[g]))] for g in gs], int)
    ev = ev[np.argsort(-np.abs(h[ev]))] if len(ev) else ev               # 큰 사건 순
    hg = np.abs(h[ev]) / sig
    f["n_events"] = len(ev)
    f["D1"] = float(np.log10(hg[1])) if len(hg) > 1 else float(np.log10(8))
    f["D6"] = float(np.log10(lo.std() / sig))
    i5 = np.flatnonzero(np.abs(h) > 5 * sig)
    f["D2"] = (len(np.split(i5, np.flatnonzero(np.diff(i5) > 250) + 1)) if len(i5) else 0) / 2.0
    m = np.ones(len(h), bool)
    for p in ev: m[max(p - 250, 0):p + 250] = False
    f["D3"] = float(np.log10(kurtosis(h[m]) + 3))
    cen = []
    for p in ev[:8]:
        s = h[max(p - 20, 0):max(p - 20, 0) + 256]
        if len(s) == 256:
            P = np.abs(np.fft.rfft(s * W256)) ** 2; cen.append((P * FR256).sum() / P.sum())
    f["D4"] = float(np.median(cen)) if cen else np.nan
    f["R"] = float(hg[1] / hg[0]) if len(hg) > 1 else 0.0
    env = np.convolve(np.abs(hilbert(h)), np.ones(50) / 50, "same")         # 2 ms 평균 포락선
    dur = []
    for p in ev[:3]:
        a, b = max(p - 2500, 0), min(p + 2500, len(h))                        # ±100 ms 안에서
        above = env[a:b] > .2 * env[p]
        k = p - a; l = r = k
        while l > 0 and above[l - 1]: l -= 1
        while r < len(above) - 1 and above[r + 1]: r += 1
        dur.append((r - l) / FS * 1000)
    f["D8"] = float(np.median(dur)) if dur else 0.0
    f["sigma_mg"] = sig * 1000; f["load_mg"] = lo.std() * 1000
    return f
F = pd.DataFrame([feats(H[i], LO[i], SIG[i]) for i in range(len(y))], index=names)
F.insert(0, "y", y)

# 정상 기준 로버스트 z (중앙값·MAD, 기술용) → 정상 중심 거리 r
def rz(v):
    med = np.median(v[nor]); mad = 1.4826 * np.median(np.abs(v[nor] - med)) + 1e-12
    return (v - med) / mad
F["z1"], F["z6"] = rz(F.D1.values), rz(F.D6.values)
F["r"] = np.sqrt((F.z1 ** 2 + F.z6 ** 2) / 2)
F.round(4).to_csv(os.path.join(FIG, "features.csv"), encoding="utf-8-sig")
AUX = ["D2", "D3", "D4", "R", "D8"]
AUXN = {"D2": "미세 틱 (5σ 사건/초)", "D3": "배경 첨도 log", "D4": "사건 중심 주파수 (Hz)", "R": "2위/1위 사건 높이",
        "D8": "타격 지속 (ms)"}


# %% 그림 1: 두 축 지도와 정상 중심 거리
log("[거리 r 구간별 고장 비율] (정상 중심 = 정상 120개 중앙값, 라벨로 잡은 기술용)")
for lo_, hi_ in ((0, 1), (1, 1.5), (1.5, 2), (2, 99)):
    s = (F.r >= lo_) & (F.r < hi_)
    log(f"  r {lo_}–{hi_ if hi_ < 99 else '∞'}: 파일 {s.sum():3d} · 고장 {(s & abn).sum():2d} ({(s & abn).sum() / s.sum() * 100:3.0f}%) · "
        f"정상 {' '.join(short(n) for n in F.index[s & nor]) if hi_ == 99 else ''}")
LEFT = F.index[(F.z6 < -2) & (F.z1 > 0)]
OUT = F.index[F.r >= 2]
fig, ax = plt.subplots(figsize=(10, 9))
for lab, s, c, mk in (("정상", nor, NOR, "s"), ("고장", abn, ABN, "o")):
    ax.scatter(F.z6[s], F.z1[s], color=c, s=28, alpha=.6, lw=0, marker=mk, label=lab)
th = np.linspace(0, 2 * np.pi, 200)
for rr in (1, 2):
    ax.plot(rr * np.sqrt(2) * np.cos(th), rr * np.sqrt(2) * np.sin(th), color="k", lw=.8, ls="--")
    ax.annotate(f"r = {rr}", (rr * np.sqrt(2) * .71, -rr * np.sqrt(2) * .71), fontsize=9)
for n in set(LEFT) | set(OUT):
    ax.annotate(short(n), (F.z6[n], F.z1[n]), xytext=(4, 3), textcoords="offset points", fontsize=8,
                color=ABN if F.y[n] else NOR)
ax.set_xlabel("D6 하중 대비 고주파 (정상 기준 z) → 오른쪽: 하중은 큰데 고주파 바탕은 조용")
ax.set_ylabel("D1 유격 반복 (정상 기준 z) → 위: 큰 타격이 두 번 이상")
ax.set_title("정상 중심 거리 r = √((z1² + z6²)/2). 라벨 = 왼쪽 무리(z6 < −2, z1 > 0)와 r ≥ 2 파일", fontsize=10)
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); save(fig, "1_map")


# %% 질문 1: 왼쪽 무리 (z6 < −2, z1 > 0) — 무엇이 D6를 낮추나 (σ가 커서인가, 하중이 없어서인가)
log("\n[왼쪽 무리] z6 < −2 이고 z1 > 0")
log(f"  정상 중앙: σ {F.sigma_mg[nor].median():.1f} mg · 하중 {F.load_mg[nor].median():.1f} mg · 사건 {F.n_events[nor].median():.0f} · 틱 {F.D2[nor].median():.1f}/s")
for n in LEFT:
    log(f"  {n}: σ {F.sigma_mg[n]:6.2f} mg · 하중 {F.load_mg[n]:6.2f} mg · 사건 {F.n_events[n]:3d} · 틱 {F.D2[n]:5.1f}/s · "
        f"D1 {F.D1[n]:.2f} · D6 {F.D6[n]:.2f} · 지속 {F.D8[n]:.1f} ms")
LN = F.index[(F.z6 < -2) & nor]
log(f"  (같은 z6 < −2의 정상 {len(LN)}개: D1 z 최대 {F.z1[LN].max():.2f} — 두 번째 큰 사건이 없음)")
show = list(LEFT) + list(F.loc[LN].sort_values("z1", ascending=False).index[:2])
tt = np.arange(X.shape[1]) / FS
fig, axs = plt.subplots(len(show), 1, figsize=(16, 1.8 * len(show)), sharex=True)
for a, n in zip(axs, show):
    i = names.index(n)
    a2 = a.twinx(); a2.plot(tt, LO[i], color="#555", lw=.8); a2.set_yticks([])
    a.plot(tt, H[i], lw=.4, color=ABN if y[i] else NOR)
    a.set_title(f"{n} · σ {F.sigma_mg[n]:.2f} mg · 하중 {F.load_mg[n]:.2f} mg · 사건 {F.n_events[n]} · 틱 {F.D2[n]:.1f}/s "
                f"(색 = 고역통과, 검은 선 = 저주파 <20 Hz, 축 따로)", fontsize=8.5, loc="left")
    a.grid(alpha=.3)
axs[-1].set_xlabel("시간 (s)")
fig.tight_layout(); save(fig, "2_left_waves")


# %% 질문 2: r ≥ 2 바깥에서 정상과 고장을 가르는 보조 축
log(f"\n[r ≥ 2] 파일 {len(OUT)}개: 고장 {int(F.y[OUT].sum())} · 정상 {int((F.y[OUT] == 0).sum())} ({' '.join(short(n) for n in OUT if F.y[n] == 0)})")
log("  보조 축별: 바깥 정상 값 → 바깥 고장 중 '정상 범위 밖'인 수 (낮은 쪽/높은 쪽 둘 다 봄) · 150개 전체 AUC")
on, oa = [n for n in OUT if F.y[n] == 0], [n for n in OUT if F.y[n] == 1]
for c in AUX:
    v = F[c].fillna(F[c].median())
    lo_, hi_ = v[on].min(), v[on].max()
    log(f"  {c} {AUXN[c]:22s} 정상 {' '.join(f'{v[n]:.2f}' for n in on)} · 고장 중 정상 최소 미만 {int((v[oa] < lo_).sum())} / "
        f"최대 초과 {int((v[oa] > hi_).sum())} (/{len(oa)}) · 전체 AUC {roc_auc_score(y, v):.2f}")
log("  바깥 파일별 값:")
log(F.loc[OUT, ["y", "r", "D1", "D6"] + AUX].round(2).sort_values("y").to_string())

fig, axs = plt.subplots(1, len(AUX), figsize=(18, 5))
for a, c in zip(axs, AUX):
    for k, (grp, col) in enumerate(((on, NOR), (oa, ABN))):
        vv = F.loc[grp, c].values
        a.scatter(np.full(len(vv), k) + np.linspace(-.12, .12, len(vv)), vv, color=col, s=36, alpha=.8, lw=0)
        for j, n in enumerate(grp):
            a.annotate(short(n), (k + .16, vv[j]), fontsize=7, color=col)
    a.set_xticks([0, 1]); a.set_xticklabels(["바깥 정상", "바깥 고장"]); a.set_xlim(-.5, 1.7)
    a.set_title(AUXN[c], fontsize=9); a.grid(alpha=.3)
fig.suptitle("r ≥ 2 바깥 파일만: 보조 축으로 정상과 고장이 갈리는가", fontsize=11)
fig.tight_layout(); save(fig, "3_outer_aux")


# %% 참고: D만으로 모델 (quick_cv, 탐색 확인 — EXP 아님)
DM = F[["D1", "D6"] + AUX].fillna(F[["D1", "D6"] + AUX].median()).values
q = quick_cv(DM, y, np.arange(len(y)), make_et())
nn = q["oof"][:, nor]
risk = lambda n: float(np.mean([(nn[k] >= q["oof"][k, names.index(n)]).mean() if F.y[n] else (nn[k] < q["oof"][k, names.index(n)]).mean()
                                for k in range(len(q["oof"]))]))
log(f"\n[참고] ET(D1 D6 + 보조 5) quick_cv PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} (EXP-021 0.946)")
log("  위험도 (고장 = 위 정상 비율, 정상 = 아래 정상 비율): " + " · ".join(
    f"{short(n)} {risk(n):.2f}" for n in ["abn_014", "abn_017", "abn_022", "abn_002", "abn_029", "nor_025", "nor_055", "nor_116", "nor_009", "nor_054", "nor_086"]))

open(os.path.join(FIG, "numbers.txt"), "w", encoding="utf-8").write("\n".join(LOG))
print("done")
