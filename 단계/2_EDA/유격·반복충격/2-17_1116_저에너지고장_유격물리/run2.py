# %% [markdown]
# 2-17 2차 — 1차(run.py)의 방법 문제를 고친다
# 1차에서 본 것
#   - 2초 = 하중 주기 2–4번, 파일당 충격 3–5개 → 파일 하나로 위상 잠금을 검정할 힘이 없다 (z > 3: 정상 5/91 · 고장 2/28 = 우연 수준)
#   - 하중 사건 하나가 충격 "다발"을 만든다 (A009: 봉우리 하나에 12개) → 10 ms 사건을 독립 타격으로 세면 반복·고름이 부풀려진다
#   - 하중 주파수 f0가 두 무리: 1–3 Hz(차체 출렁임) · 10–20 Hz(바퀴 튐, 스프링 아래 질량 공진)
# 2차 변경 (모두 라벨 보기 전에 정의)
#   1) 다발 = 하중 주기 T0 = 1/f0의 1/4 안에 이어지는 충격들 (하중 사건 하나 = 반 주기 안의 한 쪽 극값 근처)
#   2) 충격/하중 비 Q = 다발 최대 |h| (g) / 그 순간 국소 하중 (g). 국소 하중 = 20 Hz 저역 신호의 |값| 최대, 다발 ±T0/4
#      σ(바탕 잡음)가 아니라 원인(하중)으로 나눈 무차원 비 = "하중 1 g당 얼마나 세게 쳤나" (전달률과 같은 꼴)
#      간극 타격: 작은 하중에도 큰 타격 → Q 큼. 노면 충격: 입력이 클수록 타격과 하중이 함께 큼 → Q가 일정 수준
#   3) 위상은 파일 단위 검정을 버리고 무리(정상/고장 × 하중 무리)로 모아서 본다 (다발 단위, Rayleigh)
#   하중 무리 = f0 < 6 Hz (차체) / ≥ 6 Hz (바퀴 튐). 6 Hz는 두 물리 대역(1–3, 10–20 Hz) 사이 → 라벨 무관
# 실행: conda run -n qe-phm python "단계/2_EDA/유격·반복충격/2-17_1116_저에너지고장_유격물리/run2.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from sklearn.metrics import roc_auc_score
from src.exp import load_train
from src.features import feature_table_knock
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs2"); os.makedirs(FIG, exist_ok=True)
FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"

X, y, names = load_train(); y = np.asarray(y); N = len(y)
short = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]
lab = lambda s: s.replace("abn_", "A").replace("nor_", "N")
log = []
def say(*a):
    s = " ".join(str(v) for v in a); print(s); log.append(s)

HP = butter(4, 200, btype="highpass", fs=FS, output="sos")
LP = butter(4, 20, btype="lowpass", fs=FS, output="sos")
GAP = int(0.010 * FS)

def detect(h):
    s0 = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    m = np.ones(len(h), bool)
    for p in np.flatnonzero(np.abs(h) > 8 * s0):
        m[max(p - GAP, 0):p + GAP] = False
    hb = h[m] if m.sum() > len(h) // 4 else h
    sig = 1.4826 * np.median(np.abs(hb - np.median(hb))) + 1e-12
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    if not len(idx):
        return sig, np.array([], int)
    gs = np.split(idx, np.flatnonzero(np.diff(idx) > GAP) + 1)
    return sig, np.array([g[np.argmax(np.abs(h[g]))] for g in gs], int)

def f0_of(xc):
    sp = np.abs(np.fft.rfft(xc * np.hanning(len(xc)), n=4 * len(xc))) ** 2
    fq = np.fft.rfftfreq(4 * len(xc), 1 / FS)
    b = (fq >= 1) & (fq <= 20)
    return float(fq[b][np.argmax(sp[b])])

# %% 파일별
rows, bur = [], []
for i in range(N):
    xc = X[i] - X[i].mean()
    h = sosfiltfilt(HP, xc); lo = sosfiltfilt(LP, xc)
    sig, pk = detect(h)
    f0 = f0_of(xc); T4 = int(FS / f0 / 4)
    ld = sosfiltfilt(butter(2, [max(f0 * .6, .5), f0 * 1.4], btype="bandpass", fs=FS, output="sos"), xc)
    ph = np.angle(hilbert(ld))
    # 다발
    groups = np.split(pk, np.flatnonzero(np.diff(pk) > T4) + 1) if len(pk) else []
    B = []
    for g in groups:
        p = g[np.argmax(np.abs(h[g]))]
        a, b = max(p - T4, 0), min(p + T4, len(lo))
        load = np.abs(lo[a:b]).max()
        B.append(dict(file=short[i], y=int(y[i]), t=p / FS, n_in=len(g), peak_g=abs(h[p]), snr=abs(h[p]) / sig,
                      load_g=load, Q=abs(h[p]) / (load + 1e-6), phase=np.degrees(ph[p])))
    bur += B
    Qs = np.sort([b["Q"] for b in B])[::-1]
    pg = np.sort([b["peak_g"] for b in B])[::-1]
    rows.append(dict(file=short[i], y=int(y[i]), f0=f0, regime="바퀴" if f0 >= 6 else "차체", sigma_mg=sig * 1e3,
                     load_rms_mg=lo.std() * 1e3, n_imp=len(pk), n_burst=len(B),
                     Q1=np.log10(Qs[0]) if len(Qs) else np.nan, Q2=np.log10(Qs[1]) if len(Qs) > 1 else np.nan,
                     P2_ratio=pg[1] / pg[0] if len(pg) > 1 else 0.0))
F = pd.DataFrame(rows); Bd = pd.DataFrame(bur)
K = feature_table_knock(X); F = pd.concat([F, K], axis=1)
F["quiet"] = False; F.loc[np.argsort(F.sigma_mg.values)[:40], "quiet"] = True
F.to_csv(os.path.join(FIG, "files.csv"), index=False, encoding="utf-8-sig")
Bd.to_csv(os.path.join(FIG, "bursts.csv"), index=False, encoding="utf-8-sig")

# %% 1. 하중 무리
say("== 하중 무리 (f0 < 6 Hz 차체 / ≥ 6 Hz 바퀴 튐) ==")
say(pd.crosstab([F.quiet, F.regime], F.y).to_string())
say("f0 히스토그램 (Hz 칸: 정상/고장)")
for lo_, hi_ in [(1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 15), (15, 21)]:
    s = (F.f0 >= lo_) & (F.f0 < hi_)
    say(f"  {lo_:>2}–{hi_:<2}: {int((s & (F.y == 0)).sum()):>3} / {int((s & (F.y == 1)).sum()):>2}")
say("다발 수: 정상 중앙 {:.0f} · 고장 중앙 {:.0f}; 충격 수/다발 수 = 정상 {:.2f} · 고장 {:.2f}".format(
    F.n_burst[F.y == 0].median(), F.n_burst[F.y == 1].median(),
    F.n_imp[F.y == 0].sum() / max(F.n_burst[F.y == 0].sum(), 1), F.n_imp[F.y == 1].sum() / max(F.n_burst[F.y == 1].sum(), 1)))

# %% 2. 위상 (무리로 모아서, 다발 단위)
def rayleigh(d):
    z = np.exp(1j * np.radians(d)); R = abs(z.mean()); n = len(d)
    return n, R, n * R ** 2, np.degrees(np.angle(z.mean()))
say("\n== 다발 위상 (무리로 모음). 0° = 하중 최대, ±180° = 최소, ±90° = 반전. z > 3 ≈ p < .05 ==")
for reg in ["차체", "바퀴"]:
    for q, qn in [(True, "조용한"), (None, "전체")]:
        for g, gn in [(0, "정상"), (1, "고장")]:
            fs_ = F[(F.regime == reg) & (F.y == g) & ((F.quiet) if q else True)].file
            d = Bd[Bd.file.isin(fs_)].phase.values
            if len(d) >= 3:
                n, R, z, mu = rayleigh(d)
                # 극값 근처(|cos φ| > 0.7) 비율: 무관이면 0.5
                ext = (np.abs(np.cos(np.radians(d))) > np.cos(np.radians(45))).mean()
                say(f"  {reg} · {qn:3} · {gn}: 다발 {n:3} · R {R:.2f} · z {z:5.1f} · 평균 위상 {mu:6.0f}° · 극값 ±45° 안 {ext:.2f} (무관 0.50)")

# %% 3. 충격/하중 비 Q vs 기존 K
say("\n== 파일 축 AUC (150개, 정의 고정 후 평가만) ==")
for c, nm in [("Q1", "Q1 최대 충격/하중"), ("Q2", "Q2 두 번째 다발 충격/하중"), ("P2_ratio", "다발 2위/1위 (g)"),
              ("K1_second_snr", "K1 (현행)"), ("K2_load_to_hf", "K2 (현행)"), ("K3_top2_ratio", "K3 (현행)")]:
    for q, qn in [(slice(None), "전체"), (F.quiet.values, "조용한40")]:
        d = F[q] if not isinstance(q, slice) else F
        v = d[c].fillna(d[c].min() - 1 if d[c].notna().any() else 0)
        say(f"  {nm:22} {qn:6}: AUC {roc_auc_score(d.y, v):.3f}")
say("\n== 조용한 40개 (Q2 순) ==")
cols = ["file", "y", "regime", "f0", "sigma_mg", "load_rms_mg", "n_imp", "n_burst", "Q1", "Q2", "P2_ratio", "K1_second_snr", "K3_top2_ratio"]
say(F[F.quiet].sort_values("Q2", ascending=False)[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
nq = F[(F.y == 0)].Q2
say(f"\n정상 120개 Q2 분위: 50% {nq.quantile(.5):.2f} · 95% {nq.quantile(.95):.2f} · 최대 {nq.max():.2f} (NaN {int(nq.isna().sum())})")
aq = F[F.y == 1]
say("고장 Q2 > 정상 95%: " + " ".join(lab(f) for f in aq[aq.Q2 > nq.quantile(.95)].file))

# %% 그림: 다발 단위 충격(g) vs 국소 하중(g)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
for ax, reg in zip(axs, ["차체", "바퀴"]):
    for g, c in [(0, NOR), (1, ABN)]:
        fs_ = F[(F.regime == reg) & (F.y == g)].file
        d = Bd[Bd.file.isin(fs_)]
        q = d.file.isin(F.file[F.quiet])
        ax.scatter(d.load_g[~q], d.peak_g[~q], s=10, c=c, alpha=.4)
        ax.scatter(d.load_g[q], d.peak_g[q], s=30, c=c, edgecolors="k", alpha=.9)
    for f in ["abn_014", "abn_017", "abn_022", "abn_024", "abn_009", "nor_054", "nor_086"]:
        d = Bd[(Bd.file == f)]
        if len(d) and F.set_index("file").loc[f, "regime"] == reg:
            for _, r in d.iterrows():
                ax.annotate(lab(f), (r.load_g, r.peak_g), fontsize=7)
    xx = np.logspace(-4.5, 0.5, 10)
    for k in [1, 10, 100]:
        ax.plot(xx, k * xx, color="0.7", lw=.8, ls=":"); ax.text(xx[-3], k * xx[-3], f"Q={k}", fontsize=8, color="0.5")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("다발 순간 국소 하중 (g, 20 Hz 저역 |최대|)")
    ax.set_ylabel("다발 최대 충격 |h| (g)"); ax.set_title(f"{reg} 무리 — 점 = 다발, 테두리 = 조용한 40개")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_충격대하중.png"), dpi=80); plt.close(fig)

with open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
