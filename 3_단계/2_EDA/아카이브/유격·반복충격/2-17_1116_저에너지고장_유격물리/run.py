# %% [markdown]
# 2-17 저에너지 고장 다시 보기 — 간극(유격) 타격의 물리 지문 (1차)
# 출발: 2-12와 같은 "조용한 40개"(충격을 뺀 고역 바탕 σ 하위 40개, 라벨 없이 고정)
# 원칙 (사용자 11-16): 이 데이터에만 맞는 규칙이 아니라 물리적으로 성립하는 규칙
#   - 정의는 라벨을 보기 전에 고정한다. 문턱은 라벨로 고르지 않고 "관계가 없을 때의 기대값"(귀무 분포)과 비교한다
#   - 라벨은 색칠·집계에만 쓴다
# 물리 모델: 간극이 있는 연결부는 하중(힘)의 방향이 바뀔 때 부품이 간극을 가로질러 반대 벽을 친다
#   G1 위상 잠금   : 타격이 하중 진동의 특정 위상에 몰린다 → 위상 벡터 강도 R, Rayleigh z = nR² (무관하면 z ~ 1, z > 3 ≈ p < 0.05)
#                    어느 위상인지는 가정하지 않는다 (간극 크기에 따라 반전점 뒤 지연이 다르므로)
#   G2 하중 독립성 : 간극 타격의 크기는 간극을 닫는 속도로 정해지고 하중 크기와 약하게만 관련 → 충격 높이 vs 국소 하중의 순위 상관
#                    노면 충격은 입력이 크면 암도 크게 출렁여 양의 상관을 기대
#   G3 크기 고름   : 같은 간극·질량 → 상위 5개 충격 높이의 log10 표준편차가 작다
# 하중 = 저주파 가속도(관성력의 대리값). 파일마다 1–20 Hz에서 가장 큰 주파수 f0를 찾고 f0 ±40% 대역통과 → 힐베르트 위상·포락선
# 충격 = 200 Hz 고역통과 |h| > 8σ, 10 ms 안은 한 사건 (2-12와 같음). 단 σ는 충격 ±10 ms를 가린 뒤 다시 잰다
# 실행: conda run -n qe-phm python "3_단계/2_EDA/아카이브/유격·반복충격/2-17_1116_저에너지고장_유격물리/run.py"

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
from scipy.stats import spearmanr
from src.exp import load_train
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"

X, y, names = load_train(); y = np.asarray(y); N = len(y)
short = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]
lab = lambda i: short[i].replace("abn_", "A").replace("nor_", "N")
log = []
def say(*a):
    s = " ".join(str(v) for v in a); print(s); log.append(s)

# %% 파일별 처리 (라벨 무관)
HP = butter(4, 200, btype="highpass", fs=FS, output="sos")
GAP = int(0.010 * FS)

def detect(h):
    """충격 = |h| > 8σ, 10 ms 안은 한 사건. σ는 충격 ±10 ms를 가린 바탕에서 다시 잼"""
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

def load_band(xc):
    """1–20 Hz에서 가장 큰 주파수 f0 → f0 ±40% 대역통과 하중 신호, 위상, 포락선"""
    sp = np.abs(np.fft.rfft(xc * np.hanning(len(xc)), n=4 * len(xc))) ** 2
    fq = np.fft.rfftfreq(4 * len(xc), 1 / FS)
    b = (fq >= 1) & (fq <= 20)
    f0 = float(fq[b][np.argmax(sp[b])])
    lo, hi = max(f0 * 0.6, 0.5), f0 * 1.4
    ld = sosfiltfilt(butter(2, [lo, hi], btype="bandpass", fs=FS, output="sos"), xc)
    a = hilbert(ld)
    return f0, ld, np.angle(a), np.abs(a)

rows, imp = [], []
CACHE = {}
for i in range(N):
    xc = X[i] - X[i].mean()
    h = sosfiltfilt(HP, xc)
    sig, pk = detect(h)
    f0, ld, ph, env = load_band(xc)
    edge = int(0.5 / f0 * FS)                              # 대역통과 위상이 불안정한 양 끝 반 주기 제외
    pk_in = pk[(pk > edge) & (pk < len(xc) - edge)]
    hg = np.abs(h[pk_in]) / sig
    Lr = env[pk_in] / (np.median(env) + 1e-12)               # 국소 하중 / 파일 중앙 하중
    n = len(pk_in)
    R = float(np.abs(np.exp(1j * ph[pk_in]).mean())) if n else np.nan
    mu = float(np.angle(np.exp(1j * ph[pk_in]).mean())) if n else np.nan
    top = np.sort(hg)[::-1][:5]
    r = dict(file=short[i], y=int(y[i]), sigma_mg=sig * 1000, load_mg=ld.std() * 1000, f0=f0, n_imp=n,
             G1_R=R, G1_z=n * R ** 2 if n else np.nan, G1_mu_deg=np.degrees(mu) if n else np.nan,
             G2_rho=float(spearmanr(hg, Lr)[0]) if n >= 4 else np.nan,
             G2_lowload=float((Lr < 1).mean()) if n else np.nan,
             G3_spread=float(np.log10(top).std()) if len(top) >= 3 else np.nan)
    rows.append(r)
    for p, a, l in zip(pk_in, hg, Lr):
        imp.append(dict(file=short[i], y=int(y[i]), t=p / FS, height=a, height_g=abs(h[p]),
                        phase_deg=np.degrees(ph[p]), load_rel=l))
    CACHE[i] = (h, sig, ld, pk_in)
F = pd.DataFrame(rows); I = pd.DataFrame(imp)
F.to_csv(os.path.join(FIG, "files.csv"), index=False, encoding="utf-8-sig")
I.to_csv(os.path.join(FIG, "impacts.csv"), index=False, encoding="utf-8-sig")

# 조용한 40개 (σ 하위 40, 라벨 무관)
quiet = np.argsort(F.sigma_mg.values)[:40]
F["quiet"] = False; F.loc[quiet, "quiet"] = True
say(f"[조용한 40개] 고장 {F.y[quiet].sum()} · 정상 {40 - F.y[quiet].sum()}, σ 상한 {F.sigma_mg.values[quiet].max():.2f} mg")

# %% 요약
def summ(df, tag):
    say(f"\n== {tag} ==")
    for g, nm in [(0, "정상"), (1, "고장")]:
        d = df[df.y == g]
        say(f"{nm} {len(d)}: f0 중앙 {d.f0.median():.1f} Hz · 충격 0개 {int((d.n_imp == 0).sum())} · 충격 ≥4 {int((d.n_imp >= 4).sum())}")
        say(f"   G1 z>3 (위상 잠금, p<.05): {int((d.G1_z > 3).sum())}/{int(d.G1_z.notna().sum())}"
            f" · G1_R 중앙 {d.G1_R.median():.2f}")
        say(f"   G2 ρ 중앙 {d.G2_rho.median():.2f} (n≥4: {int(d.G2_rho.notna().sum())}) · 저하중 타격 비율 중앙 {d.G2_lowload.median():.2f}")
        say(f"   G3 spread 중앙 {d.G3_spread.median():.3f}")
summ(F, "150개 전체"); summ(F[F.quiet], "조용한 40개")

say("\n== 조용한 40개 파일별 (σ 순) ==")
cols = ["file", "y", "sigma_mg", "load_mg", "f0", "n_imp", "G1_R", "G1_z", "G1_mu_deg", "G2_rho", "G2_lowload", "G3_spread"]
say(F.loc[quiet, cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# %% 그림 1: 조용한 40개 — 하중 신호와 충격 위치
fig, axs = plt.subplots(10, 4, figsize=(20, 26), sharex=True)
t = np.arange(X.shape[1]) / FS
for ax, i in zip(axs.T.ravel(), quiet):
    h, sig, ld, pk = CACHE[i]
    c = ABN if y[i] else NOR
    sc = np.abs(h).max() + 1e-12
    ax.plot(t, h / sc, color="0.75", lw=0.3)
    ax.plot(t, ld / (np.abs(ld).max() + 1e-12), color=c, lw=1.2)
    ax.axhline(0, color="k", lw=0.4)
    if len(pk):
        ax.scatter(pk / FS, ld[pk] / (np.abs(ld).max() + 1e-12), s=18 + 4 * np.abs(h[pk]) / sig / 8,
                   color="k", zorder=3)
    r = F.iloc[i]
    ax.set_title(f"{lab(i)}  σ {r.sigma_mg:.1f} mg · f0 {r.f0:.1f} Hz · 충격 {r.n_imp} · z {r.G1_z:.1f}",
                 fontsize=9, color=c)
    ax.set_yticks([])
fig.suptitle("조용한 40개 (σ 오름차순, 열 우선) — 색 선 = 하중 신호(f0 ±40%, 최대 1로), 회색 = 200 Hz 고역, 점 = 충격(크기 ∝ 높이)", y=0.995)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_조용한40_하중과충격.png"), dpi=70); plt.close(fig)

# %% 그림 2: 충격 위상 분포 (조용한 40 / 전체, 정상 vs 고장)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
bins = np.linspace(-180, 180, 25)
for ax, sel, tt in [(axs[0], I.file.isin(F.file[F.quiet]), "조용한 40개"), (axs[1], np.ones(len(I), bool), "150개 전체")]:
    for g, c, nm in [(0, NOR, "정상"), (1, ABN, "고장")]:
        d = I[sel & (I.y == g)]
        ax.hist(d.phase_deg, bins, density=True, histtype="step", lw=2, color=c, label=f"{nm} 충격 {len(d)}")
    ax.set_title(f"{tt}: 충격 시점의 하중 위상 (0° = 하중 최대, ±90° = 반전(0점))")
    ax.set_xlabel("위상 (도)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_충격위상분포.png"), dpi=80); plt.close(fig)

# %% 그림 3: 파일 지도 G1 z × G2 ρ, 그리고 G3
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (xc_, yc_, xl, yl) in zip(axs, [("G1_z", "G2_rho", "G1 위상 잠금 Rayleigh z (log)", "G2 높이–하중 순위 상관 ρ"),
                                        ("G1_z", "G3_spread", "G1 위상 잠금 Rayleigh z (log)", "G3 상위 5 높이 log 표준편차")]):
    for g, c in [(0, NOR), (1, ABN)]:
        d = F[F.y == g]
        ax.scatter(np.clip(d[xc_], 0.05, None), d[yc_], s=np.where(d.quiet, 45, 12), c=c,
                   edgecolors=np.where(d.quiet, "k", "none"), alpha=0.8)
    for _, r in F[F.quiet & F.G1_z.notna()].iterrows():
        ax.annotate(r.file.replace("abn_", "A").replace("nor_", "N"), (max(r[xc_], 0.05), r[yc_]), fontsize=7)
    ax.axvline(3, color="0.5", ls="--"); ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel(yl)
axs[0].axhline(0, color="0.5", ls="--")
axs[0].set_title("테두리 = 조용한 40개. 점선 z = 3 (위상 무관이면 넘을 확률 5%)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_파일지도.png"), dpi=80); plt.close(fig)

with open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
