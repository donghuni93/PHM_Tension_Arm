# %% 2-38 저주파 경보 후보 (11-25): 학습에 없을 수 있는 고장(스토퍼 접촉 · 반전 유격 타격 · 유체부싱 누유)용 정상 기준 OR 경보 후보
# 정의 · 방향 · 대역은 라벨을 보기 전에 물리로 고정했다(README 정의 절). 판단은 라벨 없이: 정상 안 안정성 · 하중 의존 · 정상 분위 오경보 · val 이동.
# 고장 라벨은 맨 끝 '참고'에만 쓴다(선택에 쓰지 않음).
import glob, os, sys
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import skew, spearmanr, mannwhitneyu, ks_2samp
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.exp import OUTLIER_NORMALS
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True); FS = 25000

fl, grp = [], []
for g, p in (("normal", "TeamProject_train_변경/normal"), ("abnormal", "TeamProject_train_변경/abnormal"), ("val", "TeamProject_val/val")):
    s = sorted(glob.glob(os.path.join(ROOT, "2_data", p, "*.csv"))); fl += s; grp += [g] * len(s)
names = np.array([os.path.basename(f)[:-4] for f in fl]); grp = np.array(grp)
X = np.stack([np.loadtxt(f) for f in fl]); Xc = X - X.mean(1, keepdims=True)
N, A, V = grp == "normal", grp == "abnormal", grp == "val"
OUT = np.isin(names, OUTLIER_NORMALS); NFIT = N & ~OUT          # 문턱 계산에서만 이상치 정상 3개 제외 (오경보 집계에는 포함)
sos = lambda kind, f: butter(4, f, kind, fs=FS, output="sos")
E = 2500; n_cut = X.shape[1] - 2 * E; H = n_cut // 2              # 양 끝 0.1 s 버림 → 1.8 s, 반쪽 = 0.9 s
cut = lambda a: a[..., E:-E]

# %% 공통 신호 (전체 파일에 양방향 필터 → 끝 0.1 s 버림)
lo20 = sosfiltfilt(sos("lowpass", 20), Xc, axis=1)
b500 = sosfiltfilt(sos("bandpass", [20, 500]), Xc, axis=1)
bhf = sosfiltfilt(sos("bandpass", [1600, 11000]), Xc, axis=1)
b200 = sosfiltfilt(sos("bandpass", [20, 200]), Xc, axis=1)
b30 = sosfiltfilt(sos("bandpass", [5, 30]), Xc, axis=1)
hp1k = sosfiltfilt(sos("highpass", 1000), Xc, axis=1)
DEC = 100                                                          # 포락선 50 Hz 저역 후 100개 중 1개 → 250 Hz
smooth_env = lambda b: sosfiltfilt(sos("lowpass", 50), np.abs(hilbert(b, axis=1)), axis=1)
env_lf, env_hf = cut(smooth_env(b500))[:, ::DEC], cut(smooth_env(bhf))[:, ::DEC]
lo20c, b500c, b200c, b30c = cut(lo20), cut(b500), cut(b200), cut(b30)
load = lo20c.std(1)

# %% 지표 함수 (구간 하나에 대해)
def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
def prox(s):                                                       # 스트로크 끝단 근접도 = |s − 중앙| / std
    return np.abs(s - np.median(s)) / s.std()

def randdec(y, seg):
    """랜덤 감소 서명: +1σ 상향 교차를 방아쇠로 길이 seg 구간 평균 → D/D(0). 방아쇠 수도 반환"""
    s = y.std(); i = np.where((y[:-1] < s) & (y[1:] >= s))[0] + 1; i = i[i + seg <= len(y)]
    if len(i) < 3: return None, len(i)
    D = np.mean([y[k:k + seg] for k in i], 0); return D / D[0], len(i)

def damp_delta(D, fs):
    """서명의 지배 주기 Td(영점 교차 간격 × 2)와 주기당 로그 감소 δ = −(log|극값| 기울기) × Td (무차원, ζ ≈ δ/2π)
    극값 = 0 ~ 4 Td 안의 |D| 국소 최대(첫 점 포함). 극값 < 3개면 NaN"""
    if D is None: return np.nan, np.nan
    z = np.where(np.diff(np.sign(D)) != 0)[0]
    if len(z) < 2: return np.nan, np.nan
    Td = 2 * np.mean(np.diff(z[:6])) / fs
    a = np.abs(D); m = int(min(len(D) - 1, 4 * Td * fs))
    pk = [0] + [k for k in range(1, m) if a[k] >= a[k - 1] and a[k] > a[k + 1]]
    if len(pk) < 3: return np.nan, Td
    t = np.array(pk) / fs; sl = np.polyfit(t, np.log(a[pk] + 1e-12), 1)[0]
    return float(-sl * Td), Td

RD = {"200": (b200c, 10, 0.10), "30": (b30c, 50, 0.50)}           # 대역: (신호, 솎기, 구간 길이 s). 20–200 Hz → 2.5 kHz, 5–30 Hz → 500 Hz

def indicators(sl):
    """구간 sl(잘린 신호 기준 인덱스)에서 지표 계산 → dict 배열"""
    out = {}
    out["skew_b500"] = np.abs(skew(b500c[:, sl], axis=1))
    out["skew_lo20"] = np.abs(skew(lo20c[:, sl], axis=1))
    esl = slice(sl.start // DEC if sl.start else None, sl.stop // DEC if sl.stop else None)
    out["corr_env_a"] = np.array([corr(a, b) for a, b in zip(env_lf[:, esl], env_hf[:, esl])])
    out["corr_prox_b"] = np.array([corr(prox(s[::DEC]), b) for s, b in zip(lo20c[:, sl], env_hf[:, esl])])
    for k, (sig, d, segs) in RD.items():
        r = [damp_delta(randdec(y[::d], int(segs * FS / d))[0], FS / d) for y in sig[:, sl]]
        out[f"damp_{k}"] = np.array([v[0] for v in r]); out[f"Td_{k}"] = np.array([v[1] for v in r])
        out[f"ntrig_{k}"] = np.array([randdec(y[::d], int(segs * FS / d))[1] for y in sig[:, sl]])
    return out

full = indicators(slice(None)); h1 = indicators(slice(0, H)); h2 = indicators(slice(H, 2 * H))

# %% 필터만의 감쇠: 백색잡음에 같은 필터 · 같은 절차 (200회)
rng = np.random.default_rng(0); W = rng.standard_normal((200, X.shape[1]))
noise = {}
for k, band, (_, d, segs) in (("200", [20, 200], RD["200"]), ("30", [5, 30], RD["30"])):
    w = cut(sosfiltfilt(sos("bandpass", band), W, axis=1))
    r = np.array([damp_delta(randdec(y[::d], int(segs * FS / d))[0], FS / d) for y in w])
    noise[k] = r
for k in ("200", "30"):                                            # 지표 = δ_파일 / 백색잡음 δ 중앙  (< 1 = 필터만보다 덜 감쇠)
    ref = np.nanmedian(noise[k][:, 0])                             # (첫 실행의 log 비는 δ ≤ 0 파일에서 NaN · 발산 → 선형 비로 바꿈, 라벨 보기 전 수치 문제)
    for dct in (full, h1, h2): dct[f"damp_rel_{k}"] = dct[f"damp_{k}"] / ref

# %% 기존 열: L2_hfl(H18 재계산, 2-37과 같은 식) · L1_lock S (캐시)
F = pd.DataFrame(index=names)
IND = {"skew_b500": "high", "skew_lo20": "high", "corr_env_a": "high", "corr_prox_b": "high", "damp_rel_200": "low", "damp_rel_30": "low"}
for c in IND: F[c] = full[c]
for c in ("Td_200", "Td_30", "ntrig_200", "ntrig_30", "damp_200", "damp_30"): F[c] = full[c]
F["load"] = load
F["L2_hfl"] = np.log10(np.median(np.abs(hilbert(cut(hp1k), axis=1)), 1) / load)
z = np.load(os.path.join(ROOT, "4_outputs", "cache", "train_X.npz"), allow_pickle=True)
ct = pd.read_csv(os.path.join(ROOT, "4_outputs", "cache", "feat2_train.csv")); ct.index = list(z["names"])
cv = pd.read_csv(os.path.join(ROOT, "4_outputs", "cache", "feat2_val.csv")); cv.index = [f"val_{i + 1:03d}" for i in range(len(cv))]
cc = pd.concat([ct, cv])
F["L1_lock"] = cc.reindex(names)["L1_S"].values
F["H18_cache"] = cc.reindex(names)["H18_hfl"].values
F["H1_nimp"] = cc.reindex(names)["H1_nimp"].values                # 충격 수(log1p): (a)가 광대역 충격을 재는지 확인용
F["group"] = grp; F["outlier"] = OUT
F.to_csv(os.path.join(FIG, "files.csv"), encoding="utf-8-sig")

# %% 평가 (라벨 없이)
L = ["2-38 저주파 경보 후보 — 수치 (정상 120 · val 40 · 참고 고장 30)",
     f"검증: 재계산 L2_hfl vs 캐시 H18 ρ(전체) {spearmanr(F.L2_hfl, F.H18_cache, nan_policy='omit')[0]:+.3f}"
     f", L1_lock 결측 {int(F.L1_lock.isna().sum())}개", ""]
Q = 99.5
def thr(v, mask, side):
    return np.nanpercentile(v[mask], Q) if side == "high" else np.nanpercentile(v[mask], 100 - Q)
def over(v, t, side): return (v > t) if side == "high" else (v < t)
auc = lambda v, a, b: mannwhitneyu(v[a][~np.isnan(v[a])], v[b][~np.isnan(v[b])]).statistic / ((~np.isnan(v[a])).sum() * (~np.isnan(v[b])).sum())
rows = []
nidx = np.where(N)[0]
for c, side in IND.items():
    v = F[c].values.astype(float); sgn = 1 if side == "high" else -1
    rho_h = spearmanr(h1[c][N], h2[c][N], nan_policy="omit")[0]
    rho_l = spearmanr(v[N], load[N], nan_policy="omit")[0]
    fa = []
    for seed in range(20):                                         # 정상 5-폴드 × 20번: 4/5(이상치 제외)로 문턱 → 남은 1/5 정상 중 초과 수
        cnt = 0
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(nidx):
            m = np.zeros(len(v), bool); m[nidx[tr]] = True; m &= ~OUT
            cnt += int(over(v[nidx[te]], thr(v, m, side), side).sum())
        fa.append(cnt)
    t = thr(v, NFIT, side)
    n_in = names[N & over(v, t, side)]; vv = names[V & over(v, t, side)]; ab = names[A & over(v, t, side)]
    auc_v = auc(sgn * v, V, N); ks_p = ks_2samp(v[V][~np.isnan(v[V])], v[N][~np.isnan(v[N])]).pvalue
    auc_a = auc(sgn * v, A, N)
    rows.append(dict(지표=c, 경보방향=side, 반쪽ρ=rho_h, 하중ρ=rho_l, CV오경보=np.mean(fa), CV오경보_sd=np.std(fa), 문턱=t,
                     정상초과=len(n_in), 정상초과명=" ".join(n_in), val초과=len(vv), val초과명=" ".join(vv),
                     val이동AUC=auc_v, KS_p=ks_p, 참고고장초과=" ".join(ab), 참고AUC=auc_a, NaN=int(np.isnan(v).sum())))
R = pd.DataFrame(rows)
fmt = lambda r: (f"{r.지표:13s} [{r.경보방향}] 반쪽ρ {r.반쪽ρ:+.2f} | 하중ρ {r.하중ρ:+.2f} | CV 오경보/120 {r.CV오경보:.2f}±{r.CV오경보_sd:.2f} | "
                 f"전체문턱 {r.문턱:+.3f} 정상 초과 {r.정상초과} ({r.정상초과명 or '-'}) | val 초과 {r.val초과} ({r.val초과명 or '-'}) | "
                 f"val 이동 AUC {r.val이동AUC:.2f} KS p {r.KS_p:.3f} | NaN {r.NaN}")
L += ["[라벨 없는 평가] 경보 방향은 물리로 미리 정함(high = 99.5 % 위, low = 0.5 % 아래). val 이동 AUC는 경보 방향 기준(0.5 = 이동 없음)"] + [fmt(r) for r in R.itertuples()]
L += ["", "[정상 분포] 중앙 (5–95 %)"] + [f"{c:13s} {np.nanmedian(F[c][N]):+.3f} ({np.nanpercentile(F[c][N].astype(float), 5):+.3f} – {np.nanpercentile(F[c][N].astype(float), 95):+.3f})" for c in IND]
L.append(f"corr_env_a > 0 인 정상 {np.mean(F.corr_env_a[N] > 0)*100:.0f} %, > 0.3 {np.mean(F.corr_env_a[N] > 0.3)*100:.0f} % | corr_prox_b > 0 {np.mean(F.corr_prox_b[N] > 0)*100:.0f} %, > 0.3 {np.mean(F.corr_prox_b[N] > 0.3)*100:.0f} %")
for c in ("corr_env_a", "corr_prox_b", "skew_b500"):
    L.append(f"{c} ~ H1_nimp(충격 수) ρ: 정상 {spearmanr(F[c][N], F.H1_nimp[N])[0]:+.2f}, val {spearmanr(F[c][V], F.H1_nimp[V])[0]:+.2f}")
L.append("val 초과 파일의 H1_nimp(충격 수 = e^x − 1): " + ", ".join(f"{n} {np.expm1(F.H1_nimp[n]):.0f}" for n in names[V & (F.corr_env_a.values > R.loc[R.지표 == 'corr_env_a', '문턱'].item())])
          + f" | 정상 충격 수 중앙 {np.median(np.expm1(F.H1_nimp[N])):.0f}, 95 % {np.percentile(np.expm1(F.H1_nimp[N]), 95):.0f}")
# 비교 기준: 기존 열의 반쪽 안정성
hp_c = cut(hp1k); hl = lambda s: np.log10(np.median(np.abs(hilbert(hp_c[:, s], axis=1)), 1) / lo20c[:, s].std(1))
L.append(f"비교 — 반쪽 ρ(정상): L2_hfl {spearmanr(hl(slice(0, H))[N], hl(slice(H, 2*H))[N])[0]:+.2f}, 하중(0–20 Hz RMS) {spearmanr(lo20c[N, :H].std(1), lo20c[N, H:].std(1))[0]:+.2f}, 20–200 Hz RMS {spearmanr(b200c[N, :H].std(1), b200c[N, H:].std(1))[0]:+.2f}")

# 랜덤 감소: 필터만의 감쇠와 비교
L += ["", "[랜덤 감소 서명 vs 백색잡음(같은 필터, 200회)]"]
for k, band in (("200", "20–200 Hz"), ("30", "5–30 Hz")):
    nz = noise[k]; d = F[f"damp_{k}"].values; td = F[f"Td_{k}"].values
    lo5, hi95 = np.nanpercentile(nz[:, 0], [5, 95])
    L.append(f"{band}: 백색잡음 δ 중앙 {np.nanmedian(nz[:,0]):.3f} (5–95 % {lo5:.3f}–{hi95:.3f}), Td 중앙 {np.nanmedian(nz[:,1])*1e3:.1f} ms | "
             f"정상 δ 중앙 {np.nanmedian(d[N]):.3f} (5–95 % {np.nanpercentile(d[N],5):.3f}–{np.nanpercentile(d[N],95):.3f}), Td 중앙 {np.nanmedian(td[N])*1e3:.1f} ms | "
             f"정상 중 잡음 5–95 % 안 {np.mean((d[N] >= lo5) & (d[N] <= hi95))*100:.0f} % | 방아쇠 수 중앙(전체/반쪽) {np.median(F[f'ntrig_{k}'][N]):.0f}/{np.median(h1[f'ntrig_{k}'][N]):.0f} | NaN 정상 {int(np.isnan(d[N]).sum())}")

# 상관 (정상, Spearman)
cols = list(IND) + ["L2_hfl", "L1_lock", "load"]
C = F.loc[N, cols].astype(float).corr(method="spearman")
L += ["", "[상관 — 정상 120, Spearman]", C.round(2).to_string()]
L += ["", "[참고 — 고장 라벨, 선택에 쓰지 않음] 경보 방향 AUC(고장>정상 쪽) · 전체 문턱 초과 고장"]
L += [f"{r.지표:13s} AUC {r.참고AUC:.2f} | 초과 고장: {r.참고고장초과 or '-'}" for r in R.itertuples()]
open(os.path.join(FIG, "수치.txt"), "w", encoding="utf-8").write("\n".join(L))
R.to_csv(os.path.join(FIG, "요약.csv"), index=False, encoding="utf-8-sig")

# %% 그림 1: 지표별 분포 (정상 · val · 참고 고장) + 정상 문턱
lab = {"skew_b500": "|왜도| 20–500 Hz", "skew_lo20": "|왜도| 0–20 Hz", "corr_env_a": "(a) 포락선 상관 20–500 ↔ 1.6–11 kHz",
       "corr_prox_b": "(b) 끝단 근접도 ↔ 1.6–11 kHz 포락선", "damp_rel_200": "δ/δ잡음 20–200 Hz", "damp_rel_30": "δ/δ잡음 5–30 Hz"}
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5)); rng2 = np.random.default_rng(1)
for a, r in zip(ax.ravel(), R.itertuples()):
    v = F[r.지표].values.astype(float)
    for i, (m, col, nm) in enumerate(((N & ~OUT, "#4C72B0", "정상"), (V, "#7F7F7F", "val"), (A, "#DD8452", "고장(참고)"))):
        a.scatter(i + rng2.uniform(-0.18, 0.18, m.sum()), v[m], s=12, c=col, alpha=0.7, label=nm)
    a.scatter(0 + rng2.uniform(-0.18, 0.18, OUT.sum()), v[OUT], s=40, marker="D", c="#4C72B0", edgecolor="k", label="이상치 정상 ◆")
    a.axhline(r.문턱, c="crimson", ls="--", lw=1)
    a.set_xticks([0, 1, 2]); a.set_xticklabels(["정상", "val", "고장(참고)"])
    a.set_title(f"{lab[r.지표]}\n반쪽 ρ {r.반쪽ρ:+.2f} · 하중 ρ {r.하중ρ:+.2f} · val 초과 {r.val초과}", fontsize=10)
ax[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle("저주파 경보 후보 — 빨간 점선 = 정상 99.5 %(감쇠는 0.5 %) 문턱", fontsize=12); fig.tight_layout()
fig.savefig(os.path.join(FIG, "1_분포.png"), dpi=130); plt.close(fig)

# %% 그림 2: 반쪽 안정성 (앞 0.9 s vs 뒤 0.9 s, 정상)
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
for a, r in zip(ax.ravel(), R.itertuples()):
    a.scatter(h1[r.지표][N], h2[r.지표][N], s=10, c="#4C72B0")
    lim = [np.nanmin([h1[r.지표][N], h2[r.지표][N]]), np.nanmax([h1[r.지표][N], h2[r.지표][N]])]
    a.plot(lim, lim, "k:", lw=0.8); a.set_xlabel("앞 반쪽"); a.set_ylabel("뒤 반쪽")
    a.set_title(f"{lab[r.지표]}  (ρ {r.반쪽ρ:+.2f})", fontsize=10)
fig.suptitle("정상 120개: 같은 파일 앞 · 뒤 반쪽의 지표 (ρ가 높을수록 파일 성질)", fontsize=12); fig.tight_layout()
fig.savefig(os.path.join(FIG, "2_반쪽안정성.png"), dpi=130); plt.close(fig)
print("done")
