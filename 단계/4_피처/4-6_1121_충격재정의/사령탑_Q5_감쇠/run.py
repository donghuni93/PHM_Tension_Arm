# %% [markdown]
# 4-6 사령탑 Q5 — H11 "빨리 꺼지는 충격 비율" 정의 (ζ 0.08의 정체, 사전 확장, 직접 측정 τ_env)
# 충격 정의(7b-7e detect_impacts, CHOSEN)는 그대로 + 파일 앞뒤 10 ms 안 제외. 라벨은 7번 보고에만 쓴다.
# 선택 규칙(실행 전 고정):
#  - 신뢰도: 정상+val 충격 중 직접 추정 R² ≥ 0.5 인 것에서 (ζ_dict, ζ_env), (τ_dict, τ_env) 스피어만. 더 높은 쪽 양을 채택. 둘 다 < 0.5면 불안정 보고, H11 제외 제안.
#  - 추정법: 직접 측정(R² ≥ 0.5)을 H11의 추정법으로 쓴다. R² < 0.5 또는 측정 불가 충격은 H11 분자·분모에서 뺀다. 사전 추정판은 민감도로만.
#  - 문턱: train 정상의 (R² ≥ 0.5) 충격 분포 90% 분위 (ζ면 위쪽, τ면 아래쪽 10% 분위). 충격 0개 → -1.
# 실행: conda run -n qe-phm python "단계/4_피처/4-6_1121_충격재정의/사령탑_Q5_감쇠/run.py"
import os, sys, json, warnings
from functools import lru_cache
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.ndimage import median_filter
from scipy import stats
from sklearn.metrics import roc_auc_score
from src.data import load_dataset, load_folder
import src.final as F

for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
PREV = os.path.join(os.path.dirname(HERE), "사령탑_7b-7e")
sys.stdout = open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8", buffering=1)
FS = 25000
NOR_C, ABN_C, VAL_C, GRAY = "#2a78d6", "#eb6834", "#7B1FA2", "#9aa0a6"
R = {}
CHOSEN = dict(smooth=1000, base="moving", k=4.5, distance_ms=6.75)
SEG = F._SEG  # 200점 = 8 ms
ZOLD = (0.01, 0.02, 0.04, 0.08)
ZEXT = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
FREQ = np.arange(1000, 12001, 250)


# %% 새 정의 (7b-7e detect_impacts 복사) -----------------------------------------------
@lru_cache(None)
def _sos(kind, f, fs):
    return butter(4, f, btype=kind, fs=fs, output="sos")


def hp1k(X, fs=FS, f=1000):
    X = np.atleast_2d(np.asarray(X, float)); X = X - X.mean(1, keepdims=True)
    return sosfiltfilt(_sos("highpass", f, fs), X, axis=-1)


def envelope(S, lp, fs=FS):
    E = np.abs(hilbert(S, axis=-1))
    return E if lp is None else sosfiltfilt(_sos("lowpass", lp, fs), E, axis=-1)


def baseline(E, win_ms=100, fs=FS):
    n, N = E.shape
    dec = fs // 1000
    M = median_filter(E[:, ::dec], size=(1, int(round(win_ms)) | 1), mode="reflect")
    xs = np.arange(0, N, dec); xi = np.arange(N)
    return np.stack([np.interp(xi, xs, m) for m in M])


def detect_impacts(x, fs=FS):
    s = hp1k(np.asarray(x, float)[None], fs, 1000)
    E = envelope(s, CHOSEN["smooth"], fs); B = baseline(E, 100, fs)
    e, b = E[0], B[0]
    pk = find_peaks(e / b, height=CHOSEN["k"], distance=max(1, int(round(CHOSEN["distance_ms"] * fs / 1000))))[0]
    return dict(t=pk / fs, height_x=e[pk] / b[pk], height_g=e[pk], idx=pk, base=b, hp=s[0])


# %% 데이터 + 검출 + 동일성 assert + 끝단 10 ms 제외 --------------------------------------
X, y, names = load_dataset(os.path.join(_d, "data", "TeamProject_train_변경"))
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val", "val"))
names = [str(n) for n in names]; nv = [str(n) for n in nv]
assert X.shape == (150, 50000) and Xv.shape == (40, 50000)
Xall = np.vstack([X, Xv]); nall = names + nv
grp = np.array(["정상" if v == 0 else "고장" for v in y] + ["val"] * 40)
DET = [detect_impacts(Xall[i]) for i in range(len(Xall))]
rows = []
for i, nm in enumerate(nall):
    d = DET[i]
    for n in range(len(d["t"])):
        rows.append((nm, grp[i], i, n, d["t"][n], d["height_x"][n], d["height_g"][n]))
IMP = pd.DataFrame(rows, columns=["file", "group", "fi", "n", "t_s", "height_x", "height_g"])
REF = pd.read_csv(os.path.join(PREV, "충격표_전체.csv"), encoding="utf-8-sig")
assert len(REF) == len(IMP) and (REF.file.values == IMP.file.values).all() and (REF.group.values == IMP.group.values).all()
for c in ("t_s", "height_x", "height_g"):
    assert np.allclose(REF[c].values, IMP[c].values, equal_nan=True, rtol=1e-6, atol=1e-9), c
print("충격표 동일성 assert 통과", len(IMP), flush=True)
R["identity_assert"] = "끝단 제외 전 충격표가 7b-7e/충격표_전체.csv 와 파일·집단·t_s·height_x·height_g 일치 (rtol 1e-6)"
R["n_impacts_before_edge"] = int(len(IMP))
IMP["t_ms"] = IMP.t_s * 1000
IMP["idx"] = np.round(IMP.t_s * FS).astype(int)
edge = (IMP.t_ms < 10) | (IMP.t_ms > 2000 - 10)
R["n_edge_excluded"] = {g: int(edge[IMP.group == g].sum()) for g in ("정상", "val", "고장")}
IMP = IMP[~edge].reset_index(drop=True)
R["n_impacts"] = {g: int((IMP.group == g).sum()) for g in ("정상", "val", "고장")}
R["n_files"] = {g: int((grp == g).sum()) for g in ("정상", "val", "고장")}
print(R["n_edge_excluded"], R["n_impacts"], flush=True)


# %% 사전 맞춤 (현재 4칸 / 확장 7칸) ------------------------------------------------------
def make_atoms(zs):
    t = np.arange(SEG) / FS
    atoms, par = [], []
    for f in FREQ:
        for z in zs:
            w = 2 * np.pi * f
            wd, a = w * np.sqrt(1 - z ** 2), np.exp(-z * w * t)
            Q, _ = np.linalg.qr(np.stack([a * np.sin(wd * t), a * np.cos(wd * t)], 1))
            atoms.append(Q); par.append((f, z))
    return np.stack(atoms), np.array(par)


A4, P4 = make_atoms(ZOLD); A7, P7 = make_atoms(ZEXT)
a_ref, p_ref = F._ring_atoms()
assert np.allclose(np.abs(A4), np.abs(a_ref), atol=1e-8) and np.allclose(P4, p_ref), "현재 사전 재현 실패"
HPall = np.stack([d["hp"] for d in DET])
SEGS = np.stack([HPall[r.fi][r.idx:r.idx + SEG] for r in IMP.itertuples()])
assert SEGS.shape[1] == SEG
SEGS = SEGS / (np.linalg.norm(SEGS, axis=1, keepdims=True) + 1e-12)
FIT4 = np.linalg.norm(np.einsum("ks,ast->kat", SEGS, A4), axis=2)
FIT7 = np.linalg.norm(np.einsum("ks,ast->kat", SEGS, A7), axis=2)
b4, b7 = FIT4.argmax(1), FIT7.argmax(1)
IMP["f4"], IMP["z4"], IMP["fit4"] = P4[b4, 0], P4[b4, 1], FIT4.max(1)
IMP["f"], IMP["zd"], IMP["fit"] = P7[b7, 0], P7[b7, 1], FIT7.max(1)
IMP["taud"] = 1 / (IMP.zd * 2 * np.pi * IMP.f) * 1000  # ms
IMP["tau4"] = 1 / (IMP.z4 * 2 * np.pi * IMP.f4) * 1000


# %% 직접 측정: 다듬지 않은 |hilbert| 포락선, 봉우리 뒤 0.2–6 ms 로그 직선 ----------------------
ENVRAW = [np.abs(hilbert(HPall[i])) for i in range(len(Xall))]
BASE = [DET[i]["base"] for i in range(len(Xall))]
i0, i1, MIN_PTS = int(round(0.0002 * FS)), int(round(0.006 * FS)), 13


def direct(fi, idx):
    """봉우리 = 검출 위치 ±1 ms 안 다듬지 않은 포락선 최댓값. 0.2 ms부터 포락선 > 2×바탕인 연속 구간(최대 6 ms)을 직선 맞춤"""
    e, b = ENVRAW[fi], BASE[fi]
    lo, hi = max(0, idx - 25), min(len(e), idx + 26)
    p0 = lo + int(np.argmax(e[lo:hi]))
    seg = np.arange(p0 + i0, min(len(e), p0 + i1 + 1))
    ok = e[seg] > 2 * b[seg]
    if not ok[0]:
        return p0, np.nan, np.nan, 0
    n = len(ok) if ok.all() else int(np.argmin(ok))
    seg = seg[:n]
    if n < MIN_PTS:
        return p0, np.nan, np.nan, n
    tt = (seg - p0) / FS * 1000; ly = np.log(e[seg])
    sl, ic, r, *_ = stats.linregress(tt, ly)
    if sl >= 0:
        return p0, np.nan, r ** 2, n
    return p0, -1 / sl, r ** 2, n


res = [direct(r.fi, r.idx) for r in IMP.itertuples()]
IMP["p0"] = [r[0] for r in res]; IMP["tau_env"] = [r[1] for r in res]; IMP["r2"] = [r[2] for r in res]; IMP["npts"] = [r[3] for r in res]
IMP["zeta_env"] = 1 / (2 * np.pi * IMP.f * IMP.tau_env / 1000)
IMP["valid"] = IMP.r2 >= 0.5  # tau_env NaN(기울기 ≥ 0, 구간 짧음)이면 r2가 NaN 이거나 아래서 같이 제외
IMP.loc[IMP.tau_env.isna(), "valid"] = False
R["direct"] = dict(n_impacts=int(len(IMP)), n_no_window=int((IMP.npts < MIN_PTS).sum()), n_slope_nonneg=int((IMP.tau_env.isna() & (IMP.npts >= MIN_PTS)).sum()),
                   n_valid_R2_ge_0p5={g: int(IMP.valid[IMP.group == g].sum()) for g in ("정상", "val", "고장")},
                   frac_valid=float(IMP.valid.mean()))
print(R["direct"], flush=True)
NV = IMP.group.isin(["정상", "val"])

# %% 1. ζ 0.08의 정체 (현재 사전) ------------------------------------------------------------
fit_by_z = {}
for z in ZOLD:
    v = IMP.fit4[(IMP.z4 == z) & NV]
    fit_by_z[str(z)] = dict(n=int(len(v)), median=float(v.median()) if len(v) else None, q25=float(v.quantile(.25)) if len(v) else None, q75=float(v.quantile(.75)) if len(v) else None,
                            frac_fit_lt_0p5=float((v < 0.5).mean()) if len(v) else None)
R["1_fit_by_zeta_current_dict_normal+val"] = fit_by_z
v08, vo = IMP.fit4[(IMP.z4 == 0.08) & NV], IMP.fit4[(IMP.z4 != 0.08) & NV]
R["1_mannwhitney_fit_0p08_vs_others"] = dict(p=float(stats.mannwhitneyu(v08, vo, alternative="less").pvalue), median_0p08=float(v08.median()), median_others=float(vo.median()))
print(json.dumps(R["1_fit_by_zeta_current_dict_normal+val"], ensure_ascii=False), R["1_mannwhitney_fit_0p08_vs_others"], flush=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].boxplot([IMP.fit4[(IMP.z4 == z) & NV] for z in ZOLD], tick_labels=[f"{z}\n(n={int(((IMP.z4 == z) & NV).sum())})" for z in ZOLD], showfliers=False)
ax[0].set_xlabel("현재 사전에서 뽑힌 ζ"); ax[0].set_ylabel("fit"); ax[0].axhline(0.5, color=GRAY, ls="--"); ax[0].set_title("① ζ별 fit 분포 (정상+val 충격)")
for z, c in zip(ZOLD, ["#1b9e77", "#2a78d6", "#e6a117", "#d62728"]):
    ax[1].hist(IMP.fit4[(IMP.z4 == z) & NV], bins=np.linspace(0, 1, 26), histtype="step", density=True, color=c, label=f"ζ {z}", lw=1.6)
ax[1].set_xlabel("fit"); ax[1].legend(); ax[1].set_title("fit 히스토그램 (밀도)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "1_zeta별_fit.png"), dpi=130); plt.close()

# %% 2. 사전 확장 ----------------------------------------------------------------------------
ct = pd.crosstab(IMP.z4[NV], IMP.zd[NV]).reindex(index=ZOLD, columns=ZEXT, fill_value=0)
ct.to_csv(os.path.join(HERE, "교차표_ζ현재_ζ확장.csv"), encoding="utf-8-sig")
ct8 = ct.loc[0.08]
R["2_crosstab_current_vs_extended(normal+val)"] = {str(k): {str(c): int(v) for c, v in row.items()} for k, row in ct.iterrows()}
R["2_extended_marginal"] = {str(z): int((IMP.zd[NV] == z).sum()) for z in ZEXT}
R["2_frac_at_max_0p32"] = float((IMP.zd[NV] == 0.32).mean())
R["2_frac_at_max_0p32_among_old0p08"] = float((IMP.zd[NV & (IMP.z4 == 0.08)] == 0.32).mean())
R["2_old0p08_moved_to_ge0p16"] = float((IMP.zd[NV & (IMP.z4 == 0.08)] >= 0.16).mean())
R["2_fit_median_current_vs_extended"] = dict(current=float(IMP.fit4[NV].median()), extended=float(IMP.fit[NV].median()),
                                             old0p08_current=float(IMP.fit4[NV & (IMP.z4 == 0.08)].median()), old0p08_extended=float(IMP.fit[NV & (IMP.z4 == 0.08)].median()))
R["2_frac_at_min_0p005"] = float((IMP.zd[NV] == 0.005).mean())
print(ct, R["2_frac_at_max_0p32"], R["2_old0p08_moved_to_ge0p16"], R["2_fit_median_current_vs_extended"], flush=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
im = ax[0].imshow(ct.values, cmap="Blues", aspect="auto")
ax[0].set_xticks(range(len(ZEXT))); ax[0].set_xticklabels(ZEXT); ax[0].set_yticks(range(len(ZOLD))); ax[0].set_yticklabels(ZOLD)
for a in range(ct.shape[0]):
    for b in range(ct.shape[1]):
        ax[0].text(b, a, int(ct.values[a, b]), ha="center", va="center", color="k" if ct.values[a, b] < ct.values.max() * .6 else "w")
ax[0].set_xlabel("확장 사전 ζ"); ax[0].set_ylabel("현재 사전 ζ"); ax[0].set_title("② 교차표 (정상+val 충격 수)")
ax[1].bar(range(len(ZEXT)), [R["2_extended_marginal"][str(z)] for z in ZEXT], color=NOR_C)
ax[1].set_xticks(range(len(ZEXT))); ax[1].set_xticklabels(ZEXT); ax[1].set_xlabel("확장 사전 ζ"); ax[1].set_ylabel("충격 수"); ax[1].set_title(f"확장 후 ζ 분포 (0.32 = {100*R['2_frac_at_max_0p32']:.1f} %)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "2_사전확장_교차표.png"), dpi=130); plt.close()

# %% 4. 신뢰도 비교 ---------------------------------------------------------------------------
S = IMP[NV & IMP.valid]
rho_z = stats.spearmanr(S.zd, S.zeta_env); rho_t = stats.spearmanr(S.taud, S.tau_env)
S4 = IMP[NV & IMP.valid]
rho_z4 = stats.spearmanr(S4.z4, S4.zeta_env); rho_t4 = stats.spearmanr(S4.tau4, S4.tau_env)
R["4_reliability"] = dict(n=int(len(S)), rho_zeta=float(rho_z.statistic), rho_tau=float(rho_t.statistic),
                          sens_current_dict=dict(rho_zeta=float(rho_z4.statistic), rho_tau=float(rho_t4.statistic)),
                          sens_fit_ge_0p5=None)
Sf = S[S.fit >= 0.5]
R["4_reliability"]["sens_fit_ge_0p5"] = dict(n=int(len(Sf)), rho_zeta=float(stats.spearmanr(Sf.zd, Sf.zeta_env).statistic), rho_tau=float(stats.spearmanr(Sf.taud, Sf.tau_env).statistic))
rz, rt = R["4_reliability"]["rho_zeta"], R["4_reliability"]["rho_tau"]
if max(rz, rt) < 0.5:
    ADOPT = None
else:
    ADOPT = "zeta" if rz > rt else "tau"
R["4_adopted_quantity"] = ADOPT or "둘 다 불안정(0.5 미만): H11 제외 제안"
print(R["4_reliability"], R["4_adopted_quantity"], flush=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
jit = np.random.default_rng(0)
ax[0].scatter(S.zd * np.exp(jit.normal(0, .06, len(S))), S.zeta_env, s=8, alpha=.4, c=NOR_C)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("ζ_dict (확장 사전, 가로 흔들기)"); ax[0].set_ylabel("ζ_env = 1/(2π f τ_env)")
ax[0].set_title(f"③ ζ: 스피어만 {rz:.2f} (n={len(S)})")
ax[1].scatter(S.taud * np.exp(jit.normal(0, .06, len(S))), S.tau_env, s=8, alpha=.4, c=NOR_C)
lim = [min(S.taud.min(), S.tau_env.min()), max(S.taud.max(), S.tau_env.max())]; ax[1].plot(lim, lim, color=GRAY, ls="--")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("τ_dict (ms)"); ax[1].set_ylabel("τ_env (ms)"); ax[1].set_title(f"τ: 스피어만 {rt:.2f}")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "3_사전vs직접_산점도.png"), dpi=130); plt.close()

# %% 5, 6. 채택량 · 문턱 · H11 ----------------------------------------------------------------
if ADOPT is None:  # 규칙: 둘 다 불안정이면 H11 제안 보류. 그래도 보고용으로 τ로 계산해 둔다(채택 아님)
    QNAME = "tau"; note = "불안정: 채택 아님, 참고 계산(τ)"
else:
    QNAME = ADOPT; note = "채택"
col = "zeta_env" if QNAME == "zeta" else "tau_env"
dcol = "zd" if QNAME == "zeta" else "taud"
trainN = IMP[(IMP.group == "정상") & IMP.valid]
if QNAME == "zeta":
    THR = float(trainN[col].quantile(0.90)); fastf = lambda df: df[col] > THR
else:
    THR = float(trainN[col].quantile(0.10)); fastf = lambda df: df[col] < THR
IMP["fast"] = fastf(IMP) & IMP.valid
dtrain = IMP[(IMP.group == "정상")]
THR_D = float(dtrain[dcol].quantile(0.90 if QNAME == "zeta" else 0.10))  # 사전판 민감도 (fit >= 0.5)
IMP["fast_dict"] = ((IMP[dcol] > THR_D) if QNAME == "zeta" else (IMP[dcol] < THR_D)) & (IMP.fit >= 0.5)
Z04 = IMP.z4 >= 0.04  # 예전 정의 (현재 사전, ζ >= 0.04 비율)
R["6_definition"] = f"{note}: 양={QNAME}, 추정법=직접 측정(다듬지 않은 포락선 로그 직선, R2>=0.5), 문턱={'ζ_env >' if QNAME=='zeta' else 'τ_env <'} {THR:.4f}{'' if QNAME=='zeta' else ' ms'} (train 정상 R2>=0.5 충격의 {'90' if QNAME=='zeta' else '10'}% 분위)"
R["6_threshold"] = THR; R["6_threshold_dict_sensitivity"] = THR_D
nfile = len(Xall)
H11 = np.full(nfile, -1.0); H11d = np.full(nfile, -1.0); H11old = np.full(nfile, -1.0); NIMP = np.zeros(nfile, int); NVAL = np.zeros(nfile, int)
for i in range(nfile):
    d = IMP[IMP.fi == i]; NIMP[i] = len(d)
    v = d[d.valid]; NVAL[i] = len(v)
    if len(v): H11[i] = v.fast.mean()
    dd = d[d.fit >= 0.5]
    if len(dd): H11d[i] = dd.fast_dict.mean()
    if len(d): H11old[i] = (d.z4 >= 0.04).mean()
FT = pd.DataFrame(dict(file=nall, group=grp, n_impacts=NIMP, n_valid=NVAL, H11=H11, H11_dict_sens=H11d, H11_old_zeta4_ge_0p04=H11old))
FT.to_csv(os.path.join(HERE, "H11_파일별.csv"), index=False, encoding="utf-8-sig")
IMP.drop(columns=["p0"]).to_csv(os.path.join(HERE, "충격별_감쇠표.csv"), index=False, encoding="utf-8-sig")
R["6_H11_by_group"] = {g: dict(mean=float(FT.H11[FT.group == g].mean()), median=float(FT.H11[FT.group == g].median()),
                               n_minus1=int((FT.H11[FT.group == g] == -1).sum()), n_zero=int((FT.H11[FT.group == g] == 0).sum()),
                               n_gt0=int((FT.H11[FT.group == g] > 0).sum())) for g in ("정상", "val", "고장")}
print(R["6_definition"], R["6_H11_by_group"], flush=True)

# 5 주파수 의존성 (정상 충격, 채택 방법 = 직접)
Nn = IMP[(IMP.group == "정상") & IMP.valid]
r5 = stats.spearmanr(Nn.f, Nn[col]); r5d = stats.spearmanr(IMP[IMP.group == "정상"].f, IMP[IMP.group == "정상"][dcol])
Nn_f = pd.DataFrame(dict(f=Nn.f, q=Nn[col]))
band = pd.cut(Nn.f, [0, 2500, 4500, 6500, 8000, 13000])
bandmed = Nn.groupby(band)[col].agg(["count", "median"])
R["5_frequency_dependence"] = dict(quantity=col, n=int(len(Nn)), rho_f_vs_adopted=float(r5.statistic), p=float(r5.pvalue),
                                   rho_f_vs_dict_quantity_all_normal=float(r5d.statistic),
                                   band_median={str(k): dict(n=int(r["count"]), median=float(r["median"])) for k, r in bandmed.iterrows()})
# 주파수 보정 H11 (민감도): 같은 대역(1 kHz 칸이 아니라 구간) 안 순위 분위
print(R["5_frequency_dependence"], flush=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].scatter(Nn.f + jit.normal(0, 60, len(Nn)), Nn[col], s=8, alpha=.4, c=NOR_C)
ax[0].set_yscale("log"); ax[0].axhline(THR, color=ABN_C, ls="--", label=f"문턱 {THR:.3g}"); ax[0].legend()
ax[0].set_xlabel("울림 주파수 f (확장 사전, Hz)"); ax[0].set_ylabel(f"{col}" + (" (ms)" if QNAME == "tau" else "")); ax[0].set_title(f"④ 정상 충격: {col} vs f, 스피어만 {r5.statistic:.2f} (n={len(Nn)})")
bm = [Nn[col][band == b] for b in bandmed.index]
ax[1].boxplot(bm, tick_labels=[f"{str(b)}\n(n={len(x)})" for b, x in zip(bandmed.index, bm)], showfliers=False); ax[1].set_yscale("log")
ax[1].set_xlabel("f 구간 (Hz)"); ax[1].set_title("구간별 분포"); ax[1].tick_params(axis="x", labelsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "4_채택량_울림주파수.png"), dpi=130); plt.close()

# %% 7. 보고 (라벨) ----------------------------------------------------------------------------
tr = FT[FT.group != "val"]; yy = (tr.group == "고장").astype(int).values
def auc(c):
    return float(roc_auc_score(yy, tr[c].values))
R["7_AUC_abn_vs_normal_train"] = dict(H11=auc("H11"), H11_dict_sens=auc("H11_dict_sens"), old_zeta4_ge_0p04=auc("H11_old_zeta4_ge_0p04"))
vm = tr[tr.n_valid > 0]; ym = (vm.group == "고장").astype(int).values
R["7_AUC_files_with_valid_impacts_only"] = dict(n_normal=int((ym == 0).sum()), n_abn=int((ym == 1).sum()), H11=float(roc_auc_score(ym, vm.H11)))
print(R["7_AUC_abn_vs_normal_train"], R["7_AUC_files_with_valid_impacts_only"], flush=True)
TRACK = ["abn_022", "abn_015", "abn_017", "abn_002", "abn_014", "nor_045", "nor_116"]
def match(t):
    m = [n for n in nall if t.split("_")[1] in n and ("abn" in n if t.startswith("abn") else "nor" in n)]
    return m
tr_rows, track_files = [], {}
for t in TRACK:
    m = [n for n in nall if t in n]
    if not m:
        m = match(t)
    assert len(m) == 1, (t, m)
    nm = m[0]; i = nall.index(nm); d = IMP[IMP.fi == i]
    track_files[t] = dict(file=nm, n_impacts=int(len(d)), n_valid=int(NVAL[i]), H11=float(H11[i]), H11_dict_sens=float(H11d[i]), H11_old=float(H11old[i]))
    for r in d.itertuples():
        tr_rows.append(dict(trace=t, file=nm, t_ms=round(r.t_ms, 1), f=r.f, zeta_dict=r.zd, fit=round(r.fit, 3), zeta_env=None if np.isnan(r.zeta_env) else round(r.zeta_env, 4),
                            tau_dict_ms=round(r.taud, 3), tau_env_ms=None if np.isnan(r.tau_env) else round(r.tau_env, 3), R2=None if np.isnan(r.r2) else round(r.r2, 3), valid=bool(r.valid), fast=bool(r.fast)))
pd.DataFrame(tr_rows).to_csv(os.path.join(HERE, "추적파일_충격표.csv"), index=False, encoding="utf-8-sig")
R["7_tracked_files"] = track_files
print(json.dumps(track_files, ensure_ascii=False), flush=True)

# 그림 5: H11 분포
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
bins = np.linspace(-0.1, 1.0, 12)
for g, c in (("정상", NOR_C), ("val", VAL_C), ("고장", ABN_C)):
    v = FT.H11[FT.group == g]
    ax[0].hist(v, bins=np.linspace(-1.05, 1.05, 22), alpha=.5, color=c, density=True, label=f"{g} ({len(v)})")
ax[0].set_xlabel("H11 (충격 0개 파일 = −1)"); ax[0].legend(); ax[0].set_title("⑤ H11 분포 (밀도)")
pos = FT[(FT.H11 >= 0)]
ax[1].boxplot([pos.H11[pos.group == g] for g in ("정상", "val", "고장")], tick_labels=[f"{g}\n(n={int((pos.group == g).sum())})" for g in ("정상", "val", "고장")], showfliers=True)
for k, g in enumerate(("정상", "val", "고장")):
    v = pos.H11[pos.group == g]; ax[1].scatter(k + 1 + jit.normal(0, .06, len(v)), v, s=8, alpha=.5, color=[NOR_C, VAL_C, ABN_C][k])
ax[1].set_ylabel("H11"); ax[1].set_title("H11 (유효 충격 있는 파일만)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "5_H11_분포.png"), dpi=130); plt.close()

# 그림 6: 예시 충격 6개 (정상+val, 유효, 채택량 분위 5·20·40·60·80·95 %)
cand = IMP[NV & IMP.valid].sort_values(col).reset_index(drop=True)
qs = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
pick = [cand.iloc[int(q * (len(cand) - 1))] for q in qs]
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for ax, r in zip(axs.ravel(), pick):
    e, b = ENVRAW[int(r.fi)], BASE[int(r.fi)]; p0 = int(r.p0)
    a, bnd = p0 - int(0.001 * FS), p0 + int(0.008 * FS)
    tt = (np.arange(a, bnd) - p0) / FS * 1000
    ax.semilogy(tt, e[a:bnd], color=NOR_C, lw=1, label="|hilbert| (다듬지 않음)"); ax.semilogy(tt, b[a:bnd], color=GRAY, ls=":", label="바탕"); ax.semilogy(tt, 2 * b[a:bnd], color=GRAY, ls="--", lw=.7, label="2×바탕")
    ax.axvspan(0.2, 6, color="#f2f2f2", zorder=0)
    seg = np.arange(p0 + i0, p0 + i0 + int(r.npts)); ts = (seg - p0) / FS * 1000
    ln = np.polyfit(ts, np.log(e[seg]), 1); ax.semilogy(ts, np.exp(np.polyval(ln, ts)), color=ABN_C, lw=2, label="직선 맞춤")
    ax.set_title(f"{r.file} {r.t_ms:.0f} ms\nf {r.f:.0f} Hz τ_env {r.tau_env:.2f} ms ζ_env {r.zeta_env:.3f} R² {r.r2:.2f}", fontsize=9)
    ax.set_xlabel("봉우리 기준 시간 (ms)")
axs[0, 0].legend(fontsize=7)
plt.suptitle("⑥ 예시 충격 6개 (정상+val, 채택량 5·20·40·60·80·95 % 분위)"); plt.tight_layout(); plt.savefig(os.path.join(FIG, "6_예시충격_직선맞춤.png"), dpi=130); plt.close()
R["6_examples"] = [dict(file=r.file, t_ms=float(r.t_ms), f=float(r.f), tau_env=float(r.tau_env), r2=float(r.r2)) for r in pick]

# 부가: 고장 충격 분포 (보고)
R["7_abn_valid_impacts"] = dict(n_impacts=int((IMP.group == "고장").sum()), n_valid=int(IMP.valid[IMP.group == "고장"].sum()), frac_fast=float(IMP.fast[(IMP.group == "고장") & IMP.valid].mean()))
R["6_frac_fast_normal_valid_impacts"] = float(IMP.fast[(IMP.group == "정상") & IMP.valid].mean())
R["6_frac_fast_val_valid_impacts"] = float(IMP.fast[(IMP.group == "val") & IMP.valid].mean())
json.dump(R, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("done")
