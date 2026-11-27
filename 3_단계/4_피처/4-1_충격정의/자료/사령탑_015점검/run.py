# %% [markdown]
# 4-6 사령탑 015점검 — 새 충격 정의(7b-7e)가 진짜 충격을 잡는지 점검 (정의는 고치지 않음, 제안만)
# (1) 파일 끝단 가짜  (2) 넓은 대역 검사(누설)  (3) 015 위상 잠금  (4) 015 충격 36개 그림  (5) 제안별 충격 수 변화
# 분류 규칙은 실행 전에 고정: 넓음 = 1 kHz 위 칸(1-2·2-4·4-8·8-12.5k) 중 2개 이상 +0.5 dec 이상 /
#   좁음-경계 = 1-2k 칸만 +0.5 이상(다른 1k 위 칸은 미만) & 0.2-1k 상승 > 1-2k 상승 / 좁음-기타 = 나머지
# 실행: conda run -n qe-phm python "3_단계/4_피처/4-1_충격정의/자료/사령탑_015점검/run.py"
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
from src.data import load_dataset, load_folder
import src.final as F

for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
PREV = os.path.join(os.path.dirname(HERE), "사령탑_7b-7e")
FS = 25000
NOR_C, ABN_C, VAL_C, GRAY = "#2a78d6", "#eb6834", "#7B1FA2", "#9aa0a6"
GCOL = {"정상": NOR_C, "val": VAL_C, "고장": ABN_C}
R = {}
CHOSEN = dict(smooth=1000, base="moving", k=4.5, distance_ms=6.75)


# %% 새 정의 (사령탑_7b-7e/run.py 의 detect_impacts 를 그대로 복사) ---------------------
@lru_cache(None)
def _sos(kind, f, fs):
    return butter(4, f, btype=kind, fs=fs, output="sos")


def hp1k(X, fs=FS, f=1000):
    X = np.atleast_2d(np.asarray(X, float)); X = X - X.mean(1, keepdims=True)
    return sosfiltfilt(_sos("highpass", f, fs), X, axis=-1)


def envelope(S, lp, fs=FS):
    E = np.abs(hilbert(S, axis=-1))
    return E if lp is None else sosfiltfilt(_sos("lowpass", lp, fs), E, axis=-1)


def moving_median_dec(E, win_ms, fs=FS):
    dec = fs // 1000
    size = int(round(win_ms)) | 1
    return median_filter(E[:, ::dec], size=(1, size), mode="reflect"), dec


def baseline(E, mode, win_ms=100, fs=FS):
    n, N = E.shape
    if mode == "global":
        return np.repeat(np.median(E, 1, keepdims=True), N, 1)
    M, dec = moving_median_dec(E, win_ms, fs)
    xs = np.arange(0, N, dec); xi = np.arange(N)
    return np.stack([np.interp(xi, xs, m) for m in M])


def detect_impacts(x, fs=FS, smooth=None, base=None, k=None, distance_ms=None, base_win_ms=100, hp=1000, search_ms=30):
    smooth = CHOSEN["smooth"] if smooth is None else smooth
    base = CHOSEN["base"] if base is None else base
    k = CHOSEN["k"] if k is None else k
    distance_ms = CHOSEN["distance_ms"] if distance_ms is None else distance_ms
    s = hp1k(np.asarray(x, float)[None], fs, hp)
    E = envelope(s, smooth, fs); B = baseline(E, base, base_win_ms, fs)
    e, b = E[0], B[0]
    pk = find_peaks(e / b, height=k, distance=max(1, int(round(distance_ms * fs / 1000))))[0]
    W = int(search_ms * fs / 1000)
    rise, dec = np.full(len(pk), np.nan), np.full(len(pk), np.nan)
    for n, p in enumerate(pk):
        h = e[p] - b[p]
        back = e[max(0, p - W):p + 1][::-1] - b[p]
        i90 = np.flatnonzero(back <= 0.9 * h); i10 = np.flatnonzero(back <= 0.1 * h)
        if len(i90) and len(i10): rise[n] = (i10[0] - i90[0]) / fs * 1000
        fwd = e[p:p + W + 1] - b[p]
        j = np.flatnonzero(fwd <= h / np.e)
        if len(j): dec[n] = j[0] / fs * 1000
    return dict(t=pk / fs, height_x=e[pk] / b[pk], height_g=e[pk], rise_ms=rise, decay_ms=dec, idx=pk, env=e, base=b, hp=s[0])


# %% 데이터 + 검출 + 이전 결과와 동일성 assert -----------------------------------------
X, y, names = load_dataset(os.path.join(_d, "2_data", "TeamProject_train_변경"))
Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val", "val"))
names = [str(n) for n in names]; nv = [str(n) for n in nv]
assert X.shape == (150, 50000) and Xv.shape == (40, 50000)
Xall = np.vstack([X, Xv]); nall = names + nv
grp = np.array(["정상" if v == 0 else "고장" for v in y] + ["val"] * 40)
Xm = Xall - Xall.mean(1, keepdims=True)
DET = [detect_impacts(Xall[i]) for i in range(len(Xall))]
rows = []
for i, nm in enumerate(nall):
    d = DET[i]
    for n in range(len(d["t"])):
        rows.append((nm, grp[i], d["t"][n], d["height_x"][n], d["height_g"][n], d["rise_ms"][n], d["decay_ms"][n]))
IMP = pd.DataFrame(rows, columns=["file", "group", "t_s", "height_x", "height_g", "rise_ms", "decay_ms"])
REF = pd.read_csv(os.path.join(PREV, "충격표_전체.csv"), encoding="utf-8-sig")
assert len(REF) == len(IMP) and (REF.file.values == IMP.file.values).all() and (REF.group.values == IMP.group.values).all()
for c in ("t_s", "height_x", "height_g", "rise_ms", "decay_ms"):
    assert np.allclose(REF[c].values, IMP[c].values, equal_nan=True, rtol=1e-6, atol=1e-9), c
print("충격표 동일성 assert 통과", len(IMP), flush=True)
R["identity_assert"] = "충격표_전체.csv 와 파일·집단·t_s·height_x·height_g·rise_ms·decay_ms 모두 일치 (rtol 1e-6)"
R["n_impacts_total"] = int(len(IMP))
R["n_impacts_by_group"] = {g: int((IMP.group == g).sum()) for g in ("정상", "val", "고장")}
R["n_files_by_group"] = {g: int((grp == g).sum()) for g in ("정상", "val", "고장")}

# 울림 fit (충격 뒤 8 ms를 울림 사전에 투영, 7f와 같은 방식). 파일 끝 8 ms 안은 NaN
atoms, par = F._ring_atoms()
HPall = hp1k(Xall)
fit = np.full(len(IMP), np.nan); fr = np.full(len(IMP), np.nan); zr = np.full(len(IMP), np.nan)
r0 = 0
for i in range(len(Xall)):
    d = DET[i]
    for n, p in enumerate(d["idx"]):
        if p + F._SEG <= 50000:
            seg = HPall[i][p:p + F._SEG]; seg = seg / (np.linalg.norm(seg) + 1e-12)
            ft = np.linalg.norm(np.einsum("s,ast->at", seg, atoms), axis=1); b = int(ft.argmax())
            fit[r0 + n] = ft[b]; fr[r0 + n] = par[b, 0]; zr[r0 + n] = par[b, 1]
    r0 += len(d["t"])
IMP["fit"] = fit; IMP["f_Hz"] = fr; IMP["zeta"] = zr
IMP["t_ms"] = IMP.t_s * 1000
IMP["idx"] = np.round(IMP.t_s * FS).astype(int)
IMP["fi"] = [nall.index(f) for f in IMP.file]

# %% (1) 끝단 --------------------------------------------------------------------------
print("== (1) 끝단", flush=True)
IMP["edge_dist_ms"] = np.minimum(IMP.t_ms, 2000 - IMP.t_ms)
sens = {}
for E_ in (5, 10, 20):
    out = {}
    for g in ("정상", "val", "고장"):
        d = IMP[IMP.group == g]; ex = d[d.edge_dist_ms < E_]
        chance = 2 * E_ / 2000
        bt = stats.binomtest(len(ex), len(d), chance, alternative="greater")
        out[g] = dict(n_impacts=int(len(d)), n_excluded=int(len(ex)), n_remaining=int(len(d) - len(ex)), frac=float(len(ex) / len(d)), chance_frac=chance,
                      ratio_vs_chance=float(len(ex) / len(d) / chance), binom_p_greater=float(bt.pvalue), n_files_with_edge=int(ex.file.nunique()),
                      n_front=int((ex.t_ms < E_).sum()), n_back=int((ex.t_ms > 2000 - E_).sum()))
    sens[f"{E_}ms"] = out
R["1_edge_sensitivity"] = sens
ed = IMP[IMP.edge_dist_ms < 10].copy()
ed["side"] = np.where(ed.t_ms < 10, "앞", "뒤")


def _med(s):
    return float(s.median()) if s.notna().any() else None


R["1_edge10_shape"] = {g: dict(n=int((ed.group == g).sum()), rise_median=_med(ed[ed.group == g].rise_ms), rise_nan=int(ed[ed.group == g].rise_ms.isna().sum()),
                              decay_median=_med(ed[ed.group == g].decay_ms), decay_nan=int(ed[ed.group == g].decay_ms.isna().sum()),
                              fit_median=_med(ed[ed.group == g].fit), fit_nan=int(ed[ed.group == g].fit.isna().sum()),
                              fit_lt_0p5=int((ed[ed.group == g].fit < .5).sum()),
                              lump=int((ed[ed.group == g].rise_ms > ed[ed.group == g].decay_ms).sum())) for g in ("정상", "val", "고장")}
ins = IMP[IMP.edge_dist_ms >= 10]
R["1_inside_shape"] = {g: dict(n=int((ins.group == g).sum()), rise_median=_med(ins[ins.group == g].rise_ms), decay_median=_med(ins[ins.group == g].decay_ms),
                              fit_median=_med(ins[ins.group == g].fit), fit_lt_0p5=int((ins[ins.group == g].fit < .5).sum()),
                              frac_fit_lt_0p5=float((ins[ins.group == g].fit < .5).sum() / max(1, ins[ins.group == g].fit.notna().sum()))) for g in ("정상", "val", "고장")}
ed[["file", "group", "side", "t_ms", "height_x", "rise_ms", "decay_ms", "fit", "f_Hz"]].round(3).to_csv(os.path.join(HERE, "끝단충격_10ms.csv"), index=False, encoding="utf-8-sig")
R["1_edge10_list"] = ed[["file", "group", "side", "t_ms", "height_x", "rise_ms", "decay_ms", "fit"]].round(3).to_dict("records")
R["1_edge10_files_group"] = {g: sorted(set(ed[ed.group == g].file)) for g in ("정상", "val", "고장")}

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
bb = np.linspace(0, 100, 26)
for g in ("정상", "val", "고장"):
    d = IMP[IMP.group == g]
    ax[0].hist(d.edge_dist_ms[d.edge_dist_ms < 100], bins=bb, histtype="step", lw=2, color=GCOL[g], density=False, weights=np.ones((d.edge_dist_ms < 100).sum()) / len(d), label=f"{g} (전체 {len(d)})")
ax[0].axhline(4 / 2000 * 2 * 1.0 * 1, color="k", ls=":", label="균등 기대(4 ms 칸당 2×4/2000)")
ax[0].axvline(10, color="k", ls="--"); ax[0].set_xlabel("가까운 파일 끝까지 거리 (ms)"); ax[0].set_ylabel("전체 충격 중 비율 (4 ms 칸)"); ax[0].legend(fontsize=8); ax[0].set_title("충격의 끝단 거리 (0-100 ms)", fontsize=10)
ax[1].scatter(ins.rise_ms, ins.decay_ms, s=6, color=GRAY, alpha=.35, label="끝단 밖 전체")
for g in ("정상", "val", "고장"):
    d = ed[ed.group == g]; ax[1].scatter(d.rise_ms, d.decay_ms, s=40, color=GCOL[g], label=f"{g} 끝단 {len(d)}", edgecolor="k", lw=.6)
ax[1].plot([.1, 40], [.1, 40], "k--", lw=1); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("오름 (ms)"); ax[1].set_ylabel("꺼짐 (ms, NaN은 안 보임)"); ax[1].legend(fontsize=8)
ax[1].set_title("끝단 10 ms 충격의 모양", fontsize=10)
for g in ("정상", "val", "고장"):
    d = ed[ed.group == g]; ax[2].scatter(np.minimum(d.t_ms, 2000 - d.t_ms), d.fit, s=40, color=GCOL[g], edgecolor="k", lw=.6, label=g)
ax[2].axhline(.5, color="k", ls=":"); ax[2].set_xlabel("끝까지 거리 (ms)"); ax[2].set_ylabel("울림 fit (파일 끝 8 ms 안은 NaN이라 안 보임)"); ax[2].legend(fontsize=8); ax[2].set_title("끝단 10 ms 충격의 fit", fontsize=10)
fig.savefig(os.path.join(FIG, "1_끝단충격.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

# %% (2) 넓은 대역 검사 ------------------------------------------------------------------
print("== (2) 넓은 대역", flush=True)
BANDS = [("0.2-1k", 200, 1000), ("1-2k", 1000, 2000), ("2-4k", 2000, 4000), ("4-8k", 4000, 8000), ("8-12.5k", 8000, 12500)]
NFFT = 1024; freqs = np.fft.rfftfreq(NFFT, 1 / FS)
SEG0, SEG1 = int(round(0.001 * FS)), int(round(0.008 * FS))   # 충격 -1 ms ~ +8 ms
LEN = SEG0 + SEG1                                                # 225 샘플 = 9 ms
WIN = np.hanning(LEN)
assert LEN == 225


def seg_psd(xrow, s):
    seg = xrow[s:s + LEN]
    seg = seg - seg.mean()
    return np.abs(np.fft.rfft(seg * WIN, NFFT)) ** 2


def band_pow(P):
    return np.array([P[(freqs >= lo) & (freqs < hi)].sum() for _, lo, hi in BANDS])


def free_starts(fi, cand_starts):
    """다른 충격의 [-1,+8] ms 구간과 안 겹치는 9 ms 창 시작점만 남김 (파일 안 모든 충격 기준)"""
    imp = DET[fi]["idx"]
    return [s for s in cand_starts if 0 <= s and s + LEN <= 50000 and not np.any((imp - SEG0 < s + LEN) & (imp + SEG1 > s))]


rng = np.random.default_rng(2112)
res = []
for r_ in IMP.itertuples():
    fi = r_.fi; p = r_.idx; xr = Xm[fi]
    s_imp = p - SEG0
    row = dict(file=r_.file, group=r_.group, t_ms=r_.t_ms, fit=r_.fit)
    if s_imp < 0 or s_imp + LEN > 50000:
        row.update(ref_kind="범위밖", cls="범위밖"); res.append(row); continue
    P_imp = band_pow(seg_psd(xr, s_imp))
    # 바탕 구간 후보: 창 전체가 충격 시점 기준 [-60, -10] ms 안
    cand = free_starts(fi, range(p - int(0.060 * FS), p - int(0.010 * FS) - LEN + 1))
    if len(cand):
        s_ref = max(cand)                                          # 충격에 가장 가까운 충격 없는 창
        P_ref = band_pow(seg_psd(xr, s_ref)); kind = "앞"; ref_t = (s_ref - p) / FS * 1000
    else:
        fs_ = free_starts(fi, range(0, 50000 - LEN))
        if len(fs_) < 1:
            row.update(ref_kind="없음", cls="기준없음"); res.append(row); continue
        pick = rng.choice(fs_, size=20, replace=True)
        P_ref = np.median(np.stack([band_pow(seg_psd(xr, s)) for s in pick]), 0); kind = "파일"; ref_t = np.nan
    dec = np.log10(P_imp / P_ref)
    row.update(ref_kind=kind, ref_t_ms=ref_t, **{f"d_{b[0]}": float(v) for b, v in zip(BANDS, dec)})
    n_up = int((dec[1:] >= 0.5).sum())
    if n_up >= 2: cl = "넓음"
    elif dec[1] >= 0.5 and n_up == 1 and dec[0] > dec[1]: cl = "좁음-경계"
    else: cl = "좁음-기타"
    row["cls"] = cl; row["n_up"] = n_up
    res.append(row)
RES = pd.DataFrame(res)
RES["edge_dist_ms"] = IMP.edge_dist_ms.values
RES["idx"] = IMP.idx.values
RES.round(3).to_csv(os.path.join(HERE, "충격별_대역분류.csv"), index=False, encoding="utf-8-sig")
CLS = ["넓음", "좁음-경계", "좁음-기타", "범위밖", "기준없음"]
R["2_class_by_group"] = {g: {c: int(((RES.group == g) & (RES.cls == c)).sum()) for c in CLS} for g in ("정상", "val", "고장")}
R["2_class_frac_by_group"] = {g: {c: float(((RES.group == g) & (RES.cls == c)).sum() / max(1, (RES.group == g).sum())) for c in CLS} for g in ("정상", "val", "고장")}
R["2_ref_kind"] = {k: int((RES.ref_kind == k).sum()) for k in ("앞", "파일", "범위밖", "없음")}
valid = RES[RES.cls.isin(["넓음", "좁음-경계", "좁음-기타"])].copy()
R["2_fit_vs_class"] = {c: dict(n=int((valid.cls == c).sum()), fit_lt_0p5=int(((valid.cls == c) & (valid.fit < .5)).sum()), fit_nan=int(((valid.cls == c) & valid.fit.isna()).sum()),
                               frac_fit_lt_0p5=float(((valid.cls == c) & (valid.fit < .5)).sum() / max(1, ((valid.cls == c) & valid.fit.notna()).sum())),
                               fit_median=_med(valid[valid.cls == c].fit)) for c in ("넓음", "좁음-경계", "좁음-기타")}
R["2_fit_overall_lt_0p5"] = dict(n=int((valid.fit < .5).sum()), of=int(valid.fit.notna().sum()))
R["2_class_of_fit_lt_0p5"] = {c: int(((valid.cls == c) & (valid.fit < .5)).sum()) for c in ("넓음", "좁음-경계", "좁음-기타")}
R["2_band_rise_median_by_class"] = {c: {b[0]: float(valid[valid.cls == c][f"d_{b[0]}"].median()) for b in BANDS} for c in ("넓음", "좁음-경계", "좁음-기타") if (valid.cls == c).any()}
R["2_band_rise_median_by_group"] = {g: {b[0]: float(valid[valid.group == g][f"d_{b[0]}"].median()) for b in BANDS} for g in ("정상", "val", "고장")}
TRACE = ["abn_015", "abn_022", "abn_002", "abn_014", "nor_045"]
R["2_trace"] = {nm: dict({c: int(((RES.file == nm) & (RES.cls == c)).sum()) for c in CLS}, total=int((RES.file == nm).sum())) for nm in TRACE}
bf = RES[RES.cls == "좁음-경계"].groupby("file").size().sort_values(ascending=False)
R["2_border_top_files"] = [(str(k), int(v), int((RES.file == k).sum())) for k, v in bf.head(10).items()]
R["2_border_files_n"] = int(len(bf))
R["2_border_files_by_group"] = {g: int(sum(1 for f in bf.index if grp[nall.index(f)] == g)) for g in ("정상", "val", "고장")}
# 파일 전체의 대역 파워(평균 PSD)를 정상 중앙값과 비교 (dec)
Pfile = np.array([band_pow(np.mean(np.abs(np.fft.rfft(Xm[i][:49152].reshape(-1, 1024) * np.hanning(1024), axis=1)) ** 2, 0)) for i in range(len(Xall))])
Pn = np.median(Pfile[grp == "정상"], 0)
R["2_file_band_dec_vs_normal_median"] = {nm: {b[0]: float(np.log10(Pfile[nall.index(nm), j] / Pn[j])) for j, b in enumerate(BANDS)} for nm in TRACE}

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
w = 0.25
for j, g in enumerate(("정상", "val", "고장")):
    ax[0].bar(np.arange(3) + (j - 1) * w, [R["2_class_frac_by_group"][g][c] for c in CLS[:3]], w, color=GCOL[g], label=f"{g} (n={int((valid.group == g).sum())})")
ax[0].set_xticks(range(3), CLS[:3]); ax[0].set_ylabel("집단 안 비율"); ax[0].legend(fontsize=8); ax[0].set_title("(2) 충격 분류 비율", fontsize=10)
data_ = [valid[valid.cls == c].fit.dropna().values for c in ("넓음", "좁음-경계", "좁음-기타")]
ax[1].boxplot(data_, tick_labels=[f"{c}\n(n={len(v)})" for c, v in zip(("넓음", "좁음-경계", "좁음-기타"), data_)]); ax[1].axhline(.5, color="k", ls=":")
ax[1].set_ylabel("울림 fit"); ax[1].set_title("분류별 fit", fontsize=10)
mk = {"넓음": "o", "좁음-경계": "s", "좁음-기타": "^"}
for c in mk:
    d = valid[valid.cls == c]; ax[2].scatter(d["d_1-2k"], d["d_0.2-1k"], marker=mk[c], s=14, alpha=.5, label=c, color={"넓음": NOR_C, "좁음-경계": ABN_C, "좁음-기타": GRAY}[c])
d = valid[valid.file == "abn_015"]; ax[2].scatter(d["d_1-2k"], d["d_0.2-1k"], marker="x", s=40, color="k", label="abn_015")
ax[2].plot([-1, 3], [-1, 3], "k--", lw=1); ax[2].axvline(.5, color="k", ls=":"); ax[2].set_xlabel("1-2 kHz 상승 (dec)"); ax[2].set_ylabel("0.2-1 kHz 상승 (dec)"); ax[2].legend(fontsize=8)
ax[2].set_title("1-2k vs 0.2-1k 상승 (대각 위 = 아래 대역이 더 큼)", fontsize=10)
fig.savefig(os.path.join(FIG, "2_대역분류.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
for a, nm in zip(axs, TRACE):
    d = valid[valid.file == nm]
    for c, col in (("넓음", NOR_C), ("좁음-경계", ABN_C), ("좁음-기타", GRAY)):
        for _, rr in d[d.cls == c].iterrows():
            a.plot(range(5), [rr[f"d_{b[0]}"] for b in BANDS], color=col, alpha=.5, lw=1)
    a.axhline(.5, color="k", ls=":"); a.set_xticks(range(5), [b[0] for b in BANDS], rotation=30, fontsize=8)
    a.set_title(f"{nm}: " + " ".join(f"{c} {R['2_trace'][nm][c]}" for c in CLS[:3]), fontsize=9)
axs[0].set_ylabel("충격 구간 vs 바탕 구간 파워 상승 (dec)")
fig.savefig(os.path.join(FIG, "2_추적파일_칸별상승.png"), dpi=105, bbox_inches="tight"); plt.close(fig)

# %% (3) 015 위상 잠금 ----------------------------------------------------------------
print("== (3) 위상 잠금", flush=True)
sos_bp = butter(4, [20, 60], btype="bandpass", fs=FS, output="sos")
BP = sosfiltfilt(sos_bp, Xm, axis=-1)
PH = np.angle(hilbert(BP, axis=-1))


def rayleigh(ph):
    n = len(ph); z = np.mean(np.exp(1j * ph)); Rv = float(np.abs(z))
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * Rv) ** 2)) - (1 + 2 * n))   # Zar 근사
    return Rv, float(min(1, p)), float(np.angle(z))


ph_rows = []
for i, nm in enumerate(nall):
    idx = DET[i]["idx"]
    if len(idx) >= 2: Rv, p, mu = rayleigh(PH[i][idx])
    else: Rv = p = mu = np.nan
    ph_rows.append(dict(file=nm, group=grp[i], n=len(idx), R=Rv, p=p, mean_phase=mu, band_energy_frac=float((BP[i] ** 2).sum() / (Xm[i] ** 2).sum())))
PHR = pd.DataFrame(ph_rows).set_index("file")
r15 = PHR.loc["abn_015"]
nor10 = PHR[(PHR.group == "정상") & (PHR.n >= 10)]
nor5 = PHR[(PHR.group == "정상") & (PHR.n >= 5)]
allg5 = PHR[PHR.n >= 5]
line99 = float(np.percentile(nor5.R, 99))
# 순열 기준: 같은 파일의 위상 시계열에서 무작위 시점 n개 → R (충격 수 n이 달라도 공평한 비교)
rng3 = np.random.default_rng(3)
pperm = {}; r99 = {}
for nm in allg5.index:
    i = nall.index(nm); n = int(allg5.loc[nm, "n"])
    pos = rng3.integers(0, 50000, size=(4000, n))
    Rr = np.abs(np.mean(np.exp(1j * PH[i][pos]), axis=1))
    pperm[nm] = float((Rr >= allg5.loc[nm, "R"]).mean()); r99[nm] = float(np.percentile(Rr, 99))
PHR["R_perm99"] = pd.Series(r99); PHR["p_perm"] = pd.Series(pperm)
PHR.round(5).to_csv(os.path.join(HERE, "위상잠금_파일별.csv"), encoding="utf-8-sig")
R["3_abn_015"] = dict(n=int(r15.n), R=float(r15.R), p_rayleigh=float(r15.p), mean_phase_rad=float(r15.mean_phase), band_energy_frac_20_60=float(r15.band_energy_frac),
                      p_perm=pperm["abn_015"], R_perm99=r99["abn_015"])
R["3_normal_n_ge10"] = dict(n_files=int(len(nor10)), R_median=float(nor10.R.median()), R_p95=float(nor10.R.quantile(.95)), R_max=float(nor10.R.max()), R_max_file=str(nor10.R.idxmax()),
                            frac_p_rayleigh_lt_0p01=float((nor10.p < .01).mean()), n_p_rayleigh_lt_0p01=int((nor10.p < .01).sum()))
R["3_normal_n_ge5"] = dict(n_files=int(len(nor5)), R_median=float(nor5.R.median()), R_p99_line=line99, R_max=float(nor5.R.max()), R_max_file=str(nor5.R.idxmax()))
R["3_files_n_ge5_by_group"] = {g: int((allg5.group == g).sum()) for g in ("정상", "val", "고장")}
above = allg5[allg5.R > line99].sort_values("R", ascending=False)
R["3_above_line99"] = [dict(file=str(k), group=str(v.group), n=int(v.n), R=float(v.R), p_rayleigh=float(v.p), p_perm=float(PHR.loc[k, "p_perm"]), mean_phase=float(v.mean_phase)) for k, v in above.iterrows()]
R["3_perm_p_lt_0p01"] = [dict(file=k, group=str(PHR.loc[k, "group"]), n=int(PHR.loc[k, "n"]), R=float(PHR.loc[k, "R"]), p_perm=v) for k, v in sorted(pperm.items(), key=lambda kv: kv[1]) if v < 0.01]
R["3_perm_p_lt_0p01_by_group"] = {g: int(sum(1 for d in R["3_perm_p_lt_0p01"] if d["group"] == g)) for g in ("정상", "val", "고장")}
R["3_perm_p_lt_0p05_normal"] = int(sum(1 for k, v in pperm.items() if v < .05 and PHR.loc[k, "group"] == "정상"))
t15 = DET[nall.index("abn_015")]["t"] * 1000
gap15 = np.diff(t15)
R["3_015_gaps_ms"] = [round(float(g), 2) for g in gap15]
R["3_015_gap_n"] = int(len(gap15)); R["3_015_gap_in_23_29"] = int(((gap15 >= 23) & (gap15 <= 29)).sum()); R["3_015_gap_in_20_30"] = int(((gap15 >= 20) & (gap15 <= 30)).sum())
R["3_015_gap_lt_30"] = int((gap15 < 30).sum()); R["3_015_gap_median"] = float(np.median(gap15))
gn = np.concatenate([np.diff(DET[i]["t"]) * 1000 for i in range(len(Xall)) if grp[i] == "정상" and len(DET[i]["t"]) >= 2])
R["3_normal_gap_frac_23_29"] = float(((gn >= 23) & (gn <= 29)).mean()); R["3_normal_gap_n"] = int(len(gn))
fax = np.fft.rfftfreq(50000, 1 / FS); sp = np.abs(np.fft.rfft(Xm[nall.index("abn_015")] * np.hanning(50000))) ** 2
m2 = (fax >= 20) & (fax <= 60)
R["3_015_psd_peak_20_60_Hz"] = float(fax[m2][np.argmax(sp[m2])])

fig = plt.figure(figsize=(17, 5))
a0 = fig.add_subplot(1, 3, 1, projection="polar")
i15 = nall.index("abn_015"); ph15 = PH[i15][DET[i15]["idx"]]
a0.hist(ph15, bins=18, range=(-np.pi, np.pi), color=ABN_C, alpha=.8)
a0.set_title(f"abn_015 충격 시점의 20-60 Hz 위상\nn={len(ph15)}  R={r15.R:.3f}  Rayleigh p={r15.p:.2g}  순열 p={pperm['abn_015']:.3g}", fontsize=10)
a1 = fig.add_subplot(1, 3, 2)
rg = np.random.default_rng(0)
for j, g in enumerate(("정상", "val", "고장")):
    d = allg5[allg5.group == g]; a1.scatter(j + rg.uniform(-.2, .2, len(d)), d.R, s=22, color=GCOL[g], alpha=.75)
a1.scatter([2.25], [r15.R], marker="x", color="k", s=70); a1.annotate("abn_015", (2.27, r15.R), fontsize=8)
a1.axhline(line99, color="k", ls="--", label=f"정상(충격≥5) 99% = {line99:.2f}")
a1.set_xticks([0, 1, 2], [f"정상 {R['3_files_n_ge5_by_group']['정상']}", f"val {R['3_files_n_ge5_by_group']['val']}", f"고장 {R['3_files_n_ge5_by_group']['고장']}"])
a1.set_ylabel("위상 집중도 R (충격≥5 파일)"); a1.legend(fontsize=8); a1.set_title("파일별 R", fontsize=10)
a2 = fig.add_subplot(1, 3, 3)
a2.hist(gap15, bins=np.arange(0, 120, 2.5), color=ABN_C, alpha=.8)
a2.axvspan(23, 29, color="k", alpha=.15, label="35-44 Hz 한 주기 (23-29 ms)"); a2.set_xlabel("abn_015 연속 충격 간격 (ms)"); a2.set_ylabel("개수"); a2.legend(fontsize=8)
a2.set_title(f"015 충격 간격 (23-29 ms 안 {R['3_015_gap_in_23_29']}/{R['3_015_gap_n']})", fontsize=10)
fig.savefig(os.path.join(FIG, "3_015_위상잠금.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.hist(nor10.R, bins=np.linspace(0, 1, 26), color=NOR_C, alpha=.7, label=f"정상(충격≥10) {len(nor10)}개 파일")
ax.axvline(r15.R, color=ABN_C, lw=2, label=f"abn_015 R={r15.R:.3f}"); ax.set_xlabel("R"); ax.set_ylabel("파일 수"); ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "3_정상R분포.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

# %% (4) 015 충격 36개 -------------------------------------------------------------------
print("== (4) 015 그림", flush=True)
d15 = DET[i15]
sub = IMP[IMP.file == "abn_015"].reset_index(drop=True)
rs = RES[RES.file == "abn_015"].reset_index(drop=True)
assert len(sub) == 36 and len(rs) == 36
CLCOL = {"넓음": "#2e9e5b", "좁음-경계": ABN_C, "좁음-기타": GRAY, "범위밖": "k", "기준없음": "k"}
PER = 6
for sh in range(int(np.ceil(36 / PER))):
    fig, axs = plt.subplots(2, PER, figsize=(4.3 * PER, 8.2))
    for c in range(PER):
        n = sh * PER + c
        r_ = sub.iloc[n]; cl = rs.iloc[n]; p = int(r_.idx)
        lo, hi = max(0, p - 250), min(50000, p + 251); tms = (np.arange(lo, hi) - p) / FS * 1000
        a = axs[0, c]
        a.plot(tms, d15["hp"][lo:hi], lw=.6, color=GRAY); a.plot(tms, d15["env"][lo:hi], color=ABN_C, lw=1.8, label="포락선")
        a.plot(tms, d15["base"][lo:hi], color=NOR_C, ls="--", lw=1.2, label="바탕"); a.plot(tms, CHOSEN["k"] * d15["base"][lo:hi], color="k", ls=":", lw=1.2, label="문턱 4.5x")
        a.axvspan(-1, 8, color="y", alpha=.12)
        a.set_title(f"{r_.t_ms:.2f} ms · {r_.height_x:.1f}x\nf {r_.f_Hz/1000:.2f}k ζ {r_.zeta:.2f} fit {r_.fit:.2f}\n[{cl.cls}] 끝단거리 {r_.edge_dist_ms:.0f} ms", fontsize=8.5, color=CLCOL[cl.cls])
        a.set_xlabel("ms (충격 기준)"); a.set_ylabel("g")
        if c == 0: a.legend(fontsize=7)
        b = axs[1, c]; s_imp = p - SEG0
        if s_imp >= 0 and s_imp + LEN <= 50000:
            Pi = seg_psd(Xm[i15], s_imp); m = freqs <= 12500
            b.semilogy(freqs[m] / 1000, np.maximum(Pi[m], 1e-14), color=ABN_C, lw=1.2, label="충격 [-1,+8] ms")
            if np.isfinite(cl.get("ref_t_ms", np.nan)):
                Pr = seg_psd(Xm[i15], p + int(round(cl.ref_t_ms * FS / 1000)))
                b.semilogy(freqs[m] / 1000, np.maximum(Pr[m], 1e-14), color=NOR_C, lw=1.2, label="바탕 구간(앞)")
            for e_ in (0.2, 1, 2, 4, 8): b.axvline(e_, color="k", ls=":", lw=.7)
            b.set_xscale("log"); b.set_xlim(0.1, 12.5)
            b.set_title("상승(dec) " + " ".join(f"{bb_[0].replace('k', '')}:{cl['d_' + bb_[0]]:+.1f}" for bb_ in BANDS), fontsize=7.5)
        b.set_xlabel("kHz"); b.set_ylabel("파워")
        if c == 0: b.legend(fontsize=7)
    fig.suptitle(f"abn_015 새 충격 {sh * PER + 1}-{(sh + 1) * PER} / 36 (노랑 = [-1,+8] ms, 제목 색 = 분류: 초록 넓음 · 주황 좁음-경계 · 회색 좁음-기타)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, f"4_015_충격_{sh + 1}.png"), dpi=85, bbox_inches="tight"); plt.close(fig)
t015 = sub[["t_ms", "height_x", "rise_ms", "decay_ms", "f_Hz", "zeta", "fit", "edge_dist_ms"]].copy()
t015["cls"] = rs.cls.values; t015["ref_kind"] = rs.ref_kind.values
for b in BANDS: t015["d_" + b[0]] = rs["d_" + b[0]].values
t015.round(3).to_csv(os.path.join(HERE, "015_충격표_분류.csv"), index=False, encoding="utf-8-sig")
inside = sub.edge_dist_ms >= 10
R["4_015"] = dict(n=36, cls={c: int((rs.cls == c).sum()) for c in CLS}, edge_lt10=int((~inside).sum()), fit_lt_0p5=int((sub.fit < .5).sum()), fit_nan=int(sub.fit.isna().sum()),
                  ref_kind={k: int((rs.ref_kind == k).sum()) for k in ("앞", "파일")},
                  cls_inside10={c: int(((rs.cls == c) & inside).sum()) for c in CLS[:3]},
                  cls_fit_lt_0p5={c: int(((rs.cls == c) & (sub.fit < .5)).sum()) for c in CLS[:3]},
                  wide_inside_fit_ge_0p5=int(((rs.cls == "넓음") & inside & (sub.fit >= .5)).sum()),
                  wide_inside_fit_lt_0p5=int(((rs.cls == "넓음") & inside & (sub.fit < .5)).sum()),
                  border_or_other_inside=int(((rs.cls != "넓음") & inside).sum()),
                  cls_by_time_ms={c: [round(float(v), 1) for v in sub.t_ms[rs.cls == c]] for c in CLS[:3]},
                  fit_lt_0p5_list=[(round(float(a), 1), str(b)) for a, b in zip(sub.t_ms[sub.fit < .5], rs.cls[sub.fit < .5])],
                  last_impact=dict(t_ms=float(sub.t_ms.iloc[-1]), height_x=float(sub.height_x.iloc[-1]), rise_ms=float(sub.rise_ms.iloc[-1]), decay_ms=float(sub.decay_ms.iloc[-1]), cls=str(rs.cls.iloc[-1])))

# %% (5) 제안별 충격 수 변화 (정의는 고치지 않음) ----------------------------------------
print("== (5) 제안", flush=True)
RES["is_edge10"] = (RES.edge_dist_ms < 10).values
RES["is_edge20"] = (RES.edge_dist_ms < 20).values


def count_tbl(keep):
    out = {}
    for g in ("정상", "val", "고장"):
        m = RES.group == g
        per = pd.Series(keep.values & m.values).groupby(RES.file.values).sum()
        zero_files = int(sum(1 for f in per.index if grp[nall.index(f)] == g and per[f] == 0))
        out[g] = dict(before=int(m.sum()), after=int((m & keep).sum()), removed=int((m & ~keep).sum()), files_with_zero_after=zero_files)
    return out


R["5_proposals"] = {"끝단10ms제외": count_tbl(~RES.is_edge10), "끝단20ms제외": count_tbl(~RES.is_edge20), "좁음경계제외": count_tbl(RES.cls != "좁음-경계"),
                    "둘다(10ms+경계)": count_tbl(~RES.is_edge10 & (RES.cls != "좁음-경계")), "넓음만(10ms밖)": count_tbl(~RES.is_edge10 & (RES.cls == "넓음"))}
R["5_015_after"] = dict(edge10=int(((RES.file == "abn_015") & ~RES.is_edge10).sum()), border=int(((RES.file == "abn_015") & (RES.cls != "좁음-경계")).sum()),
                        both=int(((RES.file == "abn_015") & ~RES.is_edge10 & (RES.cls != "좁음-경계")).sum()), wide_only=int(((RES.file == "abn_015") & ~RES.is_edge10 & (RES.cls == "넓음")).sum()))
json.dump(R, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("완료")
