# %% [markdown]
# 4-6 사령탑 7a+ · 7g — 충격 정의 A(사건)·B(울림 충격)의 허점 확인 (라벨 무관, 고장 비교는 보고만)
# 7a+: 합성 울림을 넣고 B의 포락선 다듬기(저역 250/500/1000 Hz·없음)가 감쇠 큰 충격을 얼마나 깎는가
# 7g : A로만 잡히고 B로는 안 잡히는 사건의 정체 (에너지 대역·오르내림·B가 놓친 이유), 둔한 충돌의 빈도
# 실행: python "단계/4_피처/4-1_충격정의/자료/사령탑_7a+_7g/run.py"
import os, sys, json, warnings
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
from src.exp import load_train
from src.data import load_folder
from src.final import impact_rings, _ring_atoms, _SEG, FS, N_SAMPLES
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
RES = {}
ms = lambda v: int(round(v * FS / 1000))

HP1K = butter(4, 1000, "highpass", fs=FS, output="sos")
HP200 = butter(4, 200, "highpass", fs=FS, output="sos")
LP = {c: butter(4, c, "lowpass", fs=FS, output="sos") for c in (250, 500, 1000)}

# %% 데이터
X, y, names = load_train()
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val", "val"))
assert X.shape == (150, 50000) and Xv.shape == (40, 50000)
XA = np.vstack([X, Xv]); ids = list(names) + list(nv)
group = np.array(["train 정상" if v == 0 else "train 고장" for v in y] + ["val"] * 40)
i22 = [i for i, n in enumerate(names) if n.startswith("abn") and int(n.split("_")[-1]) == 22][0]
print("abn_022 index", i22, names[i22])

# %% 정의 B 복사 (시점을 얻으려고) + impact_rings와 같은지 assert
atoms, par = _ring_atoms()
def b_detect(x):
    xc = x - x.mean()
    x1k = sosfiltfilt(HP1K, xc)
    env = sosfiltfilt(LP[500], np.abs(hilbert(x1k)))
    pk, _ = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS))
    pk = pk[pk + _SEG <= len(x)]
    return xc, x1k, env, pk
def ring_fit(x1k, pk):
    S = np.stack([x1k[p:p + _SEG] for p in pk]); S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, atoms), axis=2); b = fit.argmax(1)
    return par[b, 0], par[b, 1], fit.max(1)
def a_detect(xc):
    """정의 A (extract_features_impact와 같음). 사건이 없으면 빈 배열 (원 함수는 argmax를 대신 넣지만 여기서는 세지 않는다)"""
    h = sosfiltfilt(HP200, xc)
    sig = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    if not len(idx): return h, sig, np.array([], int)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > int(0.01 * FS)) + 1)
    return h, sig, np.array([g[np.argmax(np.abs(h[g]))] for g in groups])

BDATA, ADATA, rows = [], [], []
for i, x in enumerate(XA):
    xc, x1k, env, pk = b_detect(x)
    BDATA.append((xc, x1k, env, pk)); ADATA.append(a_detect(xc))
    if len(pk):
        f, z, ft = ring_fit(x1k, pk); rows += [(i, a, b, c) for a, b, c in zip(f, z, ft)]
mine = np.array(rows, float).reshape(-1, 4)
ref = impact_rings(XA)
assert ref.shape == mine.shape and np.allclose(ref[:, 0], mine[:, 0]) and np.allclose(ref[:, 1:3], mine[:, 1:3]) and np.allclose(ref[:, 3], mine[:, 3], atol=1e-9), "impact_rings와 불일치"
print("B 복사본 = impact_rings 일치 (충격", len(ref), "개)")
RES["B복사_일치"] = True; RES["B충격_총수_190파일"] = int(len(ref))

# %% 7a+ 합성 시험
GN = np.array([np.sqrt(np.mean(sosfiltfilt(HP1K, X[i] - X[i].mean()) ** 2)) for i in np.flatnonzero(y == 0)])
R0 = float(np.median(GN)); RES["7a_바탕RMS_1kHz고역_train정상중앙값_g"] = R0
RINGS = [(2000, .08), (7000, .02), (8250, .01), (1500, .2), (3000, .16), (1200, .3)]
MULT = [4, 6, 8, 12]; SM = ["none", 250, 500, 1000]; NSEED = 20
tau = {r: 1000 / (r[1] * 2 * np.pi * r[0]) for r in RINGS}
def ring_wave(f, z, A, L, phase=0.0):
    t = np.arange(L) / FS; w = 2 * np.pi * f
    return A * np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z ** 2) * t + phase)
def smooth(env, c): return env if c == "none" else sosfiltfilt(LP[c], env)
W0, W1 = ms(3), ms(6)
# (a) 잡음 없는 울림 단독: 다듬기가 처음 높이를 얼마나 남기나
ret = {}
for r in RINGS:
    sig = np.zeros(ms(200)); sig[ms(100):ms(100) + ms(60)] = ring_wave(r[0], r[1], 1.0, ms(60))
    e = np.abs(hilbert(sig))
    for c in SM:
        es = smooth(e, c); ret[(r, c)] = float(es[ms(100) - W0:ms(100) + W1].max())
# (b) 잡음 속: 20 seed
rec = []; fa = {}
for seed in range(NSEED):
    rng0 = np.random.default_rng(1000 + seed)
    base = sosfiltfilt(HP1K, rng0.standard_normal(N_SAMPLES)); base *= R0 / np.sqrt(np.mean(base ** 2))
    eb = np.abs(hilbert(base)); med_raw = np.median(eb)
    for c in SM:
        es = smooth(eb, c); m = np.median(es)
        for k in (4, 6):
            fa.setdefault((c, k), []).append(len(find_peaks(es, height=k * m, distance=ms(5))[0]))
    for mult in MULT:
        rng = np.random.default_rng(seed * 10 + mult); x = base.copy()
        order = rng.permutation(len(RINGS)); t0s = {}
        for slot, ri in enumerate(order):
            t0 = int((0.1 + 0.3 * slot) * FS + rng.integers(0, ms(100)))
            f, z = RINGS[ri]; A = mult * med_raw
            x[t0:t0 + ms(40)] += ring_wave(f, z, A, ms(40), rng.uniform(0, 2 * np.pi)); t0s[ri] = t0
        e = np.abs(hilbert(x))
        for c in SM:
            es = smooth(e, c); m = np.median(es)
            for ri, t0 in t0s.items():
                rec.append(dict(seed=seed, mult=mult, sm=str(c), ring=ri, ratio=es[t0 - W0:t0 + W1].max() / m))
D7 = pd.DataFrame(rec)
rows = []
for (ri, mult, sm), g in D7.groupby(["ring", "mult", "sm"]):
    f, z = RINGS[ri]
    rows.append(dict(f_Hz=f, zeta=z, tau_ms=round(tau[RINGS[ri]], 2), A0_배=mult, 다듬기=sm, 봉우리_중앙값_평균=g.ratio.mean(), sd=g.ratio.std(),
                     남은비율_단독=ret[(RINGS[ri], int(sm) if sm != "none" else "none")], 검출률_k4=(g.ratio >= 4).mean(), 검출률_k6=(g.ratio >= 6).mean()))
T7 = pd.DataFrame(rows); T7.to_csv(os.path.join(HERE, "합성_표.csv"), index=False, encoding="utf-8-sig")
RES["7a_합성"] = {f"{r[0]}Hz_z{r[1]}": dict(tau_ms=round(tau[r], 2), 남은비율={str(c): round(ret[(r, c)], 3) for c in SM}) for r in RINGS}
RES["7a_검출률"] = {}
for r in RINGS:
    for sm in map(str, SM):
        for mult in MULT:
            g = T7[(T7.f_Hz == r[0]) & (T7.zeta == r[1]) & (T7.A0_배 == mult) & (T7.다듬기 == sm)].iloc[0]
            RES["7a_검출률"][f"{r[0]}Hz_z{r[1]}|{sm}|A0={mult}"] = dict(k4=round(float(g.검출률_k4), 2), k6=round(float(g.검출률_k6), 2), 비율=round(float(g.봉우리_중앙값_평균), 2))
RES["7a_잡음만_봉우리수_평균"] = {f"{c}|k={k}": round(float(np.mean(v)), 2) for (c, k), v in fa.items()}
with open(os.path.join(HERE, "합성_표.txt"), "w", encoding="utf-8") as fo:
    fo.write("남은 비율(잡음 없는 울림 단독, 다듬은 봉우리/처음 높이)\nring(f,z) tau_ms | none 250 500 1000\n")
    for r in RINGS: fo.write(f"{r} {tau[r]:.2f} | " + " ".join(f"{ret[(r, c)]:.3f}" for c in SM) + "\n")
    fo.write("\n잡음만 신호의 가짜 봉우리 수/파일 (다듬기|k)\n")
    for kk, v in RES["7a_잡음만_봉우리수_평균"].items(): fo.write(f"{kk}: {v}\n")
    for k in (4, 6):
        fo.write(f"\n검출률 k={k}  (행: ring, A0 | 열: none 250 500 1000)\n")
        for r in RINGS:
            for mult in MULT:
                fo.write(f"{r} A0={mult:>2} | " + " ".join(f"{RES['7a_검출률'][f'{r[0]}Hz_z{r[1]}|{sm}|A0={mult}']['k'+str(k)]:.2f}" for sm in map(str, SM)) + "\n")
    fo.write("\n봉우리/중앙값 평균 (행: ring, A0 | 열: none 250 500 1000)\n")
    for r in RINGS:
        for mult in MULT:
            fo.write(f"{r} A0={mult:>2} | " + " ".join(f"{RES['7a_검출률'][f'{r[0]}Hz_z{r[1]}|{sm}|A0={mult}']['비율']:.2f}" for sm in map(str, SM)) + "\n")

COL = {"none": "#555555", "250": "#d62728", "500": "#eb6834", "1000": "#2a78d6"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
tl = [tau[r] for r in RINGS]; order_t = np.argsort(tl); tl = np.array(tl)[order_t]
for c in SM:
    ax[0].plot(tl, [ret[(RINGS[j], c)] for j in order_t], "o-", color=COL[str(c)], label=("다듬기 없음" if c == "none" else f"저역 {c} Hz"))
for r in RINGS: ax[0].annotate(f"{r[0]/1000:g}k ζ{r[1]}", (tau[r], ret[(r, 500)]), fontsize=7, xytext=(3, -10), textcoords="offset points")
ax[0].set_xscale("log"); ax[0].set_xlabel("꺼짐 시간 τ = 1/(ζ·2πf)  (ms)"); ax[0].set_ylabel("다듬은 봉우리 / 이상적 처음 높이"); ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8)
ax[0].set_title("(a) 잡음 없는 울림 단독: 다듬기가 남기는 높이", fontsize=10)
for c in SM:
    v = [D7[(D7.ring == j) & (D7.mult == 8) & (D7.sm == str(c))].ratio.mean() for j in order_t]
    ax[1].plot(tl, v, "o-", color=COL[str(c)], label=("다듬기 없음" if c == "none" else f"저역 {c} Hz"))
ax[1].axhline(6, color="k", ls=":"); ax[1].axhline(4, color="gray", ls=":"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("꺼짐 시간 τ (ms)"); ax[1].set_ylabel("봉우리 / 신호 포락선 중앙값 (A0 = 8배, 20 seed 평균)"); ax[1].legend(fontsize=8)
ax[1].set_title("(b) 잡음 속: 문턱 6배(검은 점선)·4배(회색)와 비교", fontsize=10)
fig.savefig(os.path.join(FIG, "합성_남은높이.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 7a+ 실제 abn_022: B 충격 5개와 A 사건 10개 위치에서 다듬기별 봉우리/중앙값
xc22, x1k22, env22, pk22 = BDATA[i22]; h22, sig22, ev22 = ADATA[i22]
e22 = np.abs(hilbert(x1k22)); E22 = {"none": e22, **{c: sosfiltfilt(LP[c], e22) for c in (250, 500, 1000)}}
def real_ratio(t):
    a, b = max(t - W0, 0), min(t + W1, len(e22)); out = {}
    for c, e in E22.items(): out[str(c)] = float(e[a:b].max() / np.median(e))
    raw = e22[a:b].max()
    for c in (250, 500, 1000): out[f"남은_{c}"] = float(E22[c][a:b].max() / raw)
    return out
real = {"B충격": [dict(t_ms=round(t / FS * 1000, 1), **real_ratio(int(t))) for t in pk22], "A사건": [dict(t_ms=round(t / FS * 1000, 1), **real_ratio(int(t))) for t in ev22]}
RES["7a_abn022_실제"] = {k: [{kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in d.items()} for d in v] for k, v in real.items()}
pd.DataFrame(real["B충격"]).to_csv(os.path.join(HERE, "022_B충격_다듬기별.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(real["A사건"]).to_csv(os.path.join(HERE, "022_A사건_다듬기별.csv"), index=False, encoding="utf-8-sig")
print("022 B\n", pd.DataFrame(real["B충격"]).round(2).to_string()); print("022 A\n", pd.DataFrame(real["A사건"]).round(2).to_string())

# %% 7g A·B 짝짓기
TOL = ms(5)
pair = []
for i in range(len(XA)):
    pk = BDATA[i][3]; ev = ADATA[i][2]
    a_has = np.array([len(pk) > 0 and np.abs(pk - e).min() <= TOL for e in ev], bool)
    b_has = np.array([len(ev) > 0 and np.abs(ev - p).min() <= TOL for p in pk], bool)
    pair.append(dict(file=ids[i], group=group[i], nA=len(ev), nB=len(pk), A만=int((~a_has).sum()), 둘다_A쪽=int(a_has.sum()), B만=int((~b_has).sum()), 둘다_B쪽=int(b_has.sum())))
P = pd.DataFrame(pair)
G = P.groupby("group").agg(파일수=("file", "size"), A사건=("nA", "sum"), B충격=("nB", "sum"), A만=("A만", "sum"), 둘다_A쪽=("둘다_A쪽", "sum"), B만=("B만", "sum"), 둘다_B쪽=("둘다_B쪽", "sum"))
G["파일당_A만"] = (G.A만 / G.파일수).round(2); G["A만_있는_파일"] = P[P.A만 > 0].groupby("group").size().reindex(G.index).fillna(0).astype(int)
print(G.to_string()); RES["7g_짝짓기_집단별"] = json.loads(G.to_json(orient="index"))
print("022 짝짓기\n", P[P.file == names[i22]].to_string())

# %% 7g A만 사건 특성
LEV = ms(1); LEN = ms(9); WIN = np.hanning(LEN)
FQ = np.fft.rfftfreq(2048, 1 / FS)
def seg_pow(xc, s): return np.abs(np.fft.rfft(xc[s:s + LEN] * WIN, 2048)) ** 2
bm = {"lo": (FQ >= 200) & (FQ < 1000), "hi": (FQ >= 1000) & (FQ < 12500)}
def quiet_pows(i, n=50):
    xc = BDATA[i][0]; ev = ADATA[i][2]; pk = BDATA[i][3]; rng = np.random.default_rng(i); busy = np.concatenate([ev, pk]).astype(int); out = []
    for _ in range(4000):
        s = int(rng.integers(0, N_SAMPLES - LEN))
        if len(busy) and np.abs(busy - (s + LEV)).min() < ms(10) + LEN: continue
        out.append(seg_pow(xc, s))
        if len(out) == n: break
    return np.array(out)
def rise_decay(h, t, base):
    env = sosfiltfilt(LP[1000], np.abs(hilbert(h)))
    a, b = max(t - ms(3), 0), min(t + ms(3), len(env)); p = a + int(env[a:b].argmax()); pv = env[p]
    s = env[max(p - ms(20), 0):p + 1][::-1]
    def back(q):
        w = np.flatnonzero(s < base + q * (pv - base)); return w[0] if len(w) else np.nan
    rise = (back(.1) - back(.9)) / FS * 1000
    d = env[p:p + ms(50)]; w = np.flatnonzero(d < base + (pv - base) / np.e)
    return rise, (w[0] / FS * 1000 if len(w) else np.nan)
EVR = []; QP = {}
for i in range(len(XA)):
    xc, x1k, envB, pk = BDATA[i]; h, sig, ev = ADATA[i]
    if not len(ev): continue
    QP[i] = quiet_pows(i); qmed = {k: np.median(QP[i][:, bm[k]].sum(1)) for k in bm}
    hb = np.median(np.abs(hilbert(h))); mB = np.median(envB)
    for t in ev:
        t = int(t); s = t - LEV; row = dict(i=i, file=ids[i], group=group[i], t_ms=t / FS * 1000, A高_σ=abs(h[t]) / sig)
        row["B매칭"] = bool(len(pk) and np.abs(pk - t).min() <= TOL)
        if s >= 0 and s + LEN <= N_SAMPLES:
            P_ = seg_pow(xc, s); pw = {k: P_[bm[k]].sum() for k in bm}; tot = pw["lo"] + pw["hi"]
            row.update(몫_200_1k=pw["lo"] / tot, 몫_1k위=pw["hi"] / tot, dec_200_1k=np.log10(pw["lo"] / qmed["lo"]), dec_1k위=np.log10(pw["hi"] / qmed["hi"]))
        else: row.update(몫_200_1k=np.nan, 몫_1k위=np.nan, dec_200_1k=np.nan, dec_1k위=np.nan)
        row["오름_ms"], row["꺼짐_ms"] = rise_decay(h, t, hb)
        r = envB[max(t - ms(2), 0):t + ms(4)].max() / mB; row["B포락선비"] = r
        if row["B매칭"]: cls = "B가 잡음"
        elif t + _SEG > N_SAMPLES: cls = "iv 파일 끝 8 ms"
        elif r < 3: cls = "i 1 kHz 위 에너지 부족"
        elif r < 6: cls = "ii 6배 문턱 미달"
        else: cls = "iii 5 ms 간격 규칙 흡수"
        row["분류"] = cls; EVR.append(row)
EV = pd.DataFrame(EVR)
Aonly = EV[~EV.B매칭]
cnt = Aonly.groupby(["group", "분류"]).size().unstack(fill_value=0); print(cnt.to_string())
RES["7g_A만_분류별_집단"] = json.loads(cnt.to_json(orient="index"))
COLS = ["몫_200_1k", "몫_1k위", "dec_200_1k", "dec_1k위", "오름_ms", "꺼짐_ms", "A高_σ", "B포락선비"]
sm_ = Aonly.groupby("분류")[COLS].median().round(2); print(sm_.to_string())
RES["7g_A만_분류별_중앙값"] = json.loads(sm_.to_json(orient="index"))
mt = EV[EV.B매칭][COLS].median().round(2)
RES["7g_A와B둘다잡힌_중앙값"] = json.loads(mt.to_json()); print("둘다", mt.to_dict())

T22 = EV[EV.i == i22].copy(); T22["t_ms"] = T22.t_ms.round(1)
T22c = T22.drop(columns=["i", "file", "group"]).round(2)
T22c.to_csv(os.path.join(HERE, "022_사건표.csv"), index=False, encoding="utf-8-sig"); print(T22c.to_string())
RES["7g_abn022_사건"] = json.loads(T22c.to_json(orient="records"))
RES["7g_abn022_분류"] = T22.분류.value_counts().to_dict()

# %% 022 사건별 그림
qm22 = 10 * np.log10(np.median(QP[i22], axis=0) + 1e-30); q10 = 10 * np.log10(np.percentile(QP[i22], 10, axis=0) + 1e-30); q90 = 10 * np.log10(np.percentile(QP[i22], 90, axis=0) + 1e-30)
fm = FQ >= 100
for n_, r in enumerate(T22.itertuples()):
    t = int(ev22[n_]); fig, ax = plt.subplots(2, 1, figsize=(9, 7.8)); fig.subplots_adjust(hspace=.5); sl = slice(max(t - ms(10), 0), t + ms(10)); tt = (np.arange(len(h22))[sl] - t) / FS * 1000
    ax[0].plot(tt, h22[sl], color="#2a78d6", lw=.8, label="200 Hz 고역"); ax[0].plot(tt, x1k22[sl], color="#eb6834", lw=.8, alpha=.85, label="1 kHz 고역")
    ax[0].axvspan(-1, 8, color="gray", alpha=.1); ax[0].axvline(0, color="k", lw=.6, ls=":"); ax[0].set_xlabel("사건 기준 시간 (ms, 회색 = 스펙트럼 구간)"); ax[0].set_ylabel("g"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"abn_022 사건 {n_+1}/10  t={r.t_ms:.1f} ms  A {r.A高_σ:.1f}σ  B {'매칭' if r.B매칭 else '없음'}  → {r.분류}", fontsize=10)
    if t - LEV >= 0 and t - LEV + LEN <= N_SAMPLES:
        Pe = seg_pow(xc22, t - LEV); ax[1].semilogx(FQ[fm], 10 * np.log10(Pe[fm] + 1e-30), color="#d62728", label="사건 -1~+8 ms")
    ax[1].semilogx(FQ[fm], qm22[fm], color="#555", label="사건 없는 구간 중앙값 (50개)"); ax[1].fill_between(FQ[fm], q10[fm], q90[fm], color="gray", alpha=.25, label="10–90%")
    ax[1].axvspan(200, 1000, color="#2a78d6", alpha=.08); ax[1].axvspan(1000, 12500, color="#eb6834", alpha=.06); ax[1].set_xlim(100, 12500)
    ax[1].set_xlabel("주파수 (Hz)"); ax[1].set_ylabel("파워 (dB)"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"200–1000 Hz 몫 {r.몫_200_1k:.2f} (dec {r.dec_200_1k:.2f}) · 1 kHz 위 몫 {r.몫_1k위:.2f} (dec {r.dec_1k위:.2f}) · 오름 {r.오름_ms:.2f} ms · 꺼짐 {r.꺼짐_ms:.1f} ms · B포락선비 {r.B포락선비:.1f}", fontsize=8)
    fig.savefig(os.path.join(FIG, f"022_사건{n_+1:02d}.png"), dpi=105, bbox_inches="tight"); plt.close(fig)

# %% 둔한 충돌: 분류 i + 200–1000 Hz 몫 ≥ 0.5 (문턱은 결과를 보기 전에 정함)
DULL_Q = 0.5
EV["둔한충돌"] = (~EV.B매칭) & EV.분류.str.startswith("i ") & (EV.몫_200_1k >= DULL_Q)
EV.to_csv(os.path.join(HERE, "A사건_전체.csv"), index=False, encoding="utf-8-sig")
nd = EV.groupby("i").둔한충돌.sum(); P["둔한충돌"] = [int(nd.get(i, 0)) for i in range(len(P))]
P.to_csv(os.path.join(HERE, "AB_짝짓기_파일별.csv"), index=False, encoding="utf-8-sig")
dist = {}
for g, d in P.groupby("group"):
    vc = d.둔한충돌.clip(upper=4).value_counts().sort_index(); dist[g] = {("4+" if k == 4 else str(k)): int(v) for k, v in vc.items()}
    dist[g].update({"파일수": int(len(d)), "1개 이상 파일비율": round(float((d.둔한충돌 > 0).mean()), 3), "평균/파일": round(float(d.둔한충돌.mean()), 2), "최대": int(d.둔한충돌.max())})
RES["7g_둔한충돌_정의"] = "A만 + 분류 i(B포락선비<3) + 200–1000 Hz 몫 >= 0.5"
RES["7g_둔한충돌_파일당분포"] = dist; print(json.dumps(dist, ensure_ascii=False, indent=1))
sens = {}
for q in (0.3, 0.5, 0.7):
    m = (~EV.B매칭) & EV.분류.str.startswith("i ") & (EV.몫_200_1k >= q); c = EV[m].groupby("group").size(); sens[str(q)] = {g: int(c.get(g, 0)) for g in ("train 정상", "val", "train 고장")}
RES["7g_둔한충돌_문턱민감도_사건수"] = sens
RES["7g_둔한충돌_val_파일"] = [ids[i] for i in range(len(P)) if group[i] == "val" and P.둔한충돌[i] > 0]
RES["7g_둔한충돌_train고장_파일"] = [ids[i] for i in range(len(P)) if group[i] == "train 고장" and P.둔한충돌[i] > 0]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, g, c in zip(ax, ("train 정상", "val", "train 고장"), ("#2a78d6", "#7B1FA2", "#eb6834")):
    d = P[P.group == g].둔한충돌.clip(upper=6); a.bar(*np.unique(d, return_counts=True), color=c); a.set_title(f"{g} ({len(d)}개) — 파일당 둔한 충돌", fontsize=10); a.set_xlabel("파일당 개수 (6+ 합침)"); a.set_ylabel("파일 수")
fig.savefig(os.path.join(FIG, "둔한충돌_파일당.png"), dpi=115, bbox_inches="tight"); plt.close(fig)
fig, a = plt.subplots(figsize=(6.5, 5))
for c_, col in zip(sorted(EV.분류.unique()), ("#2a78d6", "#eb6834", "#2ca02c", "#d62728", "#999999")):
    d = EV[EV.분류 == c_]; a.scatter(d.몫_200_1k, d.B포락선비, s=14, alpha=.6, label=f"{c_} ({len(d)})", color=col)
a.axhline(3, color="k", ls=":"); a.axhline(6, color="k", ls=":"); a.axvline(DULL_Q, color="gray", ls="--"); a.set_yscale("log"); a.set_xlabel("200–1000 Hz 몫"); a.set_ylabel("B 포락선 봉우리/중앙값"); a.legend(fontsize=7)
fig.savefig(os.path.join(FIG, "A사건_몫_vs_B비.png"), dpi=115, bbox_inches="tight"); plt.close(fig)
RES["7g_A사건_총수"] = {g: int((EV.group == g).sum()) for g in ("train 정상", "val", "train 고장")}
with open(os.path.join(HERE, "result.json"), "w", encoding="utf-8") as f: json.dump(RES, f, ensure_ascii=False, indent=1, default=float)
print("완료")
