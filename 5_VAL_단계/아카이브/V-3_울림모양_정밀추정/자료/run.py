# %% [markdown]
# V-3 울림 모양 정밀 추정 (11-27)
# - 질문: 강한 충격의 울림(주파수 f · 감쇠비 ζ · 단일 모드 깨끗함 · 맥놀이)을 사전 칸 오차 없이 재면,
#         train 정상의 강타와 val 고립 강타 후보(012 · 032 · 034 · 036 · 010 · 014)가 갈리는가
# - 순서: ① 합성 신호로 추정기 검증 (기존 250 Hz × ζ 4칸 / VP 연속 최적화 / Matrix Pencil 2모드)
#         ② train 150 · val 40의 강한 충격 전체에 적용 ③ train 정상 분포(라벨 없이 정한 분위) 위에 val을 놓는다
# - 충격 검출 · 울림 창은 제출 모델과 같다(1–11 kHz 대역통과, 충격 뒤 8 ms). 창 시작만 봉우리 + 0.2 ms (필터 앞울림 회피)
# - 기준(분위)은 train 정상만으로 정한다. val 후보 이름은 표시에만 쓴다
#   conda run -n qe-phm python 5_VAL_단계/아카이브/V-3_울림모양_정밀추정/자료/run.py

# %% 설정
import os, sys, json
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt, hilbert
import src.viz  # noqa: F401 (한글 글꼴)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams["axes.unicode_minus"] = False
G = FuncFormatter(lambda v, _: f"{v:g}")                         # 로그 축 눈금을 일반 숫자로 (글꼴에 − 기호 없음)
from src.final import _ring_atoms

FS = 25000; N = int(0.008 * FS); OFF = int(0.0002 * FS)       # 8 ms 창, 봉우리 + 0.2 ms부터
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
CAND = ["val_012", "val_032", "val_034", "val_036", "val_010", "val_014"]
A250, PAR = _ring_atoms()


# %% 추정기
def basis(f, z, n=N):
    """감쇠 사인파 한 모드의 2열 기저 (sin · cos)"""
    t = np.arange(n) / FS; w = 2 * np.pi * f
    a = np.exp(-z * w * t); wd = w * np.sqrt(max(1 - z * z, 1e-6))
    return np.column_stack([a * np.sin(wd * t), a * np.cos(wd * t)])


def proj_fit(B, s):
    """정규화 신호 s를 기저 B에 투영한 크기 (0–1, 기존 fit과 같은 정의)"""
    Q, _ = np.linalg.qr(B); return float(np.linalg.norm(Q.T @ (s / np.linalg.norm(s))))


def est_grid(s):
    """기존: 1–12 kHz 250 Hz × ζ {0.01, 0.02, 0.04, 0.08}"""
    s = s[:N] / np.linalg.norm(s[:N]); fit = np.linalg.norm(np.einsum("s,ast->at", s, A250), axis=1); k = fit.argmax()
    return dict(f=PAR[k, 0], z=PAR[k, 1], fit=float(fit[k]))


ZF = np.geomspace(0.003, 0.3, 40)
def est_vp(s, n=None, f_init=None):
    """VP(변수 투영): 기존 사전(또는 f_init) → 주변 ±300 Hz(20 Hz) × ζ 40칸 → (f, ln ζ) 연속 최적화. 진폭 · 위상은 매번 최소제곱
    n: 창 길이(점, 기본 N = 8 ms). f_init: 처음 주파수(없으면 기존 8 ms 사전으로 찾음 — 이때 s는 N점 이상)"""
    n = N if n is None else n
    f0 = est_grid(s[:N])["f"] if f_init is None else f_init
    s = s[:n] / np.linalg.norm(s[:n])
    best = max(((f, z) for f in np.arange(max(600, f0 - 300), f0 + 301, 20) for z in ZF), key=lambda p: proj_fit(basis(*p, n), s))
    def res(p):
        B = basis(p[0], np.exp(p[1]), n); c = np.linalg.lstsq(B, s, rcond=None)[0]; return s - B @ c
    r = least_squares(res, [best[0], np.log(best[1])], bounds=([300, np.log(1e-3)], [12500, np.log(0.5)]), x_scale=[50, .1])
    f, z = r.x[0], float(np.exp(r.x[1]))
    return dict(f=f, z=z, fit=proj_fit(basis(f, z, n), s))


def est_mp(s, modes=2):
    """Matrix Pencil: 복소 극점 2·modes개 → 양의 주파수 모드(f, ζ, 진폭) 큰 순. 2모드 fit = 두 모드 기저 투영"""
    s = s[:N] / np.linalg.norm(s[:N]); n = len(s); L = n // 3; M = 2 * modes
    Y = np.lib.stride_tricks.sliding_window_view(s, L + 1)
    _, _, Vh = np.linalg.svd(Y, full_matrices=False); V = Vh[:M].T
    zs = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    zs = zs[(np.angle(zs) > 2 * np.pi * 300 / FS) & (np.abs(zs) > 1e-3)]
    if len(zs) == 0:
        return dict(f1=np.nan, z1=np.nan, f2=np.nan, z2=np.nan, ratio=np.nan, fit2=np.nan)
    lam = np.log(zs) * FS; f = np.abs(lam.imag) / (2 * np.pi); z = np.clip(-lam.real / np.abs(lam), 1e-4, 0.99)
    k = np.arange(n)[:, None]; Vd = zs[None, :] ** k
    amp = np.abs(np.linalg.lstsq(np.hstack([Vd, Vd.conj()]), s.astype(complex), rcond=None)[0][:len(zs)])
    o = np.argsort(-amp); f, z, amp = f[o], z[o], amp[o]
    B = np.hstack([basis(fi, zi) for fi, zi in zip(f[:modes], z[:modes])])
    return dict(f1=f[0], z1=z[0], f2=f[1] if len(f) > 1 else np.nan, z2=z[1] if len(f) > 1 else np.nan,
                ratio=amp[1] / amp[0] if len(f) > 1 else 0.0, fit2=proj_fit(B, s))


def est_env(x, p, f):
    """교차 확인: f ± 600 Hz 대역 포락선의 ln 기울기 → ζ_env, 직선 R²(순수 지수 감쇠에 가까울수록 1)"""
    sos = butter(4, (max(200, f - 600), min(12000, f + 600)), btype="bandpass", fs=FS, output="sos")
    seg = x[max(0, p - 1500):p + 2 * N]; e = np.abs(hilbert(sosfiltfilt(sos, seg))); k0 = min(1500, p)
    base = np.median(e[:max(10, k0 - 250)]) + 1e-12; k = k0 + int(np.argmax(e[k0:k0 + 50])); end = k + 1
    while end < len(e) and end < k + 2 * N and e[end] > 4 * base:
        end += 1
    le = np.log(e[k:end] + 1e-12)
    if len(le) < 10:
        return np.nan, np.nan
    t = np.arange(len(le)) / FS; b, a = np.polyfit(t, le, 1); r = le - (a + b * t)
    return -b / (2 * np.pi * f), 1 - r.var() / le.var()


# %% ① 합성 검증
rng = np.random.default_rng(0); rows = []
for f0 in (1830, 3110, 7125, 9060):
    for z0 in (0.005, 0.01, 0.02, 0.04, 0.08):
        for snr in (30, 10):                                   # 충격 진폭 / 잡음 표준편차 (실제 충격 배수 수십 배 ↔ 약한 충격)
            for rep in range(8):
                t = np.arange(N) / FS; w = 2 * np.pi * f0
                s = np.exp(-z0 * w * t) * np.sin(w * np.sqrt(1 - z0 ** 2) * t + rng.uniform(0, 2 * np.pi)) + rng.standard_normal(N) / snr
                g, v, m = est_grid(s), est_vp(s), est_mp(s, 1)
                rows.append(dict(f0=f0, z0=z0, snr=snr, grid=g["z"], vp=v["z"], mp=m["z1"], fvp=v["f"], fmp=m["f1"], fitg=g["fit"], fitv=v["fit"]))
SY = pd.DataFrame(rows)
for c in ("grid", "vp", "mp"):
    SY[c + "_err"] = np.abs(SY[c] / SY.z0 - 1)
syn = SY.groupby("snr")[["grid_err", "vp_err", "mp_err"]].median().round(3).to_dict()
syn_z = SY.groupby(["snr", "z0"])[["grid_err", "vp_err", "mp_err"]].median().round(3)
# 맥놀이 합성: 두 모드 (Δf, 진폭비) → MP 2모드가 되찾는가
beat = []
for df_, r_ in [(250, .8), (500, .5), (1000, .7)]:
    t = np.arange(N) / FS
    s = sum(c * np.exp(-0.01 * 2 * np.pi * f * t) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f, c in [(7000, 1), (7000 + df_, r_)]) + rng.standard_normal(N) / 30
    v, m = est_vp(s), est_mp(s, 2)
    beat.append(dict(df=df_, ratio=r_, vp_f=round(v["f"]), vp_z=round(v["z"], 4), vp_fit=round(v["fit"], 3),
                     mp_f=[round(m["f1"]), round(m["f2"])], mp_z=[round(m["z1"], 4), round(m["z2"], 4)], mp_ratio=round(m["ratio"], 2), fit2=round(m["fit2"], 3)))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for a, snr in zip(axs, (30, 10)):
    d = SY[SY.snr == snr]
    for c, lab, mk in [("grid", "기존 사전(ζ 4칸)", "s"), ("vp", "VP 연속 최적화", "o"), ("mp", "Matrix Pencil", "^")]:
        q = d.groupby("z0")[c].median(); a.plot(q.index, q.values, mk + "-", label=lab, alpha=.85)
    a.plot([.004, .1], [.004, .1], "k:", lw=1, label="참값")
    a.set_xscale("log"); a.set_yscale("log"); a.xaxis.set_major_formatter(G); a.yaxis.set_major_formatter(G); a.set_xlabel("참 ζ"); a.set_title(f"잡음: 충격 진폭 / 잡음 = {snr}", fontsize=10); a.grid(alpha=.3)
axs[0].set_ylabel("추정 ζ (주파수 4개 × 위상 8번의 중앙값)"); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "합성검증.png"), dpi=130); plt.close(fig)

# %% ② 실제 강한 충격 전체
Itr, Iva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_imps.joblib"))
_, _, y, ntr, nva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))
hx_all = np.concatenate([np.asarray(im["height_x"]) for im, yy in zip(Itr, y) if yy == 0])
HX_MIN = float(np.percentile(hx_all, 75))                       # 강한 충격 = train 정상 충격 크기의 75 % 분위 이상 (라벨 없이 정함)
R = []
for src_, I, names, lab in [("train", Itr, ntr, None), ("val", Iva, nva, "val")]:
    for i, im in enumerate(I):
        g = lab or ("train 정상" if y[i] == 0 else "train 고장"); x = im["hp"]
        for p, hx in zip(im["idx"], im["height_x"]):
            if hx < HX_MIN or p + OFF + N > len(x):
                continue
            s = x[p + OFF:p + OFF + N]; v = est_vp(s); m = est_mp(s, 2); ze, r2 = est_env(x, int(p), v["f"])
            R.append(dict(file=names[i], 그룹=g, t=round(p / FS, 4), hx=round(float(hx), 1), f=v["f"], z=v["z"], fit=v["fit"],
                          f2=m["f2"], z2=m["z2"], ratio=m["ratio"], fit2=m["fit2"], gain=m["fit2"] - v["fit"],
                          df=abs(m["f2"] - m["f1"]), z_env=ze, r2_env=r2))
T = pd.DataFrame(R); T.to_csv(os.path.join(HERE, "강한충격표.csv"), index=False, encoding="utf-8-sig")

# %% ③ train 정상 분포 위에 val
nrm = T[T.그룹 == "train 정상"]
Q = {k: float(np.percentile(nrm[k].dropna(), q)) for k, q in [("fit", 95), ("fit", 99), ("gain", 5), ("z", 95), ("z", 5)] for k in [k]}
Q = {f"fit_q95": float(np.percentile(nrm.fit, 95)), "fit_q99": float(np.percentile(nrm.fit, 99)),
     "gain_q05": float(np.percentile(nrm.gain, 5)), "z_q05": float(np.percentile(nrm.z, 5)), "z_q95": float(np.percentile(nrm.z, 95))}
hi = lambda d: d[(d.f >= 5500) & (d.f < 8500)]                 # 6–7 kHz 강타 비교용 (5.5–8.5 kHz)
summ = {}
for g, d in T.groupby("그룹"):
    summ[g] = dict(강한충격=len(d), 파일=int(d.file.nunique()), fit_q95초과=int((d.fit > Q["fit_q95"]).sum()),
                   fit_q99초과=int((d.fit > Q["fit_q99"]).sum()),
                   고주파_5_5_8_5k=len(hi(d)), 고주파_fit_q95초과=int((hi(d).fit > Q["fit_q95"]).sum()),
                   고주파_fit_q95초과_파일=sorted(hi(d)[hi(d).fit > Q["fit_q95"]].file.unique().tolist()))
COL = {"train 정상": "C0", "train 고장": "C1", "val": "C4"}
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
for g in ["train 정상", "train 고장", "val"]:
    d = T[T.그룹 == g]; kw = dict(s=10 + d.hx.clip(upper=200) / 3, alpha=.45 if g != "val" else .7, color=COL[g], label=g)
    axs[0].scatter(d.f / 1000, d.z, **kw); axs[1].scatter(d.fit, d.gain, **kw); axs[2].scatter(d.f / 1000, d.fit, **kw)
for c in CAND:
    d = T[T.file == c]
    if len(d) == 0:
        continue
    r = d.loc[d.hx.idxmax()]
    for a, (xx, yy) in zip(axs, [(r.f / 1000, r.z), (r.fit, r.gain), (r.f / 1000, r.fit)]):
        a.scatter(xx, yy, s=70, facecolor="none", edgecolor="k", lw=1); a.annotate(c[4:], (xx, yy), fontsize=8, xytext=(4, 3), textcoords="offset points")
axs[0].set_yscale("log"); axs[0].yaxis.set_major_formatter(G); axs[0].set_xlabel("울림 주파수 f (kHz, VP)"); axs[0].set_ylabel("감쇠비 ζ (VP, 로그)")
axs[1].set_xlabel("단일 모드 fit (VP)"); axs[1].set_ylabel("2모드 fit 증가 (MP - VP)")
axs[1].axvline(Q["fit_q95"], color="C0", ls="--", lw=1); axs[1].text(Q["fit_q95"], axs[1].get_ylim()[1] * .95, " 정상 95%", color="C0", fontsize=8, va="top")
axs[2].set_xlabel("울림 주파수 f (kHz)"); axs[2].set_ylabel("단일 모드 fit"); axs[2].axhline(Q["fit_q95"], color="C0", ls="--", lw=1)
for a in axs:
    a.grid(alpha=.3)
axs[0].legend(fontsize=8, loc="lower left")
fig.suptitle(f"강한 충격(train 정상 충격 크기 75 % 분위 {HX_MIN:.1f}배 이상) · 점 크기 = 충격 배수 · ○ = val 고립 강타 후보의 가장 큰 충격", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "울림분포.png"), dpi=130); plt.close(fig)

# 후보 · 고립 강타 파일의 가장 큰 충격 표
iso = [d["file"] for d in json.load(open(os.path.join(ROOT, "5_VAL_단계", "V-2_약한고장_파형확인", "자료", "고립강타_목록.json"), encoding="utf-8"))]
top = T[T.file.isin(iso + ["val_039", "val_016"])].sort_values("hx").groupby("file").tail(1).sort_values(["그룹", "file"])
top_tab = top[["file", "그룹", "hx", "f", "z", "fit", "f2", "ratio", "gain", "z_env", "r2_env"]].round(
    dict(f=0, z=4, fit=3, f2=0, ratio=2, gain=3, z_env=4, r2_env=2))
from src.exp import md_table
open(os.path.join(HERE, "고립강타_정밀.md"), "w", encoding="utf-8").write(md_table(top_tab))
json.dump(dict(합성_ζ상대오차_중앙=syn, 합성_ζ별={str(k): v for k, v in syn_z.to_dict(orient="index").items()}, 맥놀이합성=beat,
               강한충격_기준배수=HX_MIN, train정상_분위=Q, 그룹요약=summ),
          open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("done")

# %% ④ ζ 신뢰성 검증 (11-27 추가)
# 논리: 지수 감쇠 하나라면 (a) 시작점을 늦춰도 · 창을 바꿔도 ζ는 같아야 하고 (b) 독립된 두 방법(VP · MP)이 같은 값을 내야 하며
#       (c) 같은 파일 · 같은 모드의 반복 충격은 비슷한 ζ를 내야 한다. (d) 실제 처리 경로(대역통과 · 검출 봉우리 시작 · 실제 바탕 잡음 · 약한 2번째 모드)를 넣은 합성에서 참값을 되찾아야 한다
from src.impacts import highpass
def vp_at(x, p, off_ms, win_ms, f_init):
    """봉우리 + off_ms부터 win_ms 창의 VP ζ (처음 주파수는 기준 추정값으로 고정 → 같은 모드를 따라간다)"""
    o, n = int(off_ms * FS / 1000), int(win_ms * FS / 1000)
    s = x[p + o:p + o + n]
    if len(s) < n:
        return np.nan, np.nan
    v = est_vp(s, n=n, f_init=f_init)
    return v["z"], v["fit"]

# (d) 현실 합성: 실제 train 정상 바탕(충격 없는 구간)에 감쇠 사인파를 넣고 → 1–11 kHz 대역통과 → 검출기와 같은 포락선 봉우리 → +0.2 ms부터 VP
from scipy.signal import butter as _b
sos_bp = _b(4, (1000, 11000), btype="bandpass", fs=FS, output="sos"); sos_lp = _b(4, 1000, btype="lowpass", fs=FS, output="sos")
Xtr_raw = None
quiet = [im["hp"] for im, yy in zip(Itr, y) if yy == 0 and len(im["idx"]) == 0][:20]   # 충격 0개 정상 파일의 1–11 kHz 신호 = 실제 바탕
syn2 = []
for f0 in (1900, 6800, 7150):
    for z0 in (0.01, 0.02, 0.033, 0.08):
        for second in (0.0, 0.3):
            for rep in range(6):
                bg = quiet[rep % len(quiet)][5000:5000 + 2000].copy()
                t = np.arange(1500) / FS; w = 2 * np.pi * f0; amp = 30 * bg.std()          # 바탕 대비 약 30배 충격
                sig = amp * np.exp(-z0 * w * t) * np.sin(w * np.sqrt(1 - z0 ** 2) * t)
                if second:
                    w2 = 2 * np.pi * (f0 + 400); sig += second * amp * np.exp(-z0 * w2 * t) * np.sin(w2 * t + 1.0)
                raw = bg.copy(); raw[300:1800] += sig
                xh = sosfiltfilt(sos_bp, raw); e = sosfiltfilt(sos_lp, np.abs(hilbert(xh)))
                p = 300 + int(np.argmax(e[300:400]))                                            # 검출기처럼 포락선 봉우리
                v = est_vp(xh[p + OFF:p + OFF + N]); m = est_mp(xh[p + OFF:p + OFF + N], 2)
                syn2.append(dict(f0=f0, z0=z0, second=second, vp=v["z"], fit=v["fit"], mp=m["z1"], lag_ms=(p - 300) / FS * 1000))
S2 = pd.DataFrame(syn2); S2["vp_err"] = S2.vp / S2.z0 - 1; S2["mp_err"] = S2.mp / S2.z0 - 1
syn2_tab = S2.groupby(["second", "f0", "z0"]).agg(vp_err=("vp_err", "median"), mp_err=("mp_err", "median"), fit=("fit", "median"), lag_ms=("lag_ms", "median")).round(3)

# (a)(b) 실제 강한 충격: 시작점 0.2 · 0.6 · 1.2 ms, 창 4 · 8 ms, VP vs MP
Iall = {n: im for n, im in zip(ntr, Itr)} | {n: im for n, im in zip(nva, Iva)}
stab = []
for r in T.itertuples():
    x = Iall[r.file]["hp"]; p = int(round(r.t * FS))
    z_a, _ = vp_at(x, p, 0.6, 8, r.f); z_b, _ = vp_at(x, p, 1.2, 8, r.f); z_c, _ = vp_at(x, p, 0.2, 4, r.f)
    m1 = est_mp(x[p + OFF:p + OFF + N], 1)["z1"]
    stab.append(dict(file=r.file, 그룹=r.그룹, fit=r.fit, z=r.z, z_off06=z_a, z_off12=z_b, z_win4=z_c, z_mp1=m1))
ST = pd.DataFrame(stab)
lr = lambda a, b: np.abs(np.log(a / b))                             # 비율 오차 (0.1 ≈ 10 %)
ST["d_off06"] = lr(ST.z_off06, ST.z); ST["d_off12"] = lr(ST.z_off12, ST.z); ST["d_win4"] = lr(ST.z_win4, ST.z); ST["d_mp"] = lr(ST.z_mp1, ST.z)
ST["fit_bin"] = pd.cut(ST.fit, [0, .6, .75, .9, 1.01], labels=["<0.6", "0.6–0.75", "0.75–0.9", "≥0.9"])
stab_tab = ST.groupby("fit_bin", observed=True)[["d_off06", "d_off12", "d_win4", "d_mp"]].median().round(3)
stab_tab["n"] = ST.groupby("fit_bin", observed=True).size()

# (c) 같은 파일 · 같은 모드(±150 Hz)의 반복 충격 ζ 흩어짐 (fit ≥ 0.75)
rep_rows = []
for fn, d in T[T.fit >= .75].groupby("file"):
    d = d.sort_values("f")
    for f_c in d.f.unique():
        g = d[(d.f - f_c).abs() <= 150]
        if len(g) >= 3:
            rep_rows.append(dict(file=fn, 그룹=g.그룹.iloc[0], f=round(g.f.median()), n=len(g), z_med=round(g.z.median(), 4),
                                 z_cv=round(g.z.std() / g.z.mean(), 2)))
            break
REP = pd.DataFrame(rep_rows)

# 후보 파일의 가장 큰 충격: 시작점 · 창 · 방법별 ζ
cand_tab = ST.merge(T[["file", "t", "hx"]].reset_index(drop=True), left_index=True, right_index=True, suffixes=("", "_t"))
cand_tab = cand_tab[cand_tab.file.isin(iso + ["val_039", "val_016"])].sort_values("hx").groupby("file").tail(1)
cand_tab = cand_tab[["file", "그룹", "hx", "fit", "z", "z_off06", "z_off12", "z_win4", "z_mp1"]].round(4).sort_values(["그룹", "file"])

with open(os.path.join(HERE, "검증_감쇠비.md"), "w", encoding="utf-8") as fh:
    fh.write("## (d) 현실 합성: 상대오차 중앙값 (second = 400 Hz 위 2번째 모드 진폭비)\n\n" + md_table(syn2_tab.reset_index()) + "\n\n")
    fh.write("## (a)(b) 실제 강한 충격: 기준(VP, +0.2 ms · 8 ms) 대비 |ln 비율| 중앙값, fit 구간별\n\n" + md_table(stab_tab.reset_index()) + "\n\n")
    fh.write("## (c) 같은 파일 · 같은 모드 반복 충격 (fit ≥ 0.75, 3개 이상)\n\n" + md_table(REP) + "\n\n")
    fh.write("## 고립 강타 파일의 가장 큰 충격: 시작점 · 창 · 방법별 ζ\n\n" + md_table(cand_tab))
print("verify done")
