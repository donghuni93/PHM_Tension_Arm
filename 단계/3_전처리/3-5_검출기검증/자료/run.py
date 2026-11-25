# %% 2-36 충격 검출기 검증 (11-25)
# 대상: src/impacts.py detect_impacts · impact_rings_v2, 입력 = 1–11 kHz 대역통과(11-25 사용자 결정, _hp=로 넣음).
# ① 위상 무작위 대용신호(PSD 유지, 충격성 제거)로 오검출률  ② 실제 정상 파일 조용한 구간에 감쇠 사인파 주입 → 검출률 · 시점 · 울림 주파수 오차
# ③ 상수 민감도(라벨 없는 통계만)  ④ val 3–5 kHz 바탕 영향 · 한 충격 다중 검출 · 예전 검출기(final.impact_rings) 비교
# 라벨은 그룹 집계와 '참고' AUC 한 줄에만 쓴다. 상수를 라벨로 고르지 않는다. val 라벨은 없다(추정하지 않음).
import os, sys, time
import numpy as np, pandas as pd
import matplotlib, matplotlib.ticker; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, welch
from scipy.ndimage import median_filter
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
from src.impacts import detect_impacts, impact_rings_v2, _atoms50, FS
from src.final import impact_rings, _SEG
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
T0 = time.time()

SOS_BP = butter(4, [1000, 11000], btype="bandpass", fs=FS, output="sos")
SOS_HP = butter(4, 1000, btype="highpass", fs=FS, output="sos")
LP = {h: butter(4, h, btype="lowpass", fs=FS, output="sos") for h in (500, 1000, 2000)}
DEF = dict(k=4.5, base=100, sm=1000, dist=6.75)
VAR = {"기본": DEF,
       "k3.5": {**DEF, "k": 3.5}, "k4": {**DEF, "k": 4.0}, "k6": {**DEF, "k": 6.0},
       "바탕50ms": {**DEF, "base": 50}, "바탕200ms": {**DEF, "base": 200},
       "다듬기500": {**DEF, "sm": 500}, "다듬기2000": {**DEF, "sm": 2000},
       "간격3.4ms": {**DEF, "dist": 3.375}, "간격13.5ms": {**DEF, "dist": 13.5}}
EDGE = int(0.010 * FS)
ms = lambda v: int(round(v * FS / 1000))
A50, F50 = _atoms50(); ZS = (0.01, 0.02, 0.04, 0.08)


def filt(x, sos):
    x = np.asarray(x, float); return sosfiltfilt(sos, x - x.mean())


def baseline(e, win_ms):   # src.impacts._baseline과 같은 식, 창만 바꿈
    dec = FS // 1000; m = median_filter(e[::dec], size=int(round(win_ms)) | 1, mode="reflect")
    return np.interp(np.arange(len(e)), np.arange(0, len(e), dec), m)


def detect_all(xh, variants=VAR):
    """필터된 신호 → {변형 이름: 충격 표본 번호}, 기본 바탕(1000 Hz 다듬기 · 100 ms)"""
    a = np.abs(hilbert(xh)); env, base, out = {}, {}, {}
    for nm, v in variants.items():
        if v["sm"] not in env: env[v["sm"]] = sosfiltfilt(LP[v["sm"]], a)
        key = (v["sm"], v["base"])
        if key not in base: base[key] = baseline(env[v["sm"]], v["base"])
        pk = find_peaks(env[v["sm"]] / base[key], height=v["k"], distance=max(1, ms(v["dist"])))[0]
        out[nm] = pk[(pk >= EDGE) & (pk < len(xh) - EDGE)]
    b = base.get((1000, 100)); return out, (b if b is not None else baseline(sosfiltfilt(LP[1000], a), 100))


def old_detect(xhp):       # src.final.impact_rings 와 같은 식 (1 kHz 고역 → 500 Hz 다듬기 → 6 × 파일 중앙값, 5 ms)
    env = sosfiltfilt(LP[500], np.abs(hilbert(xhp)))
    pk = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS))[0]
    return pk[pk + _SEG <= len(xhp)]


def ring50(xh, idx):       # 50 Hz 사전(2-6 · 2-7 fine_ring과 같음)
    S = np.stack([xh[p:p + _SEG] for p in idx]); S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, A50), axis=2); b = fit.argmax(1)
    return F50[b], np.array(ZS)[b % 4]


def ring250(xh, idx):      # impact_rings_v2 (250 Hz 사전) 그대로
    return impact_rings_v2(None, imp={"idx": np.asarray(idx)}, _hp=xh)


def surrogate(x, rng):
    X = np.fft.rfft(x - x.mean()); ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0; ph[-1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


# %% 데이터
names = pd.read_csv(os.path.join(ROOT, "단계", "아카이브", "new_단계", "2_EDA", "2-6_바닥보정·이동·기준ET", "figs", "files_190.csv"))
PAT = {"normal": "TeamProject_train_변경/normal", "abnormal": "TeamProject_train_변경/abnormal", "val": "TeamProject_val/val"}
X = np.stack([np.loadtxt(os.path.join(ROOT, "data", PAT[g], n + ".csv")) for n, g in names.itertuples(index=False)])
files, grp = names.file.values, names.group.values
L = [f"2-36 충격 검출기 검증 — 파일 {len(files)}개 (정상 {sum(grp == 'normal')} · 고장 {sum(grp == 'abnormal')} · val {sum(grp == 'val')})"]

# 재현 확인: detect_all 기본 = detect_impacts(_hp=대역통과), old_detect = final.impact_rings
chk = all(np.array_equal(detect_all(filt(X[i], SOS_BP))[0]["기본"], detect_impacts(X[i], _hp=filt(X[i], SOS_BP))["idx"]) for i in range(0, 190, 19))
R_old = impact_rings(X); n_old_ref = np.bincount(R_old[:, 0].astype(int), minlength=len(X))
chk2 = all(len(old_detect(filt(X[i], SOS_HP))) == n_old_ref[i] for i in range(0, 190, 19))
L.append(f"재현 확인: 기본 변형 = detect_impacts(_hp=1–11k) {chk} · old_detect = final.impact_rings 충격 수 {chk2} (10개 표본 파일)")


# %% ① 실제 파일 검출 + 대용신호 오검출
NS = 10   # 파일당 대용신호 수


def real_job(i, x):
    xb, xh = filt(x, SOS_BP), filt(x, SOS_HP)
    d, b = detect_all(xb); dh, _ = detect_all(xh, {"기본": DEF}); do = old_detect(xh)
    f, P = welch(x - x.mean(), FS, nperseg=4096)
    bp = lambda lo, hi: P[(f >= lo) & (f < hi)].sum() * (f[1] - f[0])
    row = dict(i=i, base_med=float(np.median(b)), p35=bp(3000, 5000), p111=bp(1000, 11000),
               **{f"n_{k}": len(v) for k, v in d.items()}, n_hp=len(dh["기본"]), n_old=len(do))
    imps = []
    for det, xx, idx in (("bp", xb, d["기본"]), ("hp", xh, dh["기본"]), ("old", xh, do)):
        idx = np.array([p for p in idx if p + _SEG <= len(xx)], int)
        if len(idx):
            r = ring250(xx, idx); f5, _ = ring50(xx, idx)
            imps += [dict(i=i, det=det, idx=int(p), f=r[k, 0], z=r[k, 1], fit=r[k, 2], f50=f5[k]) for k, p in enumerate(idx)]
    sur = []
    rng = np.random.default_rng(1000 + i)
    for s in range(NS):
        xs = surrogate(x, rng)
        ds, _ = detect_all(filt(xs, SOS_BP)); xsh = filt(xs, SOS_HP)
        sur.append(dict(i=i, s=s, **{f"n_{k}": len(v) for k, v in ds.items()},
                        n_hp=len(detect_all(xsh, {"기본": DEF})[0]["기본"]), n_old=len(old_detect(xsh))))
    return row, imps, sur


import pickle
CACHE = os.path.join(HERE, "_cache"); os.makedirs(CACHE, exist_ok=True)   # 재실행 시간 절약용 임시 캐시 (지워도 됨)
def cached(name, fn):
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    r = fn(); pickle.dump(r, open(p, "wb")); return r
res = cached("real", lambda: Parallel(n_jobs=30)(delayed(real_job)(i, X[i]) for i in range(len(X))))
RF = pd.DataFrame([r[0] for r in res]); RF["file"] = files; RF["group"] = grp
IMP = pd.DataFrame([m for r in res for m in r[1]]); SUR = pd.DataFrame([m for r in res for m in r[2]]); SUR["group"] = grp[SUR.i]
RF.to_csv(os.path.join(FIG, "files_190_검출.csv"), index=False); IMP.to_csv(os.path.join(FIG, "impacts_190.csv"), index=False)
L.append(f"[시간] ① {time.time() - T0:.0f} s")

DETS = [f"n_{k}" for k in VAR] + ["n_hp", "n_old"]
DLAB = {**{f"n_{k}": f"1–11k {k}" for k in VAR}, "n_hp": "1k 고역만(기본 상수)", "n_old": "예전(final.impact_rings)"}
L.append(f"\n[1] 위상 무작위 대용신호 오검출 (파일 190 × 대용 {NS} = {len(SUR)}개). 값 = 대용신호 1개당 평균 검출 수 · 1개 이상 나온 비율 · 최대")
L.append(f"{'검출기':24s} | 전체 평균 | ≥1 비율 | 최대 | 정상 | 고장 | val | (실제 파일 평균 검출 수 정상/고장/val)")
for c in DETS:
    g = SUR.groupby("group")[c].mean(); r = RF.groupby("group")[c].mean()
    L.append(f"{DLAB[c]:24s} | {SUR[c].mean():.3f} | {(SUR[c] > 0).mean():.3f} | {SUR[c].max()} | {g['normal']:.3f} | {g['abnormal']:.3f} | {g['val']:.3f} | "
             f"({r['normal']:.1f}/{r['abnormal']:.1f}/{r['val']:.1f})")
top = SUR.groupby("i")["n_기본"].mean().sort_values(ascending=False).head(6)
L.append("기본(1–11k) 대용신호 오검출이 많은 파일: " + ", ".join(f"{files[i]} {v:.1f}" for i, v in top.items()))

# 참고(라벨 사용, 상수 선택에 쓰지 않음): 학습 150개에서 충격 수의 정상/고장 AUC
from sklearn.metrics import roc_auc_score
tr = RF[RF.group != "val"]
L.append("\n[참고 · 라벨 사용 · 상수 선택에 쓰지 않음] 학습 150개 충격 수 AUC(고장 > 정상): " +
         ", ".join(f"{DLAB[c]} {roc_auc_score(tr.group == 'abnormal', tr[c]):.3f}" for c in DETS))


# %% ② 주입: 정상 120개(+ val 40개는 바탕 점검용)의 조용한 시점에 감쇠 사인파 추가
FREQS = (1900, 3000, 7000, 8300, 9500); ZETAS = (0.01, 0.02, 0.04); MULTS = (1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 13, 16, 20)
MIX = ((1900, 0.01), (7000, 0.02), (9500, 0.04))     # 여러 모드가 함께 우는 충격(같은 최대 진폭으로 합침)
GRID = (np.arange(0.2, 1.81, 0.2) * FS).astype(int); NP_MAX = 6
PL = int(0.030 * FS); PAD = int(0.060 * FS)


def pulse(comps):
    t = np.arange(PL) / FS; s = np.zeros(PL)
    for f, z in comps:
        w = 2 * np.pi * f; c = np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z ** 2) * t); s += c / np.abs(c).max()
    s /= np.abs(s).max()
    buf = np.zeros(PL + 2 * PAD); buf[PAD:PAD + PL] = s           # 선형 필터라 펄스만 따로 걸러 더한다
    return sosfiltfilt(SOS_BP, buf), sosfiltfilt(SOS_HP, buf)


PULSE = {}
for f in FREQS:
    for z in ZETAS: PULSE[(f, z)] = pulse(((f, z),))
PULSE[("mix", 0)] = pulse(MIX)
MED_BASE = None   # 정상 파일 바탕 중앙값(g) — 절대 진폭 주입의 단위, 아래에서 채움


def add(xf, pf, pos, amps):
    y = xf.copy()
    for p, a in zip(pos, amps):
        lo = p - PAD; y[lo:lo + len(pf)] += a * pf
    return y


def inject_job(i, x, combos, med_base):
    xb0, xh0 = filt(x, SOS_BP), filt(x, SOS_HP)
    d0, b0 = detect_all(xb0, {"기본": DEF}); dh0 = detect_all(xh0, {"기본": DEF})[0]["기본"]; do0 = old_detect(xh0)
    allpk = np.concatenate([d0["기본"], dh0, do0])
    pos = [p for p in GRID if not np.any(np.abs(allpk - p) < ms(40))][:NP_MAX]
    if not pos: return []
    rows = []
    for key, mode, val in combos:
        pb, ph = PULSE[key]
        amps = [val * b0[p] if mode == "rel" else val * med_base for p in pos]
        xb, xh = add(xb0, pb, pos, amps), add(xh0, ph, pos, amps)
        d, _ = detect_all(xb); dh = detect_all(xh, {"기본": DEF})[0]["기본"]; do = old_detect(xh)
        for p, a in zip(pos, amps):
            r = dict(i=i, f=key[0], z=key[1], kind=mode, val=val, pos=p, amp=a, mult=a / b0[p], b_loc=b0[p])
            for nm, idx in list(d.items()) + [("hp", dh), ("old", do)]:
                h = idx[(idx >= p - ms(3)) & (idx <= p + ms(10))]
                r[f"hit_{nm}"] = len(h) > 0
                r[f"multi_{nm}"] = int(((idx >= p - ms(3)) & (idx <= p + ms(25))).sum())
                if nm == "기본" and len(h):
                    q = h[np.argmin(np.abs(h - p))]; r["dt_ms"] = (q - p) / FS * 1000
                    rr = ring250(xb, [q]); f5, z5 = ring50(xb, [q])
                    r.update(f250=rr[0, 0], z250=rr[0, 1], fit=rr[0, 2], f50=f5[0], z50=z5[0])
            rows.append(r)
    return rows


MED_BASE = float(RF.loc[RF.group == "normal", "base_med"].median())
C_REL = [((f, z), "rel", m) for f in FREQS for z in ZETAS for m in MULTS] + [(("mix", 0), "rel", m) for m in MULTS]
ABS_C = (2, 3, 4, 5, 6, 8, 10, 14)
C_BG = [((f, 0.02), "abs", c) for f in (3000, 7000) for c in ABS_C]
C_VAL = [((f, 0.02), "rel", m) for f in (3000, 7000) for m in MULTS] + C_BG
jobs = [(i, C_REL + C_BG) for i in np.where(grp == "normal")[0]] + [(i, C_VAL) for i in np.where(grp == "val")[0]]
res = cached("inject", lambda: Parallel(n_jobs=30)(delayed(inject_job)(i, X[i], c, MED_BASE) for i, c in jobs))
INJ = pd.DataFrame([r for rr in res for r in rr]); INJ["group"] = grp[INJ.i]
INJ.to_csv(os.path.join(FIG, "inject.csv.gz"), index=False, compression="gzip")
L.append(f"[시간] ② {time.time() - T0:.0f} s")
npos = INJ[(INJ.kind == 'rel') & (INJ.group == 'normal')].groupby('i').pos.nunique()
L.append(f"\n[2] 주입: 정상 {len(npos)}개 파일 · 파일당 조용한 시점 {npos.min()}–{npos.max()}개(평균 {npos.mean():.1f}, 0.2 s 간격, 세 검출기 모두 ±40 ms 안에 원래 검출 없음)")
L.append("조용한 시점이 없어 빠진 정상 파일: " + ", ".join(files[i] for i in np.where(grp == "normal")[0] if i not in set(npos.index)))
L.append(f"진폭 = 배수 × 그 시점 기본 바탕(1–11k 포락선 100 ms 중앙값). 검출 = 주입 시점 −3…+10 ms 안에 봉우리. 정상 바탕 중앙값 {MED_BASE:.4f} g")


def m_at(df, col, q):      # 검출률이 q를 처음 넘는 배수 (log 배수에서 선형 보간)
    g = df.groupby("val")[col].mean().sort_index(); m, r = np.log(g.index.values.astype(float)), g.values
    if r[0] >= q: return f"≤{np.exp(m[0]):.1f}"
    j = np.argmax(r >= q)
    if r[j] < q: return f">{np.exp(m[-1]):.0f}"
    return f"{np.exp(m[j - 1] + (q - r[j - 1]) / (r[j] - r[j - 1]) * (m[j] - m[j - 1])):.1f}"


NR = INJ[(INJ.kind == "rel") & (INJ.group == "normal")]
L.append("\n[2a] 50 % · 90 % 검출 배수 (바탕 대비 최대 진폭) — 1–11k 기본 | 1k 고역만 | 예전 검출기")
L.append("주파수 ζ | 기본 m50 m90 | 고역만 m50 m90 | 예전 m50 m90 | 배수 20에서 검출률(기본)")
for (f, z), df in NR.groupby(["f", "z"], sort=False):
    L.append(f"{str(f):5s} {z:.2f} | {m_at(df, 'hit_기본', .5):>5s} {m_at(df, 'hit_기본', .9):>5s} | {m_at(df, 'hit_hp', .5):>5s} {m_at(df, 'hit_hp', .9):>5s} | "
             f"{m_at(df, 'hit_old', .5):>5s} {m_at(df, 'hit_old', .9):>5s} | {df[df.val == 20]['hit_기본'].mean():.3f}")

H = NR[NR.hit_기본 & (NR.f != "mix")].copy(); H["ftrue"] = H.f.astype(float)
H["e250"] = H.f250 - H.ftrue; H["e50"] = H.f50 - H.ftrue
L.append("\n[2b] 울림 주파수 · 시점 정확도 (기본 검출된 주입만). e = 추정 − 참값(Hz). 250 Hz 사전 = impact_rings_v2, 50 Hz 사전 = ring_mode_feats/2-7")
L.append("주파수 ζ 배수구간 | n | 250: 중앙|e| · |e|≤250 비율 · ζ 일치 | 50: 중앙|e| · |e|≤100 비율 | 시점 중앙(ms) [10–90 %]")
H["mb"] = pd.cut(H.val, [0, 5, 10, 25], labels=["≤5", "6–10", "13–20"])
for (f, z, mb), df in H.groupby(["f", "z", "mb"], observed=True, sort=True):
    L.append(f"{int(f):5d} {z:.2f} {mb:>5s} | {len(df):4d} | {df.e250.abs().median():6.0f} · {(df.e250.abs() <= 250).mean():.2f} · {(df.z250 == z).mean():.2f} | "
             f"{df.e50.abs().median():6.0f} · {(df.e50.abs() <= 100).mean():.2f} · ζ일치 {(df.z50 == z).mean():.2f} | {df.dt_ms.median():.2f} [{df.dt_ms.quantile(.1):.2f}, {df.dt_ms.quantile(.9):.2f}]")
bad = H[H.e250.abs() > 250]
L.append(f"|e250|>250 Hz 인 경우 {len(bad)}/{len(H)} — 추정 주파수 상위: " + ", ".join(f"{k:.0f}Hz {v}" for k, v in bad.f250.value_counts().head(6).items()))

# 다중 검출 (한 주입에 −3…+25 ms 안 봉우리 수)
L.append("\n[2c] 한 주입(물리 충격 1개)에 검출이 2개 이상 나온 비율 (검출된 것 중, −3…+25 ms) — 기본")
for (f, z), df in NR[NR.hit_기본].groupby(["f", "z"], sort=False):
    L.append(f"{str(f):5s} {z:.2f}: {(df.multi_기본 >= 2).mean():.3f}  (배수 ≥10: {(df[df.val >= 10].multi_기본 >= 2).mean():.3f})")
L.append("간격 변형별(전체 주입, 검출된 것 중 ≥2 비율): " + ", ".join(f"{k} {(NR[NR[f'hit_{k}']][f'multi_{k}'] >= 2).mean():.3f}" for k in ("기본", "간격3.4ms", "간격13.5ms")))


# %% ③ 상수 민감도: 대용신호 오검출 vs 주입 검출 배수 (ζ 0.02)
L.append("\n[3] 상수 민감도 (한 번에 하나만 바꿈). 오검출 = 대용신호 1개당 평균, 검출 배수 = ζ 0.02 m50 / m90")
L.append(f"{'변형':12s} | 오검출 | " + " | ".join(f"{f}" for f in FREQS) + " | 실제 정상/고장/val 평균 검출 수")
SENS = []
for k in VAR:
    c = f"hit_{k}"; row = [f"{SUR[f'n_{k}'].mean():.3f}"]
    for f in FREQS:
        df = NR[(NR.f == f) & (NR.z == 0.02)]; row.append(f"{m_at(df, c, .5)}/{m_at(df, c, .9)}")
    r = RF.groupby("group")[f"n_{k}"].mean()
    mu = (NR[NR[c]][f"multi_{k}"] >= 2).mean()
    L.append(f"{k:12s} | " + " | ".join(row) + f" | {r['normal']:.1f}/{r['abnormal']:.1f}/{r['val']:.1f} | 주입 1개에 2개 이상 {mu:.3f}")
    SENS.append(dict(var=k, false=SUR[f"n_{k}"].mean(), **{f"r{f}": NR[(NR.f == f) & (NR.z == 0.02) & (NR.val == 5)][c].mean() for f in FREQS}))
SENS = pd.DataFrame(SENS)
L.append("(아래 그림용) 배수 5에서 검출률: " + "; ".join(f"{r.var} " + "/".join(f"{getattr(r, f'r{f}'):.2f}" for f in FREQS) for r in SENS.itertuples()))


# %% ④-a val 3–5 kHz 바탕
RF["db35"] = 10 * np.log10(RF.p35); nor = RF[RF.group == "normal"]
lo_t, hi_t = nor.db35.quantile([1 / 3, 2 / 3])
RF["bg"] = np.where(RF.group == "val", "val", np.where(RF.db35 <= lo_t, "정상 낮음", np.where(RF.db35 >= hi_t, "정상 높음", "정상 중간")))
RF.loc[RF.group == "abnormal", "bg"] = "고장"
L.append("\n[4a] 3–5 kHz 바탕과 검출 바탕")
for g, df in RF.groupby("group"):
    L.append(f"{g:8s} 3–5k 전력 중앙 {df.db35.median():.1f} dB · 기본 바탕(포락선 100 ms 중앙값) 중앙 {df.base_med.median():.4f} g")
L.append(f"val − 정상: 3–5k {RF[RF.group == 'val'].db35.median() - nor.db35.median():+.1f} dB, 바탕 {20 * np.log10(RF[RF.group == 'val'].base_med.median() / nor.base_med.median()):+.1f} dB")
L.append(f"정상 120개 상관(스피어만) 3–5k dB vs 바탕: {nor[['db35', 'base_med']].corr('spearman').iloc[0, 1]:.2f}")
bgmap = RF.set_index("i").bg; INJ["bg"] = bgmap[INJ.i].values
for b, df in RF.groupby("bg"):
    L.append(f"  {b:6s} n={len(df):3d} · 3–5k 중앙 {df.db35.median():.1f} dB · 바탕 중앙 {df.base_med.median():.4f} g [10–90 % {df.base_med.quantile(.1):.4f}–{df.base_med.quantile(.9):.4f}]")
for b in ("정상 낮음", "정상 높음", "val"):
    d = INJ[(INJ.kind == "abs") & (INJ.bg == b)]
    L.append(f"  {b}: 주입 시점 국소 바탕 중앙 {d.b_loc.median():.4f} g → 50 % 검출에 필요한 진폭 ≈ 4.7–5.5 × 이 값")
L.append("주입 (ζ 0.02) — 상대 배수 m50/m90 (바탕 대비, 정상 낮음 | 정상 높음 | val)")
for f in (3000, 7000):
    rr = [m_at(INJ[(INJ.kind == 'rel') & (INJ.f == f) & (INJ.z == .02) & (INJ.bg == b)], "hit_기본", q) for b in ("정상 낮음", "정상 높음", "val") for q in (.5, .9)]
    L.append(f"  {f}: " + " | ".join(f"{rr[2 * j]}/{rr[2 * j + 1]}" for j in range(3)))
L.append(f"주입 (ζ 0.02) — 절대 진폭(× 정상 바탕 중앙 {MED_BASE:.4f} g)별 검출률: 정상 낮음 | 정상 높음 | val")
AB = INJ[INJ.kind == "abs"]
for f in (3000, 7000):
    for c in ABS_C:
        g = AB[(AB.f == f) & (AB.val == c)].groupby("bg").hit_기본.mean()
        L.append(f"  {f} ×{c:<3d}: {g.get('정상 낮음', np.nan):.2f} | {g.get('정상 높음', np.nan):.2f} | {g.get('val', np.nan):.2f}")


# %% ④-b 실제 파일: 한 물리 충격의 다중 검출 (25 ms 묶음)
L.append("\n[4b] 실제 파일 기본(1–11k) 검출 — 25 ms 안에 이어진 검출 묶음")
D = IMP[IMP.det == "bp"].sort_values(["i", "idx"]).copy(); D["group"] = grp[D.i]
D["gap"] = D.groupby("i").idx.diff() / FS * 1000
D["cl"] = (D.gap.isna() | (D.gap > 25)).cumsum()
cs = D.groupby("cl").agg(n=("idx", "size"), group=("group", "first"))
for g, df in cs.groupby("group"):
    L.append(f"{g:8s} 검출 {df.n.sum()} · 묶음 {len(df)} · 묶음당 2개 이상 비율 {(df.n >= 2).mean():.3f} · 묶음당 평균 {df.n.mean():.2f}")
L.append("이웃 검출 간격(ms) 분포: " + ", ".join(f"≤{b} {(D.gap <= b).sum()}" for b in (10, 15, 25, 50, 100)) + f" / 간격 전체 {D.gap.notna().sum()}")
same = D[D.gap <= 25].copy(); prev = D.shift(1)[D.gap <= 25]
L.append(f"25 ms 안 이웃 쌍의 울림 주파수: 같은 대역(±500 Hz) {(np.abs(same.f.values - prev.f.values) <= 500).mean():.2f}")


# %% ④-c 예전 검출기 vs 새 검출기 (190개)
L.append("\n[4c] 예전(final.impact_rings: 1k 고역 · 500 Hz · 6×파일 중앙값 · 5 ms · 끝단 제외 없음) vs 새(1–11k · 기본 상수)")
for g, df in RF.groupby("group"):
    L.append(f"{g:8s} 충격 수 중앙(평균) 예전 {df.n_old.median():.0f}({df.n_old.mean():.1f}) · 새 1k고역 {df.n_hp.median():.0f}({df.n_hp.mean():.1f}) · 새 1–11k {df.n_기본.median():.0f}({df.n_기본.mean():.1f})")
from scipy.stats import spearmanr
L.append(f"파일별 충격 수 스피어만 (예전 vs 새 1–11k): {spearmanr(RF.n_old, RF.n_기본)[0]:.2f}")
O, N = IMP[IMP.det == "old"], IMP[IMP.det == "bp"]
mt = []
for i in range(len(X)):
    o, n = O[O.i == i], N[N.i == i]
    for r in o.itertuples():
        if len(n):
            j = np.argmin(np.abs(n.idx.values - r.idx))
            if abs(n.idx.values[j] - r.idx) <= ms(3): mt.append(dict(i=i, fo=r.f, fn=n.f.values[j], zo=r.z, zn=n.z.values[j], fito=r.fit, fitn=n.fit.values[j]))
MT = pd.DataFrame(mt)
L.append(f"시점 일치(±3 ms): 예전 {len(O)}개 중 {len(MT)}개({len(MT) / len(O):.2f}) · 새 {len(N)}개 중 {len(MT) / len(N):.2f}")
L.append(f"일치 충격의 울림 주파수(250 Hz 사전) 같음 {(MT.fo == MT.fn).mean():.2f} · |차| ≤ 250 Hz {(np.abs(MT.fo - MT.fn) <= 250).mean():.2f}")
def res_val(df, n):
    r = np.zeros(n); m = (df.f >= 8000) & (df.f < 8700) & (df.z <= 0.02); np.maximum.at(r, df.i[m].values, df.fit[m].values); return r
RES_o, RES_n = res_val(O, len(X)), res_val(N, len(X))
RES_ref = np.zeros(len(X)); m = (R_old[:, 1] >= 8000) & (R_old[:, 1] < 8700) & (R_old[:, 2] <= .02); np.maximum.at(RES_ref, R_old[m, 0].astype(int), R_old[m, 3])
L.append(f"RES(8.0–8.7k · ζ≤0.02 최대 fit) 재현 = final.ring_table: {np.allclose(RES_o, RES_ref)}")
for g in ("normal", "abnormal", "val"):
    k = grp == g
    L.append(f"{g:8s} RES ≥ 0.7 파일 수: 예전 {(RES_o[k] >= .7).sum()} · 새 1–11k {(RES_n[k] >= .7).sum()} · 판정 다른 파일 " +
             ", ".join(files[k][(RES_o[k] >= .7) != (RES_n[k] >= .7)]))
L.append(f"\n[시간] 전체 {time.time() - T0:.0f} s")
open(os.path.join(FIG, "수치.txt"), "w", encoding="utf-8").write("\n".join(L))


# %% 그림 1: 검출률 곡선 (주파수별, ζ별) + 고역만 · 예전
C = {0.01: "#1f77b4", 0.02: "#ff7f0e", 0.04: "#2ca02c"}
fig, ax = plt.subplots(1, 6, figsize=(20, 3.8), sharey=True)
for a, f in zip(ax, list(FREQS) + ["mix"]):
    for z in (ZETAS if f != "mix" else (0,)):
        df = NR[(NR.f == f) & (NR.z == z)].groupby("val")
        col = C.get(z, "k"); lab = f"ζ {z}" if z else "1.9k+7k+9.5k"
        a.plot(df.hit_기본.mean(), "-o", ms=3, color=col, label=f"{lab} 1–11k")
        a.plot(df.hit_hp.mean(), ":", color=col, label=f"{lab} 1k 고역만")
        a.plot(df.hit_old.mean(), "--", color=col, alpha=.5, label=f"{lab} 예전")
    a.set_xscale("log"); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_xticks([2, 3, 5, 10, 20]); a.set_xticklabels(["2", "3", "5", "10", "20"]); a.axhline(.5, color="gray", lw=.5); a.axhline(.9, color="gray", lw=.5)
    a.set_title(f"{f / 1000:.1f} kHz" if f != "mix" else "여러 모드 합"); a.set_xlabel("주입 최대 진폭 ÷ 그 시점 바탕")
    a.legend(fontsize=6, loc="lower right")
ax[0].set_ylabel("검출률")
fig.suptitle(f"주입 검출률 (정상 120개 · 조용한 시점) — 실선 1–11k 기본 · 점선 1k 고역만 · 파선 예전 검출기", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_검출률곡선.png"), dpi=130); plt.close(fig)

# %% 그림 2: 상수 민감도 (오검출 vs 배수 5 검출률) + 주파수 오차
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for r in SENS.itertuples():
    rec = np.mean([getattr(r, f"r{f}") for f in FREQS])
    ax[0].scatter(r.false, rec, s=40, color="C3" if r.var == "기본" else "C0"); ax[0].annotate(r.var, (r.false, rec), fontsize=8, xytext=(3, 3), textcoords="offset points")
ax[0].axvline(.05, color="gray", ls=":", lw=1); ax[0].text(.051, .5, "옛 기준 0.05", fontsize=8, color="gray")
ax[0].set_xlabel("대용신호 1개당 오검출 (평균)"); ax[0].set_ylabel("배수 5 주입 검출률 (5개 주파수 평균, ζ 0.02)"); ax[0].set_title("상수 민감도: 오검출 ↔ 검출력")
for f in FREQS:
    df = H[H.ftrue == f]; ax[1].hist(df.e250.clip(-1500, 1500), bins=np.arange(-1525, 1526, 50), histtype="step", label=f"{f / 1000:.1f} kHz")
ax[1].set_xlabel("울림 주파수 추정 - 참값 (Hz, impact_rings_v2 250 Hz 사전, ±1500에서 자름)"); ax[1].set_ylabel("주입 수"); ax[1].set_yscale("log"); ax[1].legend(fontsize=8)
ax[1].set_title("울림 주파수 정확도 (검출된 주입)")
for b, st in (("정상 낮음", "-"), ("정상 높음", "--"), ("val", ":")):
    for f, col in ((3000, "C0"), (7000, "C1")):
        g = AB[(AB.f == f) & (AB.bg == b)].groupby("val").hit_기본.mean(); ax[2].plot(np.array(g.index) * MED_BASE, g.values, st, color=col, marker="o", ms=3, label=f"{f / 1000:.0f} kHz · {b}")
ax[2].set_xscale("log"); ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[2].set_xticks([0.012, 0.02, 0.03, 0.06]); ax[2].set_xticklabels(["0.012", "0.02", "0.03", "0.06"]); ax[2].set_xlabel("주입 최대 진폭 (g, 절대)"); ax[2].set_ylabel("검출률"); ax[2].legend(fontsize=8)
ax[2].set_title("3–5 kHz 바탕 영향 (같은 g 충격, ζ 0.02)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_민감도·주파수·바탕.png"), dpi=130); plt.close(fig)

# %% 그림 3: 예전 vs 새 충격 수
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for g, col in (("normal", "C0"), ("abnormal", "C3"), ("val", "C2")):
    d = RF[RF.group == g]; ax[0].scatter(d.n_old + np.random.default_rng(0).uniform(-.2, .2, len(d)), d.n_기본, s=12, color=col, alpha=.7, label=g)
mx = max(RF.n_old.max(), RF.n_기본.max()); ax[0].plot([0, mx], [0, mx], "k:", lw=.8)
ax[0].set_xscale("symlog"); ax[0].set_yscale("symlog"); ax[0].set_xlabel("예전 검출기 충격 수"); ax[0].set_ylabel("새 검출기(1–11k) 충격 수"); ax[0].legend(); ax[0].set_title("파일별 충격 수")
ax[1].scatter(MT.fo + np.random.default_rng(1).uniform(-60, 60, len(MT)), MT.fn, s=5, alpha=.4); ax[1].plot([1000, 12000], [1000, 12000], "k:", lw=.8)
ax[1].set_xlabel("예전 울림 주파수 (Hz)"); ax[1].set_ylabel("새 울림 주파수 (Hz)"); ax[1].set_title(f"시점이 겹친 충격 {len(MT)}개의 울림 주파수")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_예전vs새.png"), dpi=130); plt.close(fig)
print("done", time.time() - T0)
