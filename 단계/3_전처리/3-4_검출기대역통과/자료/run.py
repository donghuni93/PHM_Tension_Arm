# %% 2-7 ① 검출기 입력에 11 kHz 상한(1–11 kHz 대역통과)을 걸었을 때 이벤트 · 5피처 변화 ② 원신호 포화(클리핑) 점검
# 검출기 상수(src/impacts.py)는 그대로. 입력만 바꿔 detect_impacts(_hp=)와 fine_ring에 같은 신호를 넣는다.
# 기준 = 2-6 events_190.csv (1 kHz 고역만). 라벨은 그룹 집계에만 쓰고 아무것도 고르지 않는다. val 라벨 없음.
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.impacts import detect_impacts, highpass, _atoms50, _baseline, _sos, SMOOTH_HZ, FS
from src.final import _SEG
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
B26 = os.path.join(ROOT, "단계", "아카이브", "new_단계", "2_EDA", "2-6_바닥보정·이동·기준ET", "figs")
A50, F50 = _atoms50(); ZS = (0.01, 0.02, 0.04, 0.08)
SOS_BP = butter(4, [1000, 11000], btype="bandpass", fs=FS, output="sos")

def bandpass(x):
    x = np.asarray(x, float); return sosfiltfilt(SOS_BP, x - x.mean())
def fine_ring(xh, idx):   # 2-6과 같음
    S = np.stack([xh[p:p + _SEG] for p in idx]); S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, A50), axis=2); b = fit.argmax(1)
    return np.column_stack([F50[b], np.array(ZS)[b % 4], fit.max(1)])
def clusters(idx, gap_ms=25):
    return np.concatenate([[0], np.cumsum(np.diff(idx) > gap_ms * FS / 1000)]).astype(int) if len(idx) else np.array([], int)
def events(n, g, x, xh):   # 2-6과 같은 이벤트 정의, 필터 신호만 바꿔 넣는다
    imp = detect_impacts(x, _hp=xh); idx, hg = imp["idx"], imp["height_g"]; out = []
    if len(idx):
        ring = fine_ring(xh, idx); cl = clusters(idx)
        for e in range(cl.max() + 1):
            m = np.where(cl == e)[0]; j = m[np.argmax(hg[m])]
            out.append(dict(file=n, group=g, t=idx[j] / FS, f=ring[j, 0], fit=ring[j, 2], h=hg[j], x=imp["height_x"][j]))
    return out
def base_level(xh):   # 파일 전체 바탕(100 ms 이동 중앙값)의 중앙값, g
    e = sosfiltfilt(_sos("lowpass", SMOOTH_HZ, FS), np.abs(hilbert(xh))); return float(np.median(_baseline(e)))

# %% 190개 파일: 기준(1 kHz 고역) 재계산 · 1–11 kHz 대역통과 · 포화 점검
names = pd.read_csv(os.path.join(B26, "files_190.csv")); grp = names.set_index("file").group
PAT = {"normal": "TeamProject_train_변경/normal", "abnormal": "TeamProject_train_변경/abnormal", "val": "TeamProject_val/val"}
ev_old, ev_new, lev, clip = [], [], [], []
for n, g in names.itertuples(index=False):
    x = np.loadtxt(os.path.join(ROOT, "data", PAT[g], n + ".csv"))
    xo, xb = highpass(x), bandpass(x)
    ev_old += events(n, g, x, xo); ev_new += events(n, g, x, xb)
    lev.append(dict(file=n, base_old=base_level(xo), base_new=base_level(xb), dc=x.mean()))
    # 포화: 극값 개수 · 극값에서 연속 같은 값 최대 길이 · 양자화 간격
    r = {}
    for k, v in (("max", x.max()), ("min", x.min())):
        at = x == v; d_ = np.diff(np.r_[0, at.astype(int), 0]); best = int(((np.flatnonzero(d_ == -1)) - np.flatnonzero(d_ == 1)).max())
        r[k] = v; r[f"n_{k}"] = int(at.sum()); r[f"run_{k}"] = best
    u = np.unique(x); d = np.diff(u)
    clip.append(dict(file=n, group=g, **r, step_min=float(d.min()), step_med=float(np.median(d)), n_unique=len(u),
                     max_run_any=int(np.max(np.diff(np.flatnonzero(np.r_[True, np.diff(x) != 0, True])))),
                     n_near_max=int((x >= x.max() - 2 * d.min()).sum()), n_near_min=int((x <= x.min() + 2 * d.min()).sum())))
Eo, En = pd.DataFrame(ev_old), pd.DataFrame(ev_new); LV = pd.DataFrame(lev).set_index("file"); C = pd.DataFrame(clip)
En.to_csv(os.path.join(FIG, "events_190_bp11k.csv"), index=False); C.to_csv(os.path.join(FIG, "clip_190.csv"), index=False)
E26 = pd.read_csv(os.path.join(B26, "events_190.csv"))
L = [f"기준 재계산 이벤트 {len(Eo)}개 vs 2-6 캐시 {len(E26)}개, 같은가: {len(Eo) == len(E26) and np.allclose(Eo.f.values, E26.f.values) and np.allclose(Eo.x.values, E26.x.values)}"]

# %% 파일별 이벤트 수
files = names.file.tolist()
co = Eo.groupby("file").size().reindex(files, fill_value=0); cn = En.groupby("file").size().reindex(files, fill_value=0)
L.append("\n[1] 파일별 이벤트 수(25 ms 묶음) 기준(1k 고역) → 1–11k 대역통과")
for g in ("normal", "abnormal", "val"):
    fs_ = [f for f in files if grp[f] == g]; ch = [f for f in fs_ if co[f] != cn[f]]
    L.append(f"{g:8s} 파일 {len(fs_)} | 이벤트 합 {co[fs_].sum()} → {cn[fs_].sum()} | 바뀐 파일 {len(ch)}: " + ", ".join(f"{f} {co[f]}→{cn[f]}" for f in ch))

# %% 울림 주파수 대역 분포
BANDS = [(1000, 1600), (1600, 4500), (4500, 5500), (5500, 8000), (8000, 11000), (11000, 13000)]
BL = ["1–1.6k", "1.6–4.5k", "4.5–5.5k", "5.5–8k", "8–11k", "≥11k"]
def band_tab(E):
    E = E.copy(); E["band"] = pd.cut(E.f, [b[0] for b in BANDS] + [13000], right=False, labels=BL)
    return E.groupby(["group", "band"], observed=False).size().unstack().reindex(["normal", "abnormal", "val"])
To, Tn = band_tab(Eo), band_tab(En)
L.append("\n[2] 대표 울림 주파수 대역별 이벤트 수 (기준 → 대역통과)")
L.append("group    | " + " | ".join(f"{b:>9s}" for b in BL))
for g in To.index:
    L.append(f"{g:8s} | " + " | ".join(f"{To.loc[g, b]:>3d} → {Tn.loc[g, b]:<3d}" for b in BL))
L.append(f"대역통과 후 ≥11 kHz 울림 이벤트: {int((En.f >= 11000).sum())}개 " + ", ".join(f"{r.file}({r.f:.0f}Hz)" for r in En[En.f >= 11000].itertuples()))
L.append(f"기준의 ≥11 kHz 울림 이벤트: {int((Eo.f >= 11000).sum())}개 " + ", ".join(f"{r.file}({r.f:.0f}Hz)" for r in Eo[Eo.f >= 11000].itertuples()))

# %% 같은 이벤트(시각 ±2 ms)의 울림 주파수 대역이 바뀌었나
def bnd(f): return next(i for i, b in enumerate(BANDS) if b[0] <= f < b[1]) if f < 13000 else 5
mv = []
for f in files:
    a, b = Eo[Eo.file == f], En[En.file == f]
    for r in a.itertuples():
        d = (b.t - r.t).abs()
        if len(d) and d.min() <= 0.002:
            s = b.loc[d.idxmin()]
            if bnd(r.f) != bnd(s.f): mv.append(f"{f} t={r.t:.3f}s {r.f:.0f}→{s.f:.0f}Hz ({grp[f]})")
L.append(f"짝지은 이벤트 중 대역이 바뀐 것 {len(mv)}개: " + "; ".join(mv))

# %% 5피처
LOW, HIGH = (1600, 4500), (8000, 11000)
def feats(E):
    EV = {f: g for f, g in E.groupby("file")}; rows = []
    for f in files:
        e = EV.get(f)
        if e is None: rows.append(dict(file=f, cnt_low=0, cnt_high=0, cnt_total=0, ratio_off=0.0, off_max=0.0)); continue
        lo = e[(e.f >= LOW[0]) & (e.f < LOW[1])]; hi = e[(e.f >= HIGH[0]) & (e.f < HIGH[1])]; off = pd.concat([lo, hi])
        rows.append(dict(file=f, cnt_low=len(lo), cnt_high=len(hi), cnt_total=len(e), ratio_off=(len(lo) + len(hi)) / (len(e) + 1),
                         off_max=float(off.x.max()) if len(off) else 0.0))
    return pd.DataFrame(rows).set_index("file")
Fo, Fn = feats(Eo), feats(En); Fo.to_csv(os.path.join(FIG, "feats_old.csv")); Fn.to_csv(os.path.join(FIG, "feats_bp11k.csv"))
L.append("\n[3] 5피처 그룹 중앙값 기준 → 대역통과, 값이 바뀐 파일 수 (off_max는 상대 1 % 넘게 바뀐 것)")
for c in Fo.columns:
    s = []
    for g in ("normal", "abnormal", "val"):
        fs_ = [f for f in files if grp[f] == g]
        diff = ~np.isclose(Fo.loc[fs_, c], Fn.loc[fs_, c], rtol=0.01, atol=1e-9)
        s.append(f"{g} {Fo.loc[fs_, c].median():.2f}→{Fn.loc[fs_, c].median():.2f} (바뀜 {diff.sum()})")
    L.append(f"{c:9s} | " + " | ".join(s))
L.append("off 이벤트 수(cnt_low+cnt_high)가 바뀐 파일: " + ", ".join(f"{f}({grp[f][:3]}) {Fo.cnt_low[f]+Fo.cnt_high[f]}→{Fn.cnt_low[f]+Fn.cnt_high[f]}" for f in files if Fo.cnt_low[f]+Fo.cnt_high[f] != Fn.cnt_low[f]+Fn.cnt_high[f]))
big = [f for f in files if not np.isclose(Fo.off_max[f], Fn.off_max[f], rtol=0.10)]
L.append("off_max가 10 % 넘게 바뀐 파일: " + ", ".join(f"{f}({grp[f][:3]}) {Fo.off_max[f]:.1f}→{Fn.off_max[f]:.1f}" for f in big))

# %% 12.4 kHz 톤 정상 3개
L.append("\n[4] nor_097 · nor_101 · nor_120 (12.4 kHz 톤, DC 음수)")
for f in ("nor_097", "nor_101", "nor_120"):
    a, b = Eo[Eo.file == f], En[En.file == f]
    L.append(f"{f}: DC {LV.dc[f]:.3f} g | 이벤트 {len(a)}→{len(b)} | 바탕 중앙(파일 전체) {LV.base_old[f]:.4f}→{LV.base_new[f]:.4f} g | 울림 f {list(a.f.astype(int))}→{list(b.f.astype(int))} | x {list(a.x.round(1))}→{list(b.x.round(1))}")
r = LV.base_new / LV.base_old
L.append(f"바탕 비(대역통과/기준) 전체 190개: 중앙 {r.median():.3f}, 최소 {r.min():.3f} ({r.idxmin()}), 하위 5개 " + ", ".join(f"{f} {v:.3f}" for f, v in r.nsmallest(5).items()))

# %% 포화 점검
L.append("\n[5] 원신호 포화 점검 (극값 개수 · 극값 연속 길이 · 양자화 간격)")
for g in ("normal", "abnormal", "val"):
    c = C[C.group == g]
    L.append(f"{g:8s} max 범위 [{c['max'].min():.3f}, {c['max'].max():.3f}] g, min 범위 [{c['min'].min():.3f}, {c['min'].max():.3f}] g | 극값 개수 >1 파일: max {int((c.n_max > 1).sum())}, min {int((c.n_min > 1).sum())} | 극값 연속 ≥3 파일: {int(((c.run_max >= 3) | (c.run_min >= 3)).sum())} | 최소 간격 {c.step_min.min():.2e}–{c.step_min.max():.2e} g | 같은 값 연속 최대 {c.max_run_any.max()}")
flag = C[(C.run_max >= 3) | (C.run_min >= 3) | (C.n_max >= 3) | (C.n_min >= 3)]
L.append(f"포화 의심(극값 3번 이상 또는 연속 3 이상): {len(flag)}개 " + ", ".join(f"{r.file} max {r.max:.3f}×{r.n_max}(연속 {r.run_max}) min {r.min:.3f}×{r.n_min}(연속 {r.run_min})" for r in flag.itertuples()))
L.append("극값 개수 상위 5: " + ", ".join(f"{r.file} n_max {r.n_max} n_min {r.n_min}" for r in C.assign(k=C.n_max + C.n_min).nlargest(5, "k").itertuples()))
L.append("극값 ±2 간격 안 표본 수 상위 5: " + ", ".join(f"{r.file} {r.n_near_max}/{r.n_near_min}" for r in C.assign(k=C.n_near_max + C.n_near_min).nlargest(5, "k").itertuples()))
L.append(f"|max| 또는 |min| 상위 5: " + ", ".join(f"{r.file}({r.group[:3]}) {r.max:.2f}/{r.min:.2f}" for r in C.assign(k=np.maximum(C['max'].abs(), C['min'].abs())).nlargest(5, "k").itertuples()))
L.append(f"양자화 간격(최소) 최빈값 {C.step_min.round(8).mode().iloc[0]:.3e} g, 중앙 간격 중앙 {C.step_med.median():.3e} g, 고유값 수 중앙 {int(C.n_unique.median())}")
open(os.path.join(FIG, "수치.txt"), "w", encoding="utf-8").write("\n".join(L))

# %% 그림 1: 대역별 울림 이벤트 수, 기준 vs 대역통과 / 그림 2: 파일별 이벤트 수 · off_max 기준 vs 대역통과
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
w = 0.38; xx = np.arange(len(BL))
for a, g, t in zip(ax, ("normal", "abnormal", "val"), ("정상 120", "고장 30", "val 40")):
    a.bar(xx - w / 2, To.loc[g].values, w, color="#9aa5b1", label="기준 (1 kHz 고역)")
    a.bar(xx + w / 2, Tn.loc[g].values, w, color="#2f6db5", label="1–11 kHz 대역통과")
    a.set_xticks(xx, BL, rotation=30, fontsize=8); a.set_title(t); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("이벤트 수"); ax[0].legend(fontsize=8)
fig.suptitle("대표 울림 주파수 대역별 이벤트 수"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_대역분포.png"), dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
col = {"normal": "#2f6db5", "abnormal": "#e07b28", "val": "#7a7a7a"}; lab = {"normal": "정상", "abnormal": "고장", "val": "val"}
rng = np.random.default_rng(0)
for g in ("normal", "abnormal", "val"):
    fs_ = [f for f in files if grp[f] == g]; j = rng.uniform(-.15, .15, (2, len(fs_)))
    ax[0].scatter(co[fs_] + j[0], cn[fs_] + j[1], s=14, c=col[g], alpha=.7, label=lab[g])
    ax[1].scatter(Fo.off_max[fs_], Fn.off_max[fs_], s=14, c=col[g], alpha=.7, label=lab[g])
for a, t in zip(ax, ("파일별 이벤트 수", "off_max (포락선/바탕 최대)")):
    m = max(a.get_xlim()[1], a.get_ylim()[1]); a.plot([0, m], [0, m], "k:", lw=.8); a.set_title(t)
    a.set_xlabel("기준 (1 kHz 고역)"); a.set_ylabel("1–11 kHz 대역통과"); a.grid(alpha=.3)
ax[1].set_xscale("symlog", linthresh=5); ax[1].set_yscale("symlog", linthresh=5)
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_파일별변화.png"), dpi=130); plt.close(fig)
print("done")
