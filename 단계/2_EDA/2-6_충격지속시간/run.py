# %% [markdown]
# 2-6 충격 지속 시간 — 모델이 틀린 파일(놓친 고장 · 오경보 정상)에서 충격이 얼마나 오래 울리나
# 정의 (라벨 보기 전에 고정):
# - 신호: 평균 제거 → 200 Hz 고역통과 (4차 Butterworth, sosfiltfilt)
# - 충격: |h| > 8σ (σ = MAD × 1.4826), 10 ms 안은 한 사건 (뷰어와 같음). 충격이 없으면 최대 |h| 지점 1개를 사건으로 (표시)
# - 포락선: Hilbert |h|를 0.2 ms(5점) 이동평균
# - 감쇠 시간 t10: 사건 최고점 → 포락선이 max(최고값 × 10%, 바탕 3σ) 아래로 처음 떨어질 때까지 (바탕에 막히면 표시)
# - 50% 폭 t50: 최고점 주변에서 포락선이 최고값의 50% 이상인 연속 구간 길이
# 묶음: 모델 결과(EXP-007 기준 ET, EXP-012 AE)의 5번 중 3번 이상 틀림 → 뷰어 "어느 쪽이든 틀림"과 같다
# 실행: conda run -n qe-phm python "단계/2_EDA/2-6_충격지속시간/run.py"

# %% 설정
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import load_train
import src.viz  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert

FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; N = len(y)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
Xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
ENV = np.abs(hilbert(H, axis=1))
k5 = np.ones(5) / 5
ENV = np.apply_along_axis(lambda e: np.convolve(e, k5, mode="same"), 1, ENV)

# %% 사건별 지속 시간
def events_of(h):
    sig = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    if len(idx) == 0:
        return [int(np.abs(h).argmax())], sig, False
    return [int(g[np.argmax(np.abs(h[g]))]) for g in np.split(idx, np.flatnonzero(np.diff(idx) > 250) + 1)], sig, True

R = []
for i in range(N):
    ev, sig, has = events_of(H[i]); e = ENV[i]
    rows = []
    for k in ev:
        a, b = max(0, k - 50), min(len(e), k + 50)
        p = a + int(e[a:b].argmax()); pk = e[p]          # 포락선 최고점 (±2 ms 안)
        lvl = max(0.1 * pk, 3 * sig); floor_hit = 0.1 * pk < 3 * sig
        after = np.flatnonzero(e[p:] < lvl)
        t10 = (after[0] if len(after) else len(e) - p) / FS * 1000
        hi = e >= 0.5 * pk
        l = p
        while l > 0 and hi[l - 1]: l -= 1
        r = p
        while r < len(e) - 1 and hi[r + 1]: r += 1
        tail = e[p + 125:p + 500]                              # 최고점 뒤 5–20 ms
        rv = np.median(tail) / pk if len(tail) == 375 else np.nan
        rows.append(dict(p=p, t=p / FS, pk=pk / sig, t10=t10, t50=(r - l + 1) / FS * 1000, floor=floor_hit, rv=rv))
    big = max(rows, key=lambda d: d["pk"])
    R.append(dict(big_p=big["p"], big_rv=big["rv"], med_rv=float(np.nanmedian([d["rv"] for d in rows])) if any(np.isfinite(d["rv"]) for d in rows) else np.nan,
                  n_imp=len(ev) if has else 0, big_t10=big["t10"], big_t50=big["t50"], big_pk=big["pk"], big_floor=big["floor"],
                  med_t10=float(np.median([d["t10"] for d in rows])), med_t50=float(np.median([d["t50"] for d in rows]))))

# %% 묶음 (모델 결과)
def counts(exp_dir):
    r = json.load(open(os.path.join(ROOT, "단계", "5_모델비교", exp_dir, "result.json"), encoding="utf-8"))
    c = next(iter(r["configs"].values()))
    return c["miss_counts"], c["fp_counts"]
aem, aef = counts("유의미/EXP-012_스펙트럼AE_tSNE"); etm, etf = counts("기각/EXP-007_공선성제거")
def who(n):
    tags = [m for m, d in [("ET", etm), ("AE", aem)] if d.get(n, 0) >= 3] if n.startswith("abn") else \
           [m for m, d in [("ET", etf), ("AE", aef)] if d.get(n, 0) >= 3]
    return "+".join(tags)
GRP = []
for i, n in enumerate(names):
    w = who(n)
    GRP.append(("놓친 고장" if w else "잡힌 고장") if y[i] else ("오경보 정상" if w else "문제없는 정상"))
GRP = np.array(GRP)
ORDER = ["놓친 고장", "잡힌 고장", "오경보 정상", "문제없는 정상"]

# %% 결과 표 (UTF-8)
keys = ["n_imp", "big_t10", "big_t50", "med_t10", "med_t50", "big_pk", "big_rv", "med_rv"]
V = {k: np.array([r[k] for r in R], float) for k in keys}
lines = ["묶음별 중앙값 [25%–75%] (ms, 충격 수는 개, big_pk는 바탕 σ 배, rv = 최고점 뒤 5–20 ms 포락선 중앙 / 최고값)"]
for g in ORDER:
    m = GRP == g
    lines.append(f"{g} ({m.sum()}개): " + " | ".join(f"{k} {np.nanmedian(V[k][m]):.3g} [{np.nanpercentile(V[k][m],25):.3g}–{np.nanpercentile(V[k][m],75):.3g}]" for k in keys))
lines.append("")
lines.append("틀린 파일 개별 (가장 큰 사건 기준, *=10%가 바탕 3σ에 막힘, ^=충격 없음이라 최대점 사용)")
for g in ["놓친 고장", "오경보 정상"]:
    lines.append(f"[{g}]")
    for i in np.flatnonzero(GRP == g):
        r = R[i]
        lines.append(f"  {names[i]} ({who(names[i])}): 충격 {r['n_imp']}{'^' if r['n_imp']==0 else ''}, 최대 사건 t10 {r['big_t10']:.1f} ms{'*' if r['big_floor'] else ''}, "
                     f"t50 {r['big_t50']:.2f} ms, 높이 {r['big_pk']:.0f}σ, 잔향 비 {r['big_rv']:.3f} | 사건 중앙 t10 {r['med_t10']:.1f}, t50 {r['med_t50']:.2f}, 잔향 비 {r['med_rv']:.3f}")
# 설명용 AUC (고장=1 기준, 모델 선택에 쓰지 않음). 잔향 비는 최고점이 클수록 작아지는 구조라 높이 구간을 나눠 본다
from sklearn.metrics import roc_auc_score
lines.append("")
lines.append("고장 vs 정상 AUC (값이 작을수록 고장이면 0.5 아래). 전체 / 최대 사건 높이 40σ 이상만 / 40σ 미만만")
for k in ["big_t10", "big_t50", "big_rv", "med_t10", "med_rv", "big_pk"]:
    v = V[k]; ok = np.isfinite(v); s = []
    for m in [ok, ok & (V["big_pk"] >= 40), ok & (V["big_pk"] < 40)]:
        s.append(f"{roc_auc_score(y[m], v[m]):.2f} (고장 {int(y[m].sum())}/정상 {int((~y[m].astype(bool)).sum())})" if 0 < y[m].sum() < m.sum() else "-")
    lines.append(f"  {k}: " + " / ".join(s))
import pandas as pd
pd.DataFrame({"name": names, "y": y, "group": GRP, **{k: V[k] for k in keys}}).to_csv(os.path.join(HERE, "figs", "duration_per_file.csv"), index=False, encoding="utf-8-sig")
open(os.path.join(HERE, "figs", "duration_numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))

# %% 그림 1: 묶음별 점 (가장 큰 사건 t10 / 사건 중앙 t10), 틀린 파일 이름 표시
C = {"놓친 고장": "#eb6834", "잡힌 고장": "#eb6834", "오경보 정상": "#2a78d6", "문제없는 정상": "#2a78d6"}
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
rng = np.random.default_rng(0)
from matplotlib.ticker import FuncFormatter
TICK = FuncFormatter(lambda v, _: f"{v:g}")
for a, k, title, yl in [(ax[0], "big_t10", "가장 큰 충격의 감쇠 시간 t10", "ms (로그)"),
                        (ax[1], "big_rv", "가장 큰 충격의 잔향 비 (뒤 5–20 ms 포락선 / 최고값)", "비 (로그)")]:
    for j, g in enumerate(ORDER):
        idx = np.flatnonzero(GRP == g); xs = j + rng.uniform(-0.18, 0.18, len(idx))
        err = g in ("놓친 고장", "오경보 정상")
        a.scatter(xs, V[k][idx], s=46 if err else 22, color=C[g], alpha=1 if err else 0.35, edgecolors="white", linewidths=1.5, zorder=3)
        a.hlines(np.nanmedian(V[k][idx]), j - 0.3, j + 0.3, color="#0b0b0b", lw=2, zorder=4)
        if err:
            for xx, ii in zip(xs, idx):
                a.annotate(names[ii][4:], (xx, V[k][ii]), xytext=(5, 0), textcoords="offset points", fontsize=8, color="#52514e", va="center")
    a.set_yscale("log"); a.set_xticks(range(4), [f"{g}\n({(GRP==g).sum()}개)" for g in ORDER])
    a.yaxis.set_major_formatter(TICK); a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    a.set_ylabel(yl); a.set_title(title + " (검은 선 = 중앙값)", loc="left", fontsize=11)
    a.grid(axis="y", color="#e6e5e0", lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "duration_groups.png"), dpi=130)

# %% 그림 2: 틀린 파일들의 가장 큰 사건 포락선 (최고점 = 1로 정규화, 최고점 기준 −2 ~ +30 ms)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
tt = np.arange(-50, 750) / FS * 1000
for a, g, col in [(ax[0], "놓친 고장", "#eb6834"), (ax[1], "오경보 정상", "#2a78d6")]:
    # 배경: 잡힌 고장 중앙 포락선 (비교 기준)
    def curve(i):
        p = R[i]["big_p"]; e = ENV[i]
        seg = np.full(len(tt), np.nan); a0 = p - 50
        lo, hi = max(0, a0), min(len(e), a0 + len(tt)); seg[lo - a0:hi - a0] = e[lo:hi]
        return seg / e[p]
    for i in np.flatnonzero(GRP == g):
        a.plot(tt, curve(i), color=col, lw=1, alpha=0.8)
        c = curve(i); k = np.nanargmin(np.abs(tt - 10))
        a.annotate(names[i][4:], (tt[-1], c[-1] if np.isfinite(c[-1]) else 0.01), fontsize=7, color="#52514e")
    ref = np.nanmedian(np.stack([curve(i) for i in np.flatnonzero(GRP == "잡힌 고장")]), 0)
    a.plot(tt, ref, color="#0b0b0b", lw=2, ls="--", label="잡힌 고장 중앙")
    a.axhline(0.1, color="#85847e", lw=1, ls=":"); a.text(tt[-1], 0.1, " 10%", va="center", fontsize=8, color="#85847e")
    a.set_yscale("log"); a.set_ylim(3e-3, 1.3); a.set_xlabel("최고점 이후 ms")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); a.axvspan(5, 20, color="#f0efec", zorder=0)
    a.set_title(f"{g}: 가장 큰 충격의 포락선 (최고점 = 1)", loc="left", fontsize=11); a.legend(frameon=False, fontsize=9)
    a.grid(color="#e6e5e0", lw=0.6); a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("포락선 / 최고값 (로그)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "duration_envelopes.png"), dpi=130)
print("done")
