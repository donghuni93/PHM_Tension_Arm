# %% 2-10 미끄러지는 지속음 — 2초 내내 주파수가 미끄러지는 좁은 지속음이 있으면 정상인가 고장인가
# 출발: val_001 (8.3k 스펙트럼 규칙만으로 고장, 9000 → 8130 Hz로 미끄러지는 지속음, 6-7 V001_지속음.txt)
# 정의 (결과 보기 전에 고정, 6-7과 같은 STFT · 돌출)
#   STFT 0.1 s 창(2500점) · 50 % 겹침 → 프레임별 log 파워 − 주파수 방향 ±200 Hz 이동 중앙값 = 돌출(dB)
#   돌출 ≥ 10 dB인 국소 최대 = 좁은 봉우리. 이웃 프레임에서 |Δf| ≤ 60 Hz면 같은 선으로 잇는다 (빈 프레임 2개까지 허용)
#   파일 값 = 가장 긴 선의 길이(s) · 그 선의 Δf(끝 − 시작, Hz) · 기울기(Hz/s, 선형 맞춤) · 중앙 주파수
#   "미끄러지는 지속음" = 선 길이 ≥ 1.0 s 이고 |Δf| ≥ 200 Hz
#   대역 두 가지: 전체 2–12.4 kHz · 8.3k 규칙 근처 7.8–9.0 kHz
# 출력: figs/*.png · figs/결과.txt · figs/파일표.csv
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.ndimage import median_filter
from src.exp import load_train
from src.data import load_folder
import src.viz  # noqa: F401

FS = 25000
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
NOR, ABN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#7c3aed", "#1F2937", "#6B7280", "#E5E7EB"
PROM_DB, TOL_HZ, GAP, MIN_LEN, MIN_DF = 10.0, 60.0, 2, 1.0, 200.0
BANDS = {"전체 2–12.4 kHz": (2000, 12400), "7.8–9.0 kHz": (7800, 9000)}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def spec(x):
    f, t, Z = stft(x - x.mean(), fs=FS, nperseg=2500, noverlap=1250)
    L = 10 * np.log10(np.abs(Z) ** 2 + 1e-20)
    return f, t, L - median_filter(L, size=(41, 1), mode="nearest")


def tracks(f, t, P, lo, hi):
    """봉우리를 프레임 사이로 이어 선 목록 [(프레임 번호 배열, 주파수 배열)]"""
    b = np.flatnonzero((f >= lo) & (f <= hi)); fb = f[b]; Pb = P[b]
    live, done = [], []                                    # 선 = [frames, freqs, last_frame]
    for j in range(Pb.shape[1]):
        c = Pb[:, j]
        pk = np.flatnonzero((c[1:-1] >= PROM_DB) & (c[1:-1] >= c[:-2]) & (c[1:-1] >= c[2:])) + 1
        used = set()
        for tr in sorted(live, key=lambda r: -len(r[0])):  # 긴 선부터 가장 가까운 봉우리를 가져감
            cand = [k for k in pk if k not in used and abs(fb[k] - tr[1][-1]) <= TOL_HZ]
            if cand:
                k = min(cand, key=lambda k: abs(fb[k] - tr[1][-1]))
                tr[0].append(j); tr[1].append(fb[k]); tr[2] = j; used.add(k)
        for k in pk:
            if k not in used:
                live.append([[j], [fb[k]], j])
        keep = []
        for tr in live:
            (keep if j - tr[2] <= GAP else done).append(tr)
        live = keep
    done += live
    return [(np.array(a), np.array(b_)) for a, b_, _ in done]


def file_stats(x):
    f, t, P = spec(x)
    out = {}
    for nm, (lo, hi) in BANDS.items():
        T = tracks(f, t, P, lo, hi)
        if not T:
            out[nm] = dict(len=0.0, df=0.0, slope=0.0, fmed=np.nan, f0=np.nan, f1=np.nan, maxdf=0.0); continue
        fr, fq = max(T, key=lambda r: r[0][-1] - r[0][0])
        ln = (fr[-1] - fr[0]) * (t[1] - t[0])
        sl = np.polyfit(t[fr], fq, 1)[0] if len(fr) >= 5 else 0.0
        n5 = max(1, len(fq) // 10)
        # 점검(11-22 추가, 첫 결과 뒤): 가장 긴 선만 보면 다른 고정 지속음에 가려진다 → 1 s 넘는 선 모두의 |Δf| 최대
        dfs = [abs(np.median(b_[-max(1, len(b_) // 10):]) - np.median(b_[:max(1, len(b_) // 10)])) for a, b_ in T
               if (a[-1] - a[0]) * (t[1] - t[0]) >= MIN_LEN]
        out[nm] = dict(len=ln, df=float(np.median(fq[-n5:]) - np.median(fq[:n5])), slope=sl, fmed=float(np.median(fq)),
                       f0=float(np.median(fq[:n5])), f1=float(np.median(fq[-n5:])), maxdf=float(max(dfs)) if dfs else 0.0)
    return out


X, y, names = load_train(); names = [str(n) for n in names]
vd = os.path.join(ROOT, "2_data", "TeamProject_val", "val"); vd = vd if os.path.isdir(vd) else os.path.dirname(vd)
Xv, nv = load_folder(vd); nv = [str(n) for n in nv]
rows = []
for grp, XX, NN, YY in [("train", X, names, y), ("val", Xv, nv, [np.nan] * len(nv))]:
    for x, n, yy in zip(XX, NN, YY):
        s = file_stats(x)
        r = dict(set=grp, file=n, y=yy)
        for nm in BANDS:
            for k, v in s[nm].items():
                r[f"{nm}|{k}"] = v
        rows.append(r)
D = pd.DataFrame(rows)
for nm in BANDS:
    D[f"{nm}|slide"] = (D[f"{nm}|len"] >= MIN_LEN) & (D[f"{nm}|df"].abs() >= MIN_DF)
    D[f"{nm}|sustain"] = D[f"{nm}|len"] >= MIN_LEN
D.to_csv(os.path.join(FIG, "파일표.csv"), index=False, encoding="utf-8-sig")

# %% 표
txt = [f"정의: 돌출 ≥ {PROM_DB} dB 봉우리를 |Δf| ≤ {TOL_HZ} Hz로 잇기(빈 프레임 {GAP}개 허용). 지속음 = 가장 긴 선 ≥ {MIN_LEN} s, 미끄러짐 = 그리고 |Δf| ≥ {MIN_DF} Hz", ""]
tr = D[D.set == "train"]; va = D[D.set == "val"]
for nm in BANDS:
    txt.append(f"## {nm}")
    for lab, cond in [("지속음(선 ≥ 1 s)", f"{nm}|sustain"), ("미끄러지는 지속음", f"{nm}|slide")]:
        a = int(tr[tr.y == 0][cond].sum()); b = int(tr[tr.y == 1][cond].sum())
        txt.append(f"{lab}: 정상 {a}/120 ({a / 120:.0%}) · 고장 {b}/30 ({b / 30:.0%}) · val {int(va[cond].sum())}/40")
    sl = D[D[f"{nm}|slide"]].sort_values(f"{nm}|df", key=np.abs, ascending=False)
    for _, r in sl.iterrows():
        lab = "val" if r.set == "val" else ("고장" if r.y == 1 else "정상")
        txt.append(f"  {lab} {r.file}: 길이 {r[f'{nm}|len']:.2f} s · {r[f'{nm}|f0']:.0f} → {r[f'{nm}|f1']:.0f} Hz (Δf {r[f'{nm}|df']:+.0f}) · 기울기 {r[f'{nm}|slope']:+.0f} Hz/s")
    # 점검: 1 s 넘는 선 모두 중 |Δf| 최대 (가려짐 없이)
    m = D[f"{nm}|maxdf"]
    a = int((m[(D.set == "train") & (D.y == 0)] >= MIN_DF).sum()); b = int((m[(D.set == "train") & (D.y == 1)] >= MIN_DF).sum())
    txt.append(f"[점검] 1 s 넘는 선 중 하나라도 |Δf| ≥ {MIN_DF:.0f} Hz: 정상 {a}/120 · 고장 {b}/30 · val {int((m[D.set == 'val'] >= MIN_DF).sum())}/40 "
               f"→ val: {', '.join(D[(D.set == 'val') & (m >= MIN_DF)].file.str[-3:])}")
    for lab, cond in [("정상", (D.set == "train") & (D.y == 0)), ("고장", (D.set == "train") & (D.y == 1))]:
        top = D[cond].sort_values(f"{nm}|maxdf", ascending=False).head(3)
        txt.append(f"  train {lab} |Δf| 상위 3: " + ", ".join(f"{r.file[-3:]} {r[f'{nm}|maxdf']:.0f} Hz" for _, r in top.iterrows()))
    txt.append("")
open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8").write("\n".join(txt))

# %% 그림 1: 선 길이 × |Δf|
fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, nm in zip(axs, BANDS):
    for (cond, c, mk, lab) in [(D.set.eq("train") & D.y.eq(0), NOR, "s", "정상 (train)"), (D.set.eq("train") & D.y.eq(1), ABN, "o", "고장 (train)"),
                               (D.set.eq("val"), VAL, "^", "val (라벨 없음)")]:
        d = D[cond]
        ax.scatter(d[f"{nm}|len"], d[f"{nm}|df"].abs(), s=38, c=c, marker=mk, edgecolor="white", lw=.5, label=lab, alpha=.85, zorder=3)
    ax.axvline(MIN_LEN, color=INK, ls="--", lw=1); ax.axhline(MIN_DF, color=INK, ls="--", lw=1)
    for _, r in D[D[f"{nm}|slide"]].iterrows():
        ax.annotate(("V" if r.set == "val" else ("A" if r.y == 1 else "N")) + r.file[-3:], (r[f"{nm}|len"], abs(r[f"{nm}|df"])),
                    xytext=(4, 3), textcoords="offset points", fontsize=10)
    ax.set_xlabel("가장 긴 지속음 선의 길이 (s)"); ax.set_ylabel("그 선의 주파수 변화 |Δf| (Hz)")
    ax.set_title(f"{nm} — 오른쪽 위 = 미끄러지는 지속음", loc="left", fontsize=13)
    ax.grid(color=GRID); ax.set_axisbelow(True)
axs[0].legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_길이_변화.png"), dpi=150, facecolor="white"); plt.close(fig)

# %% 그림 2: 1 s 넘는 지속음 선 (시간 × 주파수). val 미끄러짐 5개 + 비교용 train 고정 지속음 정상 2개
nm = "전체 2–12.4 kHz"
m = D[f"{nm}|maxdf"] >= MIN_DF
pick = [("val", n) for n in D[(D.set == "val") & m].file] + [("train", "nor_116"), ("train", "nor_065")]
fig, axs = plt.subplots(1, len(pick), figsize=(2.3 * len(pick), 5.2), sharey=True)
for ax, (grp, n) in zip(axs, pick):
    x = Xv[nv.index(n)] if grp == "val" else X[names.index(n)]
    f, t, P = spec(x)
    for a, b_ in tracks(f, t, P, *BANDS[nm]):
        if (a[-1] - a[0]) * (t[1] - t[0]) >= MIN_LEN:
            d = abs(np.median(b_[-max(1, len(b_) // 10):]) - np.median(b_[:max(1, len(b_) // 10)]))
            ax.plot(t[a], b_ / 1000, lw=2.4 if d >= MIN_DF else 1.4, color=VAL if (grp == "val" and d >= MIN_DF) else "#9CA3AF")
    ax.set_title(("val " if grp == "val" else "정상 ") + n[-3:], fontsize=13, color=VAL if grp == "val" else NOR)
    ax.set_xlim(0, 2); ax.set_xticks([0, 1, 2]); ax.grid(color=GRID)
axs[0].set_ylabel("지속음 주파수 (kHz)"); axs[0].set_ylim(2, 12.4)
fig.suptitle("1 s 넘는 좁은 지속음 선 — 보라 굵은 선 = |Δf| ≥ 200 Hz (미끄러짐), 회색 = 고정 지속음 · 가로 = 시간 (s)", x=0.01, ha="left", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_지속음선.png"), dpi=150, facecolor="white"); plt.close(fig)
print("done")
