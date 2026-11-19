# %% [markdown]
# 2-16 보조: 울림으로 규칙 수치 정하기 (11-16, 사용자 요청 "물리적으로 확실한 수치")
# 질문 A. 8.3 kHz 모드 울림을 충격 하나 단위로 판단할 수 있는가 (002 = 충격 1개)
#         8.0–8.7 kHz로 울린 충격의 감쇠비 ζ · 맞춤 정도 fit · 높이를 고장과 정상에서 비교
# 질문 B. "충격이 모두 암 본체 모드(5–8 kHz대)로 울린다 = 노면 충격 = 정상"을 내리는 규칙으로 쓸 수 있는가
#         EXP-030 기준: C3 문턱 ±20%에서 고장이 걸리지 않을 것 · C4 측정 구간을 1.5초로 옮겨도 고장이 조건에 들지 않을 것
# - 입력 = figs/impacts.csv (2-16_run.py, 라벨 무관). 라벨은 집계에만 쓴다. 대역 후보는 2-16 그림 ①을 보고 정했다 = 사후
# 실행: conda run -n qe-phm python "단계/2_EDA/8kHz공진/2-16_1116_울림주파수/2-16_규칙수치.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs")
IM = pd.read_csv(os.path.join(FIG, "impacts.csv"), encoding="utf-8-sig")
FL = pd.read_csv(os.path.join(FIG, "files.csv"), index_col=0, encoding="utf-8-sig")
files = FL.index; yv = FL.y
out = []
def say(*a):
    s = " ".join(str(x) for x in a); out.append(s)

# %% A. 8.0–8.7 kHz로 울린 충격 하나하나
B8 = IM[(IM.f >= 8000) & (IM.f < 8700)].copy()
say("== A. 8.0–8.7 kHz로 울린 충격 ==")
say(f"고장 {int((B8.y == 1).sum())}개 ({B8[B8.y == 1].file.nunique()}파일) · 정상 {int((B8.y == 0).sum())}개 ({B8[B8.y == 0].file.nunique()}파일)")
say(B8[B8.y == 0][["file", "t", "height", "f", "zeta", "fit"]].round(3).to_string())
for c in ("height", "zeta", "fit"):
    a, n = B8[B8.y == 1][c], B8[B8.y == 0][c]
    say(f"  {c}: 고장 중앙 {a.median():.3f} (최소 {a.min():.3f}) · 정상 값 {sorted(n.round(3).tolist())}")
say("  비교: 전체 충격의 fit 중앙 — 정상 {:.3f} · 고장 {:.3f}".format(IM[IM.y == 0].fit.median(), IM[IM.y == 1].fit.median()))
say("  002의 충격:", IM[IM.file == "abn_002"][["height", "f", "zeta", "fit"]].round(3).values.tolist())
# 파일 단위 후보 규칙: 8.0–8.7 kHz 충격 중 (fit ≥ t_fit, ζ ≤ 0.02)가 하나라도 / 비율
for tf in (0.5, 0.6, 0.7, 0.8):
    q = B8[(B8.fit >= tf) & (B8.zeta <= 0.02)]
    say(f"  [8.0–8.7 kHz, ζ ≤ 0.02, fit ≥ {tf}] 충격이 하나라도 있는 파일: 고장 {sorted(q[q.y == 1].file.str[-3:].unique())} · 정상 {sorted(q[q.y == 0].file.str[-3:].unique())}")

# %% B. 정상 근거: 충격이 모두 본체 모드로 울림
say("\n== B. 본체 모드 울림 비율 (내리는 규칙 후보) ==")
BANDS = [(5000, 8000), (5500, 7750), (6000, 7750), (6500, 7750)]
win = {"0.0–1.5 s": (0.0, 1.5), "0.25–1.75 s": (0.25, 1.75), "0.5–2.0 s": (0.5, 2.0)}


def body_frac(lo, hi, sub=IM):
    b = sub.assign(b=(sub.f >= lo) & (sub.f < hi)).groupby("file").b.mean()
    return b.reindex(files)


rows = []
for lo, hi in BANDS:
    fr = body_frac(lo, hi)
    n = IM.groupby("file").size().reindex(files).fillna(0)
    for nmin in (3, 5):
        for th in (0.8, 0.9, 1.0):
            ok = (fr >= th) & (n >= nmin)
            # C3: 문턱 ±20% (아래로만 의미: 0.8·th), C4: 1.5초 창
            th_lo = th * 0.8
            c3 = ((fr >= th_lo) & (n >= nmin) & (yv == 1))
            c4 = pd.Series(False, index=files)
            for (a, b) in win.values():
                sub = IM[(IM.t >= a) & (IM.t < b)]
                fw = body_frac(lo, hi, sub); nw = sub.groupby("file").size().reindex(files).fillna(0)
                c4 |= (fw >= th) & (nw >= nmin) & (yv == 1)
            fp_fix = [f[-3:] for f in ("nor_025", "nor_055", "nor_059", "nor_116") if ok.get(f, False)]
            rows.append(dict(대역=f"{lo / 1000:.2f}–{hi / 1000:.2f}", n_min=nmin, 문턱=th, 정상=int((ok & (yv == 0)).sum()), 고장=int((ok & (yv == 1)).sum()),
                             C3_고장=",".join(sorted(f[-3:] for f in files[c3])) or "-", C4_고장=",".join(sorted(f[-3:] for f in files[c4.fillna(False)])) or "-",
                             오경보_해소=",".join(fp_fix) or "-"))
T = pd.DataFrame(rows)
say(T.to_string(index=False))
say("\n가장 높은 고장 (5.5–7.75 kHz 비율, 충격 3개 이상):")
fr = body_frac(5500, 7750); n = IM.groupby("file").size().reindex(files).fillna(0)
say(pd.DataFrame({"비율": fr, "충격": n})[(yv == 1) & (n >= 3)].sort_values("비율", ascending=False).head(6).round(2).to_string())
say("오경보 정상:", pd.DataFrame({"비율": fr, "충격": n}).loc[["nor_025", "nor_055", "nor_059", "nor_116"]].round(2).to_string())
T.to_csv(os.path.join(FIG, "2-16_규칙후보_정상근거.csv"), index=False, encoding="utf-8-sig")

# %% 그림 A: 8.0–8.7 kHz 충격 — 높이 × fit, 색 = 정상/고장, 크기 = 1/ζ
fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
a = axs[0]
for lab, c, nm in ((0, "#2a78d6", "정상"), (1, "#eb6834", "고장")):
    s = IM[IM.y == lab]
    a.scatter(s.fit, s.height, s=8, color=c, alpha=0.15)
    s8 = B8[B8.y == lab]
    a.scatter(s8.fit, s8.height, s=40 / np.sqrt(s8.zeta / 0.01), color=c, edgecolor="k", lw=0.6, label=f"{nm}: 8.0–8.7 kHz로 울린 충격 {len(s8)}개")
for _, r in B8.iterrows():
    if r.y == 0 or r.file == "abn_002":
        a.annotate(r.file[-3:] if r.y else "N" + r.file[-3:], (r.fit, r.height), xytext=(4, 4), textcoords="offset points", fontsize=8)
a.set_yscale("log"); a.set_xlabel("감쇠 사인파 맞춤 정도 fit (1 = 완벽한 감쇠 사인파)"); a.set_ylabel("충격 높이 (포락선 바탕의 몇 배, log)")
a.set_title("A. 8.3 kHz 모드로 울린 충격 하나하나 (흐린 점 = 다른 충격, 점 크기 클수록 ζ 작음 = 오래 울림)", loc="left", fontsize=10); a.legend(fontsize=8)
a = axs[1]
fr = body_frac(5500, 7750)
rng = np.random.default_rng(0)
for lab, c, nm in ((0, "#2a78d6", "정상"), (1, "#eb6834", "고장")):
    m = (yv == lab) & (n >= 3)
    a.scatter(n[m] * np.exp(rng.uniform(-.06, .06, m.sum())), fr[m], s=24, color=c, alpha=.7, label=f"{nm} (충격 3개 이상)")
for f in ("nor_025", "nor_055", "nor_059", "nor_116", "abn_028", "abn_009", "abn_005", "abn_022"):
    if n[f] >= 1:
        a.annotate(f, (n[f], fr[f] if not np.isnan(fr[f]) else 0), xytext=(5, -3), textcoords="offset points", fontsize=8, color="#d62728" if f[0] == "n" else "#9467bd")
a.axhline(0.9, color="k", lw=1); a.axhspan(0.72, 1.0, color="#999", alpha=.15, lw=0, label="문턱 0.9의 −20% 아래까지")
a.set_xscale("log"); a.set_xlabel("충격 수 (log)"); a.set_ylabel("충격 중 5.5–7.75 kHz(본체 모드)로 울린 비율")
a.set_title("B. 정상 근거 후보 — 선 위 = '충격이 모두 본체 모드로 울림'", loc="left", fontsize=10); a.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2-16_6_규칙수치.png"), dpi=125); plt.close(fig)
with open(os.path.join(FIG, "2-16_규칙수치.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))
