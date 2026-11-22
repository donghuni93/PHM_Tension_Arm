# %% [markdown]
# 2-29 (11-22) 울림 대역 경계 점검 — 2 kHz 몰림은 실제 울림인가, 250 Hz 격자의 산물인가 (6-6 V025 진단에서 출발, 라벨은 그룹 비교에만)
# 질문 1. 울림 주파수를 촘촘하게 다시 재면 1.9–2.1 kHz에 뾰족하게 몰리는가 (정상 · 고장 · val)
#   - 지금 정의(impact_rings_v2): 충격 뒤 8 ms를 감쇠 사인파 사전(1–12 kHz 250 Hz 간격 × ζ 4개)에 투영해 가장 잘 맞는 f. 2000 Hz는 2–4k로 셈(octave_of)
#   - 촘촘한 맞춤: 같은 사전을 50 Hz 간격으로 · FFT 봉우리: 8 ms(Hann)를 8192점으로 0 채움 → 1–12 kHz 최대
# 질문 2. val의 2 kHz 충격은 몇몇 파일에 몰렸나, 40개 전반에 퍼졌나
# 질문 3. 경계 처리를 바꾸면 누가 움직이나 (라벨 없이 정한 두 방식): (가) 격자점이 경계(2000 · 4000 · 8000)와 같은 충격은 양쪽에 0.5씩
#   (나) 촘촘한 맞춤 f로 대역을 다시 정함. H7(1–2k) · H8(2–4k) 울림 비율이 어떻게 바뀌나
# 실행: conda run -n qe-phm python run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp import md_table
from src.exp2util import NS
from src.impacts import all_impacts
from src.final import _SEG
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
NOR, ABN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a63d2", "#1d2430", "#5b6575", "#d9dde3"
FS = 25000; EDGES = (2000, 4000, 8000)
ns = NS(); y = ns.y; names = list(ns.names); nvn = [f"val_{v[-3:]}" for v in ns.nv]
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")


# %% 촘촘한 사전 (50 Hz) — final._ring_atoms와 같은 식, 주파수 간격만 다름
def atoms(step):
    t = np.arange(_SEG) / FS; A, P = [], []
    for f in np.arange(1000, 12001, step):
        for z in (0.01, 0.02, 0.04, 0.08):
            w = 2 * np.pi * f; wd, a = w * np.sqrt(1 - z ** 2), np.exp(-z * w * t)
            Q, _ = np.linalg.qr(np.stack([a * np.sin(wd * t), a * np.cos(wd * t)], 1)); A.append(Q); P.append((f, z))
    return np.stack(A), np.array(P)
A50, P50 = atoms(50)
win = np.hanning(_SEG); NF = 8192; fq = np.fft.rfftfreq(NF, 1 / FS); band = (fq >= 1000) & (fq <= 12000)


def per_impact(ims, grp, fnames):
    rows = []
    for im, nm in zip(ims, fnames):
        xh = im["hp"]; pk = [p for p in im["idx"] if p + _SEG <= len(xh)]
        if not pk: continue
        S = np.stack([xh[p:p + _SEG] for p in pk]); S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        fit = np.linalg.norm(np.einsum("ks,ast->kat", S, A50), axis=2); f50 = P50[fit.argmax(1), 0]
        sp = np.abs(np.fft.rfft(S * win, NF, axis=1))[:, band]; ffft = fq[band][sp.argmax(1)]
        for j in range(len(pk)):
            rows.append(dict(file=nm, grp=grp, f250=float(im["rings"][j, 0]), f50=float(f50[j]), ffft=float(ffft[j]), g=float(im["height_g"][j]) if j < len(im["height_g"]) else np.nan))
    return rows


IT, IV = all_impacts(ns.X), all_impacts(ns.Xv)
D = pd.DataFrame(per_impact([im for im, yy in zip(IT, y) if yy == 0], "정상", [n for n, yy in zip(names, y) if yy == 0])
                 + per_impact([im for im, yy in zip(IT, y) if yy == 1], "고장", [n for n, yy in zip(names, y) if yy == 1])
                 + per_impact(IV, "val", nvn))
D.to_csv(os.path.join(FIG, "impacts_fine.csv"), index=False, encoding="utf-8-sig")
G = {"정상": NOR, "고장": ABN, "val": VAL}; NG = {"정상": 120, "고장": 30, "val": 40}

# %% 질문 1 표: 지금 2000 Hz로 잡힌 충격의 촘촘한 f · 1.5–2.5 kHz 분포
q1 = []
for gname in G:
    d = D[D.grp == gname]; e = d[d.f250 == 2000]
    q1.append({"그룹": gname, "충격": len(d), "지금 2000 Hz": f"{len(e)} ({len(e) / len(d) * 100:.0f} %)",
               "그중 촘촘한 f 중앙 [사분위]": f"{e.f50.median():.0f} [{e.f50.quantile(.25):.0f}–{e.f50.quantile(.75):.0f}]" if len(e) else "—",
               "그중 촘촘한 f < 2000": f"{int((e.f50 < 2000).sum())}/{len(e)}", "그중 FFT 봉우리 < 2000": f"{int((e.ffft < 2000).sum())}/{len(e)}",
               "1.9–2.1 kHz (촘촘한 f)": f"{int(((d.f50 >= 1900) & (d.f50 <= 2100)).sum())} ({((d.f50 >= 1900) & (d.f50 <= 2100)).mean() * 100:.0f} %)"})
T1 = pd.DataFrame(q1)

# %% 질문 2: 파일별 2 kHz 근처(촘촘한 f 1.9–2.1 kHz) 충격 수
near = (D.f50 >= 1900) & (D.f50 <= 2100)
cnt = D.assign(n2=near).groupby("file").agg(n=("f50", "size"), n2=("n2", "sum"))
allf = [(n, "정상") for n, yy in zip(names, y) if yy == 0] + [(n, "고장") for n, yy in zip(names, y) if yy == 1] + [(n, "val") for n in nvn]
C = pd.DataFrame([dict(file=f, grp=gp, n=int(cnt.n.get(f, 0)), n2=int(cnt.n2.get(f, 0))) for f, gp in allf])
q2 = []
for gname in G:
    c = C[C.grp == gname]; tot = c.n2.sum(); top = c.sort_values("n2", ascending=False)
    top3 = top.n2.head(3).sum()
    q2.append({"그룹": gname, "파일": len(c), "2 kHz 충격 있는 파일": int((c.n2 > 0).sum()), "2 kHz 충격 합": int(tot),
               "상위 3파일 몫": f"{top3}/{tot} ({top3 / max(tot, 1) * 100:.0f} %)", "상위 파일": " · ".join(f"{short(r.file)} {r.n2}/{r.n}" for r in top.head(5).itertuples() if r.n2 > 0)})
T2 = pd.DataFrame(q2)

# %% 질문 3: 경계 처리 바꾸면 H7 · H8이 어떻게 움직이나
def ratios(d, mode):
    """파일 하나의 충격 → 울림 비율 4칸. mode: 'now' = 지금(격자 250, 2000은 2–4k) · 'split' = 경계 격자점은 양쪽 0.5 · 'fine' = 촘촘한 f로 대역"""
    w = np.zeros(4)
    for f250, f50 in zip(d.f250, d.f50):
        if mode == "fine": w[np.digitize(f50, EDGES)] += 1
        elif mode == "split" and f250 in EDGES: b = EDGES.index(f250); w[b] += .5; w[b + 1] += .5
        else: w[np.digitize(f250, EDGES)] += 1
    return w / max(len(d), 1)
R = []
for f, gp in allf:
    d = D[D.file == f]
    if not len(d): continue
    r0, r1, r2 = ratios(d, "now"), ratios(d, "split"), ratios(d, "fine")
    R.append(dict(file=f, grp=gp, n=len(d), H7=r0[0], H8=r0[1], H7_split=r1[0], H8_split=r1[1], H7_fine=r2[0], H8_fine=r2[1]))
R = pd.DataFrame(R)
q3 = []
for gname in G:
    r = R[R.grp == gname]
    for lab, col in (("(가) 경계 반씩", "H8_split"), ("(나) 촘촘한 f", "H8_fine")):
        dlt = r[col] - r.H8
        q3.append({"그룹": gname, "방식": lab, "H8 바뀐 파일": f"{int((dlt.abs() > 1e-9).sum())}/{len(r)}", "H8 평균 변화": round(float(dlt.mean()), 3),
                   "H8 0.2 넘게 내려간 파일": int((dlt < -.2).sum()), "H8 중앙 (전 → 후)": f"{r.H8.median():.2f} → {r[col].median():.2f}"})
T3 = pd.DataFrame(q3)
FA = R[(R.grp == "고장") & ((R.H8_split - R.H8).abs() > 1e-9)].assign(d_split=lambda t: t.H8_split - t.H8, d_fine=lambda t: t.H8_fine - t.H8)
T4 = FA[["file", "n", "H8", "H8_split", "H8_fine"]].round(2).assign(file=lambda t: t.file.map(short)).sort_values("H8", ascending=False)
R.round(3).to_csv(os.path.join(FIG, "ring_ratio_3ways.csv"), index=False, encoding="utf-8-sig")

open(os.path.join(HERE, "표.md"), "w", encoding="utf-8").write("\n".join([
    "# 2-29 표\n", "## 질문 1. 지금 2000 Hz로 잡힌 충격을 촘촘하게 다시 재면\n", md_table(T1),
    "\n(촘촘한 f = 같은 감쇠 사인파 사전을 50 Hz 간격으로, FFT 봉우리 = 8 ms Hann을 8192점 0 채움. 8 ms 창의 실제 분해능은 약 125 Hz)\n",
    "## 질문 2. 파일별 2 kHz 근처(촘촘한 f 1.9–2.1 kHz) 충격\n", md_table(T2),
    "\n## 질문 3. 경계 처리를 바꾸면 H8(2–4k 울림 비율)이 어떻게 움직이나\n", md_table(T3),
    "\n### 2 kHz 충격이 있는 고장 파일의 H8 (지금 · 경계 반씩 · 촘촘한 f)\n", md_table(T4), ""]))

# %% 그림 1: 울림 주파수 분포 (1–3 kHz 확대) — 지금 격자 · 촘촘한 맞춤 · FFT 봉우리
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=False)
for ax, col, lab, bins in ((axs[0], "f250", "지금 정의 (250 Hz 격자)", np.arange(875, 3126, 250)), (axs[1], "f50", "촘촘한 맞춤 (50 Hz 격자)", np.arange(975, 3026, 50)),
                           (axs[2], "ffft", "FFT 봉우리 (8 ms, 0 채움)", np.arange(975, 3026, 50))):
    for gname, c in G.items():
        v = D[D.grp == gname][col]; ax.hist(v[(v >= 1000) & (v <= 3000)], bins=bins, weights=np.full(int(((v >= 1000) & (v <= 3000)).sum()), 100 / len(v)),
                                            histtype="step", lw=1.8, color=c, label=f"{gname} (충격 {len(v)})")
    ax.axvline(2000, color=INK, ls="--", lw=1); ax.text(2010, ax.get_ylim()[1] * .95, "1–2k | 2–4k 경계", fontsize=8, color=INK, va="top")
    ax.set_title(lab, loc="left", fontsize=10); ax.set_xlabel("울림 주파수 (Hz)", color=MUTED, fontsize=9); ax.grid(alpha=.25); ax.tick_params(labelsize=8)
axs[0].set_ylabel("그룹 충격 중 비율 (%)", color=MUTED, fontsize=9); axs[0].legend(fontsize=8)
fig.suptitle("그림 1 — 1–3 kHz 울림 주파수 분포 (그룹마다 전체 충격 대비 %). 왼쪽 = 지금 피처가 쓰는 값, 가운데 · 오른쪽 = 같은 충격을 촘촘하게 다시 잰 값", x=.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(os.path.join(FIG, "1_울림분포_1-3k.png"), dpi=150); plt.close(fig)

# %% 그림 2: 파일별 2 kHz 근처 충격 수 (그룹별, 많은 순)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.2), gridspec_kw=dict(width_ratios=[120, 30, 40]))
for ax, gname in zip(axs, G):
    c = C[C.grp == gname].sort_values(["n2", "n"], ascending=False)
    ax.bar(range(len(c)), c.n, color=GRID, label="충격 전체"); ax.bar(range(len(c)), c.n2, color=G[gname], label="그중 1.9–2.1 kHz")
    top = c[c.n2 > 0].head(8)
    for k, r in enumerate(top.itertuples()): ax.text(k, min(r.n, 30) + .3, short(r.file), rotation=90, fontsize=7, ha="center", va="bottom")
    ax.set_title(f"{gname} {len(c)}개 — 2 kHz 충격 있는 파일 {int((c.n2 > 0).sum())}개", loc="left", fontsize=10)
    ax.set_xticks([]); ax.set_ylim(0, 36); ax.tick_params(labelsize=8); ax.grid(axis="y", alpha=.25)
axs[0].set_ylabel("충격 수 (30에서 자름)", color=MUTED, fontsize=9); axs[0].legend(fontsize=8)
fig.suptitle("그림 2 — 파일별 충격 수와 그중 2 kHz 근처(촘촘한 f 1.9–2.1 kHz) 충격 수. 파일은 2 kHz 충격 많은 순", x=.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(os.path.join(FIG, "2_파일별_2k충격.png"), dpi=150); plt.close(fig)

# %% 그림 3: H8 전 → 후 (두 방식)
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, col, lab in ((axs[0], "H8_split", "(가) 경계 격자점 충격을 양쪽에 0.5씩"), (axs[1], "H8_fine", "(나) 촘촘한 f로 대역을 다시 정함")):
    for gname, mk in (("정상", "s"), ("고장", "o"), ("val", "^")):
        r = R[R.grp == gname]; ax.scatter(r.H8, r[col], marker=mk, s=34 if mk != "^" else 50, color=G[gname], edgecolor="white", lw=.5, alpha=.85, label=gname, zorder=3)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls="--")
    for r in R[(R[col] - R.H8).abs() > .25].itertuples():
        ax.annotate(short(r.file), (r.H8, getattr(r, col)), xytext=(4, -3), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel("H8 지금 (2–4k 울림 비율)", color=MUTED, fontsize=9); ax.set_ylabel("H8 바꾼 뒤", color=MUTED, fontsize=9)
    ax.set_title(lab, loc="left", fontsize=10); ax.grid(alpha=.25); ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03); ax.tick_params(labelsize=8)
axs[0].legend(fontsize=8, loc="upper left")
fig.suptitle("그림 3 — 경계 처리를 바꿨을 때 파일별 H8. 점선 아래 = 2–4k 비율이 내려감(1–2k로 넘어감). 이름 = 0.25 넘게 움직인 파일. 충격 0개 파일은 뺌", x=.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(os.path.join(FIG, "3_H8_전후.png"), dpi=150); plt.close(fig)
print("done", flush=True)
