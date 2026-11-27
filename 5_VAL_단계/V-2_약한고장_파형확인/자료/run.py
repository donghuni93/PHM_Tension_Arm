# %% [markdown]
# V-2 약한 고장 후보 파형 확인 (11-27)
# - 질문: V-1에서 놓친 고장 후보(8조만 010 · 034 · 036, 징후 하나만 있는 8개)의 충격은 진짜 고장 충격(약한 판)인가, 정상의 잡음인가
# - 방법: 제출 모델과 같은 충격 검출기(1–11 kHz 대역통과, src.impacts.all_impacts fmax=11000)로 파형 · 충격 확대 · 충격 세기 분포를 본다
#   기준 = val의 확실한 고장(039 · 016) · train 고장 · train 정상(징후 하나만 있는 정상 포함)
# - 라벨 없는 val 후보 목록은 V-1 결과(다른 조와 겹쳐 본 그룹 + 징후 하나)에서 온다. 진단용, 모델 튜닝에 쓰지 않는다
#   conda run -n qe-phm python 5_VAL_단계/V-2_약한고장_파형확인/자료/run.py

# %% 설정
import os, sys, json
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
import src.viz  # noqa: F401 (한글 글꼴)
import matplotlib.pyplot as plt
from src.data import load_dataset, load_folder
from src.impacts import all_impacts

FS = 25000; FMAX = 11000
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
m = joblib.load(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "모델", "final_model_062.joblib")); core = m.core_
Ttr, Tva, y, ntr, nva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))   # V-1이 만든 캐시
cols = list(core.cols) + list(core.peak)
Etr, _ = core._et_input(Ttr); Eva, _ = core._et_input(Tva)
nm = y == 0
med = np.median(Etr[nm], 0); mad = 1.4826 * np.median(np.abs(Etr[nm] - med), 0); mad[mad < 1e-9] = Etr[nm].std(0)[mad < 1e-9] + 1e-9
Ztr, Zva = (Etr - med) / mad, (Eva - med) / mad
i3, i8 = cols.index("H3_kurt_1-2k"), cols.index("H8_ring_2-4k")
pva = core.et_.predict_proba(Eva)[:, 1]

# %% 신호 · 충격 (train 150 + val 40, 같은 검출기)
sig_cache = os.path.join(ROOT, "4_outputs", "cache", "val_eda_imps.joblib")
if os.path.exists(sig_cache):
    Itr, Iva = joblib.load(sig_cache)
else:
    Xtr, _, _ = load_dataset(os.path.join(ROOT, "2_data", "TeamProject_train_변경"))
    Xva, _ = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
    Itr, Iva = all_impacts(Xtr, fmax=FMAX), all_impacts(Xva, fmax=FMAX)
    joblib.dump((Itr, Iva), sig_cache)

def summ(im):
    """파일 하나의 충격 요약: 수 · 최대 배수 · 2–4 kHz 울림 충격 수 · 그중 최대 배수 · 2–4 kHz 울림 비율"""
    hx = np.asarray(im["height_x"]); f = im["rings"][:, 0] if len(im["rings"]) else np.zeros(0)
    k = min(len(hx), len(f)); hx, f = hx[:k], f[:k]; r24 = (f >= 2000) & (f < 4000)
    return dict(n=k, hx_max=float(hx.max()) if k else 0.0, hx_med=float(np.median(hx)) if k else 0.0,
                n24=int(r24.sum()), hx24=float(hx[r24].max()) if r24.any() else 0.0, frac24=float(r24.mean()) if k else 0.0)

Str = pd.DataFrame([summ(im) for im in Itr]); Str["name"] = ntr; Str["y"] = y
Sva = pd.DataFrame([summ(im) for im in Iva]); Sva["name"] = nva

# %% 볼 파일 (그룹)
both_strong = ["val_039", "val_016"]; only8 = ["val_010", "val_036", "val_034"]
h3_only = [n for n in nva if Zva[nva.index(n), i3] > 3 and Zva[nva.index(n), i8] <= 5 and n != "val_031"]
h8_only = [n for n in nva if Zva[nva.index(n), i8] > 5 and Zva[nva.index(n), i3] <= 3]
# 기준 정상: val 점수 최저 1개 + train 정상 중 징후 하나만 있는 파일에서 각 징후가 가장 큰 1개
ev = pd.read_csv(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "출력", "근거표_submission_val_9조.csv"))
val_low = ev.sort_values("score").file.iloc[0]
tn = np.where(nm)[0]
n_h3 = ntr[tn[np.argmax(np.where(Ztr[tn, i8] <= 5, Ztr[tn, i3], -np.inf))]]
n_h8 = ntr[tn[np.argmax(np.where(Ztr[tn, i3] <= 3, Ztr[tn, i8], -np.inf))]]
SHOW = ([(n, "기준 고장(val)") for n in both_strong] + [(n, "8조만 고장") for n in only8] + [(n, "H3만") for n in h3_only]
        + [(n, "H8만") for n in h8_only] + [(val_low, "기준 정상(val 최저)"), (n_h3, "train 정상 H3만"), (n_h8, "train 정상 H8만")])
COL = {"기준 고장(val)": "C3", "8조만 고장": "C4", "H3만": "C2", "H8만": "C9", "기준 정상(val 최저)": "0.4",
       "train 정상 H3만": "C0", "train 정상 H8만": "C0"}

def get(n):
    if n in nva:
        i = nva.index(n); return Iva[i], Sva.iloc[i], Zva[i], pva[i]
    i = ntr.index(n); return Itr[i], Str.iloc[i], Ztr[i], float(core.Z_[i, 0])   # train은 OOF logit

# %% 그림 1: 파형 + 충격 표시 (1–11 kHz)
t = np.arange(50000) / FS
fig, axs = plt.subplots(len(SHOW), 1, figsize=(13, 1.05 * len(SHOW)), sharex=True)
for a, (n, g) in zip(axs, SHOW):
    im, s, z, pp = get(n); x = im["hp"]
    a.plot(t, x, lw=.3, color=COL[g])
    if len(im["idx"]):
        a.plot(im["t"], x[im["idx"]] * 0 + np.abs(x).max() * 1.05, "v", ms=3, color="k")
    lab = f"ET p {pp:.2f}" if n.startswith("val") else f"OOF logit {pp:+.1f}"
    a.set_title(f"{n} · {g} · 충격 {s.n}개 (2–4k {s.n24}) · 최대 {s.hx_max:.1f}배 · H3 z {z[i3]:+.1f} · H8 z {z[i8]:+.1f} · {lab}",
                fontsize=8, loc="left", pad=2)
    a.set_yticks([]); a.margins(x=0)
axs[-1].set_xlabel("초  (▼ = 검출된 충격, 세로 눈금은 파일마다 다름)")
fig.tight_layout(h_pad=.3); fig.savefig(os.path.join(FIG, "파형_충격.png"), dpi=130); plt.close(fig)

# %% 그림 2: 가장 큰 충격 확대 (−2 – +12 ms)
nc = 4; nr = int(np.ceil(len(SHOW) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(13, 2.2 * nr), squeeze=False)
tt = (np.arange(-50, 300)) / FS * 1000
for a, (n, g) in zip(axs.flat, SHOW):
    im, s, z, pp = get(n); x = im["hp"]
    if len(im["idx"]) == 0:
        a.text(.5, .5, "충격 없음", ha="center", transform=a.transAxes); a.set_title(n, fontsize=8); continue
    j = int(np.argmax(im["height_x"])); p0 = int(im["idx"][j]); seg = x[max(0, p0 - 50):p0 + 300]
    a.plot(tt[:len(seg)], seg, lw=.6, color=COL[g])
    f, zeta, fit = im["rings"][j] if j < len(im["rings"]) else (np.nan, np.nan, np.nan)
    a.set_title(f"{n} ({g})\n{im['height_x'][j]:.1f}배 · 울림 {f/1000:.2f} kHz · ζ {zeta:.3f} · fit {fit:.2f}", fontsize=7.5)
    a.tick_params(labelsize=7)
for a in axs.flat[len(SHOW):]:
    a.axis("off")
fig.supxlabel("충격 뒤 시간 (ms)", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(FIG, "충격_확대.png"), dpi=130); plt.close(fig)

# %% 그림 3: 충격 세기 분포 — train 정상 · 고장 위에 val 후보
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for a, (xc, yc, xl, yl) in zip(axs, [("n", "hx_max", "충격 수", "가장 큰 충격 (포락선/바탕 배수)"),
                                      ("n24", "hx24", "2–4 kHz로 운 충격 수", "그중 가장 큰 충격 (배수)")]):
    for lab, msk, c in [("train 정상", Str.y == 0, "C0"), ("train 고장", Str.y == 1, "C1")]:
        a.scatter(Str[xc][msk] + 1, Str[yc][msk].clip(lower=1), s=14, alpha=.45, color=c, label=lab)
    for n, g in SHOW:
        if not n.startswith("val"):
            continue
        s = Sva.iloc[nva.index(n)]
        a.scatter(s[xc] + 1, max(s[yc], 1), s=55, color=COL[g], edgecolor="k", lw=.6, zorder=3, label=g)
        a.annotate(n[4:], (s[xc] + 1, max(s[yc], 1)), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(xl + " + 1 (로그)"); a.set_ylabel(yl + " (로그, 1 미만은 1)"); a.grid(alpha=.3)
h, l = axs[0].get_legend_handles_labels(); u = dict(zip(l, h))
axs[1].legend(u.values(), u.keys(), fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "충격세기_분포.png"), dpi=130); plt.close(fig)

# %% 표 + 정상 분포 대비 백분위
def pct(v, ref):
    return float((ref < v).mean() * 100)
rows = []
for n, g in SHOW:
    im, s, z, pp = get(n)
    rows.append(dict(file=n, 그룹=g, 충격=int(s.n), 충격_2_4k=int(s.n24), 최대배수=round(s.hx_max, 1), 최대배수_2_4k=round(s.hx24, 1),
                     최대배수_정상백분위=round(pct(s.hx_max, Str.hx_max[Str.y == 0]), 0),
                     최대배수_2_4k_정상백분위=round(pct(s.hx24, Str.hx24[Str.y == 0]), 0),
                     H3z=round(float(z[i3]), 1), H8z=round(float(z[i8]), 1), ET=round(float(pp), 2)))
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "파일표.csv"), index=False, encoding="utf-8-sig")
ref = {k: {q: float(np.percentile(Str[k][Str.y == yy], q)) for q in (50, 90)} for k in ("hx_max", "hx24", "n24") for yy in (0,)}
reff = {k: {q: float(np.percentile(Str[k][Str.y == 1], q)) for q in (10, 50)} for k in ("hx_max", "hx24", "n24")}
json.dump(dict(train정상_분위=ref, train고장_분위=reff, 파일=rows), open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
from src.exp import md_table
open(os.path.join(HERE, "파일표.md"), "w", encoding="utf-8").write(md_table(T))
print("done")

# %% 그림 4 (11-27 추가): 고립 강타 비교 — train 정상(충격 ≤ 4 · 최대 > 30배) vs val 후보
iso_tr = [(n, "train 정상" if yy == 0 else "train 고장") for n, yy, im in zip(ntr, y, Itr)
          if 0 < len(im["height_x"]) <= 4 and max(im["height_x"]) > 30]
iso_va = [(n, "val") for n, im in zip(nva, Iva) if 0 < len(im["height_x"]) <= 4 and max(im["height_x"]) > 30]
ISO = iso_tr + iso_va
fig, axs = plt.subplots(len(ISO), 2, figsize=(13, 1.25 * len(ISO)), gridspec_kw=dict(width_ratios=[3, 1]))
for (a, b), (n, g) in zip(axs, ISO):
    im = Itr[ntr.index(n)] if n in ntr else Iva[nva.index(n)]; x = im["hp"]; c = {"train 정상": "C0", "train 고장": "C1", "val": "C4"}[g]
    j = int(np.argmax(im["height_x"])); p0 = int(im["idx"][j]); f, zeta, fit = im["rings"][j]
    a.plot(t, x, lw=.3, color=c); a.plot(im["t"], np.full(len(im["t"]), np.abs(x).max() * 1.05), "v", ms=3, color="k")
    a.set_yticks([]); a.margins(x=0)
    a.set_title(f"{n} · {g} · 충격 {len(im['t'])}개 · 최대 {im['height_x'][j]:.0f}배 · {f/1000:.2f} kHz · ζ {zeta:.3f} · fit {fit:.2f}", fontsize=8, loc="left", pad=2)
    seg = x[max(0, p0 - 50):p0 + 300]; b.plot(tt[:len(seg)], seg, lw=.5, color=c); b.set_yticks([]); b.tick_params(labelsize=7)
axs[-1, 0].set_xlabel("초"); axs[-1, 1].set_xlabel("가장 큰 충격 뒤 ms")
fig.tight_layout(h_pad=.3); fig.savefig(os.path.join(FIG, "고립강타_비교.png"), dpi=130); plt.close(fig)
json.dump([dict(file=n, 그룹=g) for n, g in ISO], open(os.path.join(HERE, "고립강타_목록.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("iso done")


