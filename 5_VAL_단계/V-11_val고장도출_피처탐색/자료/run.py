# %% [markdown]
# V-11 val 고장을 끌어내는 피처 탐색: 첨도 · 윈도우 · 스펙트로그램 (11-27)
# - 질문: train에서 "정상에도 1–2개 있다"는 이유로 버린 징후 중, val에서 아직 고장으로 판정하지 않은 30개 파일을 끌어올리는 것이 있나
# - 방법: 피처 정의는 val을 보기 전에 고정한다(대역 = 옥타브 1–2 · 2–4 · 4–8 · 8–11 kHz + 1–11 kHz, 창 20 · 50 · 100 · 200 ms, STFT 256 · 512 · 1024)
#   선 = train 정상 120개 중 2개만 넘는 값(3번째로 큰 값). 위 · 아래 두 방향
#   셈 = 그 선을 넘는 train 정상 · train 고장 · val 판정 고장 10개(F) · val 미판정 30개(U)
#   우연 기준 = U가 모두 정상이면 넘는 파일 기대 30 × 2/120 = 0.5개. 이항 p
#   이동 점검 = val U 중앙값의 train 정상 기준 강건 z. |z| ≥ 1이면 "val 전체 이동"(측정 조건 차이 의심)
#   U 안 고장은 9개(17 − 우리 TP 8)이므로 U 초과가 12개를 넘으면 이동으로 본다
# - val 라벨은 없다. 다른 조 결과로 추론한 그룹은 쓰지 않는다. 진단 · 후보 찾기용이며 문턱 튜닝에 쓰지 않는다
#   conda run -n qe-phm python 5_VAL_단계/V-11_val고장도출_피처탐색/자료/run.py

# %% 설정
import os, sys, json
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy import signal as ss, stats
from scipy.linalg import solve_toeplitz
import src.viz  # noqa: F401 (한글 글꼴)
import matplotlib.pyplot as plt

FS = 25000
CACHE = os.path.join(ROOT, "4_outputs", "cache")
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
z = np.load(os.path.join(CACHE, "val_eda_raw.npz"), allow_pickle=True)
Xtr, y, ntr, Xva, nva = z["Xtr"], z["y"], list(z["ntr"]), z["Xva"], list(z["nva"])
ev = pd.read_csv(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "출력", "근거표_submission_val_9조.csv")).set_index("file")
pred = ev.loc[nva, "pred"].values.astype(int)            # 제출 판정 (고장 10개)

BANDS = {"1-2k": (1000, 2000), "2-4k": (2000, 4000), "4-8k": (4000, 8000), "8-11k": (8000, 11000), "1-11k": (1000, 11000)}
SOS = {b: ss.butter(4, f, btype="bandpass", fs=FS, output="sos") for b, f in BANDS.items()}
WMS = [20, 50, 100, 200]
TRIM = 500                                                # 양 끝 20 ms 버림 (필터 끝단)
kurt = lambda a, ax=-1: stats.kurtosis(a, axis=ax, fisher=False)

def win(a, n):
    """창 길이 n, 절반 겹침"""
    return np.lib.stride_tricks.sliding_window_view(a, n)[:: n // 2]

def whiten(a, p=40):
    s = a[:20000]; r = np.array([s[: 20000 - k] @ s[k:] for k in range(p + 1)]) / 20000   # 지연 0..p만 (full correlate와 같은 값)
    c = solve_toeplitz(r[:p], r[1:p + 1])
    return ss.lfilter(np.r_[1, -c], 1, a)[p:]

# %% 피처 (파일 하나)
def feats(x):
    x = x - x.mean(); d = {}
    xb_all = {}
    for b in BANDS:
        xb = ss.sosfiltfilt(SOS[b], x)[TRIM:-TRIM]; xb_all[b] = xb
        env = np.abs(ss.hilbert(xb))
        d[f"K_{b}"] = kurt(xb); d[f"Kenv_{b}"] = kurt(env)
        for w in WMS:
            W = win(xb, w * 25); k = kurt(W); r = np.sqrt((W ** 2).mean(1)); med = np.median(r)
            d[f"WK{w}_{b}_max"] = k.max(); d[f"WK{w}_{b}_p90"] = np.percentile(k, 90); d[f"WK{w}_{b}_mean"] = k.mean()
            d[f"WR{w}_{b}_crest"] = r.max() / med; d[f"WR{w}_{b}_kurt"] = kurt(r); d[f"WR{w}_{b}_n4"] = float((r > 4 * med).sum())
    # AR(40) 백색화 (1–11 kHz)
    e = whiten(xb_all["1-11k"]); d["AR_K"] = kurt(e)
    for w in [20, 50, 100]:
        k = kurt(win(e, w * 25)); d[f"ARWK{w}_max"] = k.max(); d[f"ARWK{w}_p90"] = np.percentile(k, 90); d[f"ARWK{w}_mean"] = k.mean()
    # 스펙트로그램
    for N in [256, 512, 1024]:
        f, t, Z = ss.stft(x, FS, window="hann", nperseg=N, noverlap=N // 2, boundary=None, padded=False)
        P = (np.abs(Z) ** 2)[:, 2:-2]
        sel = (f >= 1000) & (f < 11000); Pf, ff = P[sel], f[sel]
        SK = (Pf ** 2).mean(1) / (Pf.mean(1) ** 2) - 2
        Et = Pf.sum(0); top = np.argsort(Et)[-3:]
        for b, (lo, hi) in BANDS.items():
            m = (ff >= lo) & (ff < hi); Eb = Pf[m].sum(0)
            d[f"SK{N}_{b}_max"] = SK[m].max(); d[f"SK{N}_{b}_mean"] = SK[m].mean()
            d[f"TK{N}_{b}"] = kurt(Eb); d[f"TC{N}_{b}"] = Eb.max() / np.median(Eb)
            if b != "1-11k":
                d[f"TS{N}_{b}"] = Eb[top].sum() / Et[top].sum()
        L = np.log(Pf + 1e-20); flux = np.clip(np.diff(L, axis=1), 0, None).sum(0)
        d[f"FLX{N}_crest"] = flux.max() / np.median(flux)
        Q = Pf / Pf.sum(0, keepdims=True); ent = -(Q * np.log(Q + 1e-20)).sum(0) / np.log(len(ff))
        d[f"ENT{N}_min"] = ent.min(); d[f"ENT{N}_top"] = ent[top].mean()
        cen = (ff[:, None] * Q).sum(0)
        d[f"CEN{N}_top"] = cen[top].mean(); d[f"CEN{N}_std"] = cen.std()
        d[f"ACT{N}"] = float((Et > 10 * np.median(Et)).mean())
    return d

out = os.path.join(CACHE, "v11_feats.csv")
if os.path.exists(out):
    Fall = pd.read_csv(out, index_col=0)
else:
    Fall = pd.DataFrame([feats(x) for x in np.vstack([Xtr, Xva])], index=ntr + nva)
    Fall.to_csv(out)
Fall.columns = ["new:" + c for c in Fall.columns]

# %% 기존 캐시 피처 (train · val 모두 있는 것)
def pair(tr, va, tag, idx=None):
    a = pd.read_csv(os.path.join(CACHE, tr)); b = pd.read_csv(os.path.join(CACHE, va))
    a.index, b.index = ntr, nva
    c = pd.concat([a, b]).select_dtypes("number"); c.columns = [f"{tag}:{k}" for k in c.columns]; return c
olds = [pair("feat_ext.csv", "feat_ext_val.csv", "ext"), pair("feat2_train.csv", "feat2_val.csv", "H"),
        pair("feat3_train.csv", "feat3_val.csv", "f3"), pair("feat_modeband_train.csv", "feat_modeband_val.csv", "MB"),
        pair("feat_ringmode_train.csv", "feat_ringmode_val.csv", "RM"), pair("feat2split_train.csv", "feat2split_val.csv", "Hs")]
for p, tag in [("3_단계/4_피처/4-6_후보/4-19_최대충격배수/자료/figs/files.csv", "4-19"), ("3_단계/4_피처/4-6_후보/4-21_new5피처/자료/figs/files.csv", "4-21")]:
    c = pd.read_csv(os.path.join(ROOT, p), index_col=0).loc[ntr + nva]
    c = c.drop(columns=[k for k in c.columns if k in ("OOF여유", "판정")]).select_dtypes("number"); c.columns = [f"{tag}:{k}" for k in c.columns]; olds.append(c)
ws = pd.DataFrame({"V3:w_segk_mean": np.r_[np.load(os.path.join(CACHE, "wsegk_train.npy")).ravel()[:150], np.load(os.path.join(CACHE, "wsegk_val.npy")).ravel()[:40]]}, index=ntr + nva)
olds.append(ws)
F = pd.concat([Fall] + olds, axis=1)
F = F.loc[:, F.std() > 0]

# %% 채점
nm, ab = np.array(y) == 0, np.array(y) == 1
Ftr, Fva = F.loc[ntr].values, F.loc[nva].values
U, Fl = pred == 0, pred == 1
uname = np.array(nva)[U]
P0 = 2.5 / 121                                            # 정상 하나가 선을 넘을 확률 (보수적으로)
rows = []
for j, col in enumerate(F.columns):
    xn = Ftr[nm, j]; med = np.median(xn); mad = 1.4826 * np.median(np.abs(xn - med)) or xn.std() + 1e-12
    zsh = (np.median(Fva[U, j]) - med) / mad
    for dr, sgn in [("위", 1), ("아래", -1)]:
        sn, sa, sv = sgn * xn, sgn * Ftr[ab, j], sgn * Fva[:, j]
        thr = np.sort(sn)[-3]
        nN, nA = int((sn > thr).sum()), int((sa > thr).sum())
        hitU = sv[U] > thr; vU, vF = int(hitU.sum()), int((sv[Fl] > thr).sum())
        rows.append(dict(피처=col, 방향=dr, 선=sgn * thr, 정상초과=nN, 고장초과=nA, val판정고장=vF, val미판정=vU,
                         p=stats.binom.sf(vU - 1, 30, P0), 이동z=sgn * zsh, 파일=" ".join(n[-3:] for n in uname[hitU])))
R = pd.DataFrame(rows)
R["판정"] = np.where(R.val미판정 > 12, "이동(초과 과다)", np.where(R.이동z.abs() >= 1, "이동(중앙값)", np.where((R.val미판정 >= 3) & (R.p < 0.01), "후보", "")))
R.sort_values(["판정", "p"], ascending=[True, True]).to_csv(os.path.join(HERE, "피처표.csv"), index=False, encoding="utf-8-sig")
C = R[R.판정 == "후보"].sort_values("p")

# %% 파일별: 몇 개 후보 피처가 끌어올리나 (계열별)
fam = lambda c: c.split(":")[0] + ":" + c.split(":")[1].split("_")[0].rstrip("0123456789") if c.startswith("new:") else c.split(":")[0]
C = C.assign(계열=C.피처.map(fam))
cnt = pd.DataFrame(0, index=[n[-3:] for n in uname], columns=sorted(C.계열.unique()))
for _, r in C.iterrows():
    for f in r.파일.split():
        cnt.loc[f, r.계열] += 1
cnt["합"] = cnt.sum(1); cnt = cnt.sort_values("합", ascending=False)
cnt.to_csv(os.path.join(HERE, "파일별_후보수.csv"), encoding="utf-8-sig")

# 우연 기준: 열 순서를 바꿔도 같은 수가 나오나 → val U 30개를 train 정상 30개(무작위)로 바꿔 같은 절차
rng = np.random.default_rng(0); null_c = []
for _ in range(20):
    pick = rng.choice(np.where(nm)[0], 30, replace=False); rest = np.setdiff1d(np.where(nm)[0], pick)
    c = 0
    for j in range(F.shape[1]):
        xn = Ftr[rest, j]
        for sgn in (1, -1):
            thr = np.sort(sgn * xn)[-3] if len(xn) >= 3 else np.inf
            # 정상 90개 기준 2개 → 120개 기준과 비율을 맞춰 3번째 대신 분위수 사용
            thr = np.quantile(sgn * xn, 1 - 2 / 120)
            v = int((sgn * Ftr[pick, j] > thr).sum())
            c += (v >= 3) and (stats.binom.sf(v - 1, 30, P0) < 0.01)
    null_c.append(c)

# %% 그림 1: 파일별 후보 수
fig, ax = plt.subplots(figsize=(10, 4.2))
top = cnt[cnt.합 > 0]
fams = [c for c in top.columns if c != "합"]
bottom = np.zeros(len(top))
cmap = plt.get_cmap("tab20")
for i, fm in enumerate(fams):
    ax.bar(top.index, top[fm], bottom=bottom, color=cmap(i % 20), label=fm, edgecolor="white", linewidth=0.6); bottom += top[fm].values
ax.set_ylabel("val 미판정 파일을 '정상 2개 선' 위로 올린 후보 피처 수"); ax.set_xlabel("val 파일 (지금 정상 판정)")
ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"후보 피처 {len(C)}개 (우연 기준: train 정상 30개로 바꾸면 평균 {np.mean(null_c):.1f}개)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_파일별_후보수.png"), dpi=130); plt.close(fig)

# %% 그림 2: 계열별 대표 후보 분포 (p가 가장 작은 것 하나씩, 최대 8개)
rep = C.drop_duplicates("계열").head(8)
if len(rep):
    fig, axs = plt.subplots(1, len(rep), figsize=(2.3 * len(rep), 4.2), squeeze=False)
    for ax, (_, r) in zip(axs[0], rep.iterrows()):
        j = list(F.columns).index(r.피처)
        grp = [(Ftr[nm, j], "train\n정상", "#1f77b4"), (Ftr[ab, j], "train\n고장", "#ff7f0e"), (Fva[Fl, j], "val\n판정고장", "#d62728"), (Fva[U, j], "val\n미판정", "#7f7f7f")]
        for k, (v, lab, col) in enumerate(grp):
            ax.scatter(np.full(len(v), k) + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=col, alpha=0.75, linewidths=0)
        ax.axhline(r.선, color="k", ls="--", lw=0.9)
        for n in r.파일.split():
            ax.annotate(n, (3.22, Fva[nva.index("val_" + n), j]), fontsize=6.5, va="center")
        ax.set_xticks(range(4), [g[1] for g in grp], fontsize=7); ax.set_title(f"{r.피처.split(':',1)[1]}\n({r.방향}, U {r.val미판정} · 고장 {r.고장초과}/30)", fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        try:
            if (Ftr[:, j] > 0).all() and Ftr[:, j].max() / max(Ftr[:, j].min(), 1e-12) > 50: ax.set_yscale("log")
        except Exception: pass
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_대표후보_분포.png"), dpi=130); plt.close(fig)

# %% 저장
res = dict(n_features=int(F.shape[1]), n_tests=int(len(R)), n_candidates=int(len(C)), null_candidates=null_c,
           shifted=int((R.판정.str.startswith("이동")).sum()),
           by_family=C.계열.value_counts().to_dict(), per_file=cnt.합.to_dict(),
           top=C.head(40)[["피처", "방향", "정상초과", "고장초과", "val판정고장", "val미판정", "p", "이동z", "파일"]].to_dict("records"))
json.dump(res, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
with open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8") as fh:
    fh.write(f"피처 {F.shape[1]}개 × 2방향 = {len(R)} 검정, 후보 {len(C)}, 우연 기준 {null_c}\n이동 판정 {res['shifted']}\n\n")
    fh.write(cnt.to_string() + "\n\n")
    fh.write(C.head(80)[["피처", "방향", "정상초과", "고장초과", "val판정고장", "val미판정", "p", "이동z", "파일"]].to_string())
print("done")
