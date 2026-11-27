# %% [markdown]
# V-6 효섭 방식으로 충격 추정 (11-27)
# - 질문: 효섭 RF는 val에서 고립 강타형(012 · 032 · 034 · 036)을 고장으로 냈다. 효섭의 충격 추정(AR(40) 백색화 → MAD 8배 봉우리 → 10 ms 간격)은
#   우리 검출기(1–11 kHz 포락선 ÷ 바탕 ≥ 4.5)와 무엇이 다르고, 효섭의 충격 피처는 train 정상 기준으로 이 파일들을 어디에 두는가
# - 효섭 계산은 src/rf_hyoseop.py(원본 복사, 고치지 않음). 피처 표는 EXP-048 캐시 feat_hyoseop_{train,val}.pkl 재사용
# - val 그룹(효섭만 고장 등)은 제출 판정끼리의 비교일 뿐 정답이 아니다. 진단용, 모델 · 문턱 튜닝에 쓰지 않는다
#   conda run -n qe-phm python 5_VAL_단계/V-3_고립강타_구간첨도/3-1_충격검출기비교/자료/run.py

# %% 설정
import os, sys, json, pickle
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
import src.viz  # noqa: F401 (한글 글꼴)
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import roc_auc_score
from src import rf_hyoseop as HY
from src.data import load_dataset, load_folder

FS = 25000; TR, P = HY.TR, HY.P
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
CACHE = os.path.join(ROOT, "4_outputs", "cache")
OUT = {}

Xtr, y, ntr = load_dataset(os.path.join(ROOT, "2_data", "TeamProject_train_변경"))
Xva, nva = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
nm = y == 0

# %% 효섭 충격 검출 (rf_hyoseop.features의 검출 부분과 같은 계산 + 위치 반환)
def hy_impacts(raw):
    """AR(40) 백색화 신호 |w| > 8σ(σ = MAD/0.6745), 10 ms 간격 → 원신호 표본 번호 · 배수(σ 단위) · 울림 주파수(큰 20개)"""
    x = raw - raw.mean()
    xh = signal.sosfiltfilt(HY.SOS["hp10"], x)
    w = HY.ar_whiten(xh)[TR:-TR]
    sig = np.median(np.abs(w)) / 0.6745
    pk, pr = signal.find_peaks(np.abs(w), height=8 * sig, distance=250)
    amp = pr["peak_heights"] / sig
    xs = xh[TR:-TR]; fr = np.fft.rfftfreq(1024, 1 / FS); m = (fr >= 1000) & (fr < 12000)
    fp = np.full(len(pk), np.nan)
    for j in np.argsort(amp)[::-1][:20]:                     # 효섭과 같이 w 번호를 xs에 그대로 씀
        seg = xs[pk[j]:pk[j] + 300]
        if len(seg) == 300:
            Sp = np.abs(np.fft.rfft(seg * np.hanning(300), 1024)) ** 2
            fp[j] = fr[m][Sp[m].argmax()]
    return dict(idx=pk + P + TR, amp=amp, fring=fp, w=w, sig=sig)

hc = os.path.join(CACHE, "val_eda_hyimps.joblib")
if os.path.exists(hc):
    Htr, Hva = joblib.load(hc)
else:
    Htr = [hy_impacts(x) for x in Xtr]; Hva = [hy_impacts(x) for x in Xva]
    for H in Htr + Hva: H.pop("w")
    joblib.dump((Htr, Hva), hc)
Itr, Iva = joblib.load(os.path.join(CACHE, "val_eda_imps.joblib"))      # 우리 검출기(1–11 kHz, V-2 캐시)

# 효섭 피처 표(EXP-048 캐시). 순서 확인: D_nimp == 우리가 다시 센 충격 수
XH = pickle.load(open(os.path.join(CACHE, "feat_hyoseop_train.pkl"), "rb"))[0]
XHv = pickle.load(open(os.path.join(CACHE, "feat_hyoseop_val.pkl"), "rb"))[0]
cnt_tr = np.round(np.exp(XH["D_nimp"].values)).astype(int); cnt_va = np.round(np.exp(XHv["D_nimp"].values)).astype(int)
ok_tr = (cnt_tr == [len(h["idx"]) for h in Htr]).mean(); ok_va = (cnt_va == [len(h["idx"]) for h in Hva]).mean()
OUT["점검_충격수_일치"] = dict(train=float(ok_tr), val=float(ok_va))
print("충격 수 일치", ok_tr, ok_va)

# %% 파일 요약 (두 검출기)
def match(a, b, tol=int(0.003 * FS)):
    """a의 충격 중 b의 충격과 ±3 ms 안에 있는 비율"""
    if len(a) == 0: return np.nan
    if len(b) == 0: return 0.0
    return float(np.mean(np.min(np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :]), 1) <= tol))

def summ(H, I):
    a = np.sort(H["amp"])[::-1]; hx = np.asarray(I["height_x"])
    f = H["fring"][np.argmax(H["amp"])] if len(a) else np.nan
    return dict(hy_n=len(a), hy_max=a[0] if len(a) else 0.0, hy_2nd=a[1] if len(a) > 1 else 0.0, hy_f_top=f,
                hy_top5=a[:5].mean() / a[0] if len(a) else np.nan,
                our_n=len(hx), our_max=hx.max() if len(hx) else 0.0,
                hy_in_our=match(H["idx"], I["idx"]), our_in_hy=match(I["idx"], H["idx"]))

Str = pd.DataFrame([summ(h, i) for h, i in zip(Htr, Itr)]); Str.insert(0, "name", ntr); Str["y"] = y
Sva = pd.DataFrame([summ(h, i) for h, i in zip(Hva, Iva)]); Sva.insert(0, "name", nva)

# val 그룹: 효섭 제출 vs 우리 제출 (판정끼리 비교, 정답 아님)
hy_sub = pd.read_csv(os.path.join(ROOT, "1_참고자료", "팀원 공유 정보", "신효섭", "1119_최종", "6_최종·제출", "제출", "submission_val_9조_상세.csv"))
our_sub = pd.read_csv(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "출력", "submission_val_9조.csv"))
hy1 = dict(zip(hy_sub.file, hy_sub.label)); hyp = dict(zip(hy_sub.file, hy_sub.prob)); us1 = dict(zip(our_sub.iloc[:, 0].str.replace(".csv", "", regex=False), our_sub.iloc[:, 1]))
def grp(n):
    a, b = hy1[n], us1[n]
    return {(1, 1): "둘 다 고장", (1, 0): "효섭만 고장", (0, 1): "우리만 고장", (0, 0): "둘 다 정상"}[(a, b)]
Sva["group"] = [grp(n) for n in nva]; Sva["hy_prob"] = [hyp[n] for n in nva]; Sva["rule83_hy"] = hy_sub.rule83.values
ISO = ["val_012", "val_032", "val_034", "val_036"]                    # V-2 고립 강타형

# %% 효섭 충격 관련 피처: train AUC · 정상 99% 분위 · val 파일의 정상 백분위
FEATS = ["D_nimp", "D_kurt", "D_crest", "D_segk", "A_segk", "b25_5_segk", "b25_5_kurt", "A_kurt", "A_crest",
         "I_density", "I_top5_ratio", "I_count_per_max", "SK_200_3k", "R_frac_2_5k", "R_ring_25_58"]
rows = []
def pct(v, ref): return float((ref < v).mean() * 100 + (ref == v).mean() * 50)
PT = {}
for c in FEATS:
    a, b = XH[c].values.astype(float), XHv[c].values.astype(float)
    ok = ~np.isnan(a); ref = a[nm & ok]
    auc = roc_auc_score(y[ok], a[ok]); d = 1 if auc >= .5 else -1                    # 고장 쪽이 커지게 방향 맞춤
    q99 = np.quantile(d * ref, .99)
    PT[c] = [pct(d * v, d * ref) if not np.isnan(v) else np.nan for v in b]
    rows.append(dict(feat=c, auc=round(max(auc, 1 - auc), 3), 방향="+" if d > 0 else "−",
                     정상99위_train고장=f"{int((d * a[(y == 1) & ok] > q99).sum())}/{int(((y == 1) & ok).sum())}",
                     정상99위_val=int(np.nansum(d * b > q99)),
                     정상99위_고립강타=",".join(n[-3:] for n in ISO if d * b[nva.index(n)] > q99) or "-"))
FT = pd.DataFrame(rows).sort_values("auc", ascending=False)
OUT["피처표"] = FT.to_dict("records")
print(FT.to_string())

# 고립 강타: 두 검출기의 최대 충격 배수가 train 정상 몇 백분위인가
for col in ["hy_max", "our_max", "hy_n", "our_n"]:
    ref = Str.loc[nm, col].values
    Sva[col + "_pct"] = [pct(v, ref) for v in Sva[col]]
OUT["고립강타"] = Sva[Sva.name.isin(ISO)][["name", "hy_n", "hy_max", "hy_max_pct", "hy_f_top", "our_n", "our_max", "our_max_pct", "hy_prob"]].round(3).to_dict("records")
# "충격 ≤ 4개 · 최대 배수 > train 정상 99%"인 파일 수 (기준은 train 정상만으로)
for det, ncol, mcol in [("효섭", "hy_n", "hy_max"), ("우리", "our_n", "our_max")]:
    q = np.quantile(Str.loc[nm, mcol], .99)
    f = lambda S: (S[ncol] <= 4) & (S[mcol] > q)  # noqa: E731
    OUT[f"고립강타_정의_{det}"] = dict(q99=round(float(q), 1), train정상=int(f(Str)[nm].sum()), train고장=int(f(Str)[~nm].sum()),
                                  val=list(Sva.name[f(Sva)].str[-3:]))

# 두 검출기 겹침 (그룹별 중앙값)
ov = pd.concat([Str.assign(group=np.where(nm, "train 정상", "train 고장")), Sva])
OUT["검출기_겹침"] = ov.groupby("group")[["hy_n", "our_n", "hy_in_our", "our_in_hy"]].median().round(2).reset_index().to_dict("records")
print(pd.DataFrame(OUT["검출기_겹침"]).to_string())

# %% 그림 1: 충격 지도 (가로 = 충격 수, 세로 = 가장 큰 충격 배수) — 효섭 · 우리
GC = {"둘 다 고장": "C3", "효섭만 고장": "C4", "우리만 고장": "C1", "둘 다 정상": "0.55"}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for a, (det, ncol, mcol, lab) in zip(axs, [("효섭: AR(40) 백색화 · 8σ", "hy_n", "hy_max", "최대 충격 (백색화 σ 배수)"),
                                         ("우리: 1–11 kHz 포락선 ÷ 바탕", "our_n", "our_max", "최대 충격 (바탕 배수)")]):
    a.scatter(Str[ncol][nm] + 1, Str[mcol][nm], s=14, c="C0", alpha=.45, label="train 정상")
    a.scatter(Str[ncol][~nm] + 1, Str[mcol][~nm], s=22, c="C1", marker="s", alpha=.8, label="train 고장")
    for g, c in GC.items():
        s = Sva[Sva.group == g]
        a.scatter(s[ncol] + 1, s[mcol], s=60, facecolors="none", edgecolors=c, lw=1.6, label=f"val {g}")
    for n in ISO:
        r = Sva[Sva.name == n].iloc[0]; a.annotate(n[-3:], (r[ncol] + 1, r[mcol]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    a.axhline(np.quantile(Str.loc[nm, mcol], .99), color="C0", ls=":", lw=1)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("충격 수 + 1"); a.set_ylabel(lab); a.set_title(det)
axs[0].legend(fontsize=7, loc="upper left")
fig.suptitle("충격 지도 — 점선 = train 정상 99% (val 색 = 제출 판정끼리 비교, 정답 아님)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_충격지도.png"), dpi=130); plt.close(fig)

# %% 그림 2: 파형 + 두 검출기 표시
SHOW = [("val_039", "val 둘 다 고장"), ("val_016", "val 둘 다 고장"), ("val_034", "val 효섭만"), ("val_012", "val 효섭만"),
        ("val_032", "val 효섭만"), ("val_036", "val 효섭만"), ("val_001", "val 효섭만(8.3k)"), ("val_013", "val 우리만"),
        (ntr[int(Str.hy_max[nm].idxmax())], "train 정상 효섭 최대"), ("abn_017", "train 고장")]
SHOW = [(n, g) for n, g in SHOW if n in nva or n in ntr]
t = np.arange(50000) / FS
fig, axs = plt.subplots(len(SHOW), 1, figsize=(13, 1.15 * len(SHOW)), sharex=True)
for a, (n, g) in zip(axs, SHOW):
    if n in nva: i = nva.index(n); x, H, I, S = Xva[i], Hva[i], Iva[i], Sva.iloc[i]
    else: i = ntr.index(n); x, H, I, S = Xtr[i], Htr[i], Itr[i], Str.iloc[i]
    xh = signal.sosfiltfilt(HY.SOS["hp200"], x - x.mean())
    a.plot(t, xh, lw=.3, color="0.3"); top = np.abs(xh).max()
    a.scatter(H["idx"] / FS, np.full(len(H["idx"]), top * 1.05), marker="v", s=14, c="C3", label="효섭")
    a.scatter(I["idx"] / FS, np.full(len(I["idx"]), -top * 1.05), marker="^", s=14, c="C0", label="우리")
    a.set_ylabel(n[-7:], rotation=0, ha="right", fontsize=8)
    a.text(1.0, .5, f" {g}\n 효섭 {S.hy_n}개 · 최대 {S.hy_max:.0f}σ\n 우리 {S.our_n}개 · 최대 {S.our_max:.0f}배", transform=a.transAxes, fontsize=7, va="center")
    a.set_yticks([])
axs[0].legend(fontsize=7, loc="upper left", ncol=2); axs[-1].set_xlabel("시간 (s)")
fig.suptitle("파형(200 Hz 고역) + 충격: ▼ 효섭(백색화 8σ) · ▲ 우리(포락선 4.5배)")
fig.tight_layout(rect=(0, 0, .9, 1)); fig.savefig(os.path.join(FIG, "2_파형_두검출기.png"), dpi=120); plt.close(fig)

# %% 그림 3: 효섭 충격 피처의 train 정상 백분위 (val 파일, 고장 방향)
order = list(FT.feat)
V = Sva.copy()
for c in order: V[c] = PT[c]
V = V.sort_values(["group", "hy_prob"], ascending=[True, False])
M = V[order].values
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=0, vmax=100)
ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{c}\nAUC {a}" for c, a in zip(FT.feat, FT.auc)], rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(V))); ax.set_yticklabels([f"{n[-3:]} {g} p={p:.2f}" for n, g, p in zip(V.name, V.group, V.hy_prob)], fontsize=7)
for (r, c), v in np.ndenumerate(M):
    if not np.isnan(v) and v >= 99: ax.text(c, r, "●", ha="center", va="center", fontsize=6, color="w")
plt.colorbar(im, ax=ax, label="train 정상 안 백분위 (고장 방향)"); ax.set_title("val 파일의 효섭 충격 피처 — ● = 정상 99% 위 (p = 효섭 RF 확률)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_피처백분위_val.png"), dpi=120); plt.close(fig)

# %% 추가: 구간 첨도 3개(segk) — 정상 99% 플래그 · 우리 H 피처와 상관
SEGK = ["D_segk", "A_segk", "b25_5_segk"]
flag_tr = np.zeros(len(y), bool); flag_va = np.zeros(len(nva), bool); per = {}
for c in SEGK:
    q = np.quantile(XH[c].values[nm], .99)
    ft, fv = XH[c].values > q, XHv[c].values > q
    per[c] = dict(train정상=int(ft[nm].sum()), train고장=int(ft[~nm].sum()), val=list(np.array(nva)[fv]))
    flag_tr |= ft; flag_va |= fv
per["셋 중 하나(OR)"] = dict(train정상=int(flag_tr[nm].sum()), train고장=int(flag_tr[~nm].sum()), val=list(np.array(nva)[flag_va]))
OUT["segk_정상99"] = per
print(per)
m062 = joblib.load(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "모델", "final_model_062.joblib")); core = m062.core_
Ttr, Tva, _, _, _ = joblib.load(os.path.join(CACHE, "val_eda_tabs.joblib"))
Etr, _ = core._et_input(Ttr); hcols = list(core.cols) + list(core.peak)
from scipy.stats import spearmanr
cor = {}
for c in SEGK + ["D_kurt", "A_kurt"]:
    r = [spearmanr(XH[c].values, Etr[:, j])[0] for j in range(Etr.shape[1])]
    j = int(np.nanargmax(np.abs(r))); cor[c] = dict(가장비슷한_H=hcols[j], rho=round(float(r[j]), 2))
OUT["segk_우리H_상관"] = cor
print(cor)
# 효섭 검출기의 기여: 백색화 전후 같은 충격의 크기 비교 — 고립 강타 파일 vs train 정상 고립 강타(046 · 060)
def zoom_rows(n):
    if n in nva: i = nva.index(n); x, H, I = Xva[i], Hva[i], Iva[i]
    else: i = ntr.index(n); x, H, I = Xtr[i], Htr[i], Itr[i]
    return x, H, I

# %% 그림 4: 효섭 검출기에만 잡힌 충격(우리 검출 ±3 ms 밖) 확대 — 012 · 032 · nor_046
ZF = ["val_012", "val_032", "nor_046"]
fig, axs = plt.subplots(len(ZF), 4, figsize=(13, 2.3 * len(ZF)))
for r, n in enumerate(ZF):
    x, H, I = zoom_rows(n)
    xh = signal.sosfiltfilt(HY.SOS["hp200"], x - x.mean())
    far = np.array([np.min(np.abs(np.asarray(I["idx"]) - k)) > 75 if len(I["idx"]) else True for k in H["idx"]])
    js = np.where(far)[0][np.argsort(H["amp"][far])[::-1]][:3]
    jb = [int(np.argmax(H["amp"]))]
    for c, j in enumerate(jb + list(js)):
        a = axs[r, c]; k = H["idx"][j]; s = slice(max(k - 50, 0), k + 300)
        a.plot((np.arange(s.start, s.stop) - k) / FS * 1000, xh[s], lw=.6, color="C3" if c else "k")
        fr = H["fring"][j]
        a.set_title(f"{n[-7:]} {'가장 큰 충격' if c == 0 else '효섭만'} {H['amp'][j]:.0f}σ" + (f" · {fr / 1000:.1f} kHz" if not np.isnan(fr) else ""), fontsize=8)
        a.tick_params(labelsize=7)
    for c in range(len(jb) + len(js), 4): axs[r, c].axis("off")
for a in axs[-1]: a.set_xlabel("ms (충격 기준)")
fig.suptitle("효섭 검출기에만 잡힌 작은 충격 (200 Hz 고역, 세로 눈금은 칸마다 다름)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_효섭만_충격확대.png"), dpi=120); plt.close(fig)
# 효섭만 잡은 충격의 울림 주파수 분포(큰 20개 안에서) — 파일 그룹별
def only_hy_f(H, I):
    out = []
    for j, k in enumerate(H["idx"]):
        if not np.isnan(H["fring"][j]) and (len(I["idx"]) == 0 or np.min(np.abs(np.asarray(I["idx"]) - k)) > 75):
            out.append(H["fring"][j])
    return np.array(out)
fo = {}
for lab, idxs, HH, II in [("train 정상", np.where(nm)[0], Htr, Itr), ("train 고장", np.where(~nm)[0], Htr, Itr)]:
    f = np.concatenate([only_hy_f(HH[i], II[i]) for i in idxs]); fo[lab] = dict(n=len(f), frac_2_5k=round(float(((f >= 2000) & (f < 5000)).mean()), 2))
for g in Sva.group.unique():
    f = np.concatenate([only_hy_f(Hva[i], Iva[i]) for i in np.where(Sva.group == g)[0]]); fo["val " + g] = dict(n=len(f), frac_2_5k=round(float(((f >= 2000) & (f < 5000)).mean()), 2) if len(f) else None)
for n in ["val_012", "val_032", "val_034", "nor_046"]:
    HH, II = (Hva[nva.index(n)], Iva[nva.index(n)]) if n in nva else (Htr[ntr.index(n)], Itr[ntr.index(n)])
    f = only_hy_f(HH, II); fo[n] = dict(n=len(f), frac_2_5k=round(float(((f >= 2000) & (f < 5000)).mean()), 2) if len(f) else None)
OUT["효섭만_충격_울림"] = fo
print(fo)

# %% 저장
cols = ["name", "group", "hy_prob", "rule83_hy", "hy_n", "hy_max", "hy_max_pct", "hy_f_top", "our_n", "our_max", "our_max_pct", "hy_in_our", "our_in_hy"]
Sva[cols].round(3).to_csv(os.path.join(HERE, "파일표_val.csv"), index=False, encoding="utf-8-sig")
Str.round(3).to_csv(os.path.join(HERE, "파일표_train.csv"), index=False, encoding="utf-8-sig")
json.dump(OUT, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("끝")
