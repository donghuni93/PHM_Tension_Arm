# %% [markdown]
# 3-2. 고주파 블록의 고역통과 차단 주파수(fc) 정하기
# 선택은 라벨 없는 기준(A)으로, 라벨을 쓰는 결과(B)는 민감도로만 보고한다. 격자·규칙은 실행 전에 고정.
# 실행: python "단계/3_전처리/3-2_1121_고역차단주파수/run.py"
import os, sys, json, warnings
warnings.filterwarnings("ignore")
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, welch, sosfreqz
from scipy.stats import spearmanr, kurtosis
from sklearn.metrics import roc_auc_score
from src import final
from src.final import FS, _SEG, _ring_atoms
from src.exp import load_train, make_et
from src.data import load_folder
from src.evaluate import quick_cv
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)

FCS = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500]          # 고정 격자
TRACK = ["abn_017", "abn_022", "abn_014", "nor_045", "nor_116", "nor_059", "nor_055"]
SEEDS = range(5)
FEATS = ["RMS_log", "첨도", "파고율", "충격수_log1p", "울림몫_5.5-7.75k", "울림몫_2.125-4.125k", "ζ≥0.04_비율", "RES_8.0-8.7k_ζ≤.02_fit"]

# %% 데이터
X, y, names = load_train()
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val", "val"))
Xa = np.vstack([X, Xv]); ntr = len(y); nall = len(Xa)
assert X.shape == (150, 50000) and Xv.shape == (40, 50000) and (y == 0).sum() == 120
Xc = Xa - Xa.mean(1, keepdims=True)
nor = np.zeros(nall, bool); nor[:ntr] = (y == 0)              # train 정상 마스크
alls = np.ones(nall, bool)                                    # 전체 190

# %% 일반화 충격 검출 (final.impact_rings의 fc 인자 버전. 사전은 그대로 1–12 kHz)
ATOMS, PAR = _ring_atoms()
LP500 = butter(4, 500, btype="lowpass", fs=FS, output="sos")

def hp(Xm, fc):
    """평균 제거된 신호 → fc 고역통과 (4차 Butterworth sosfiltfilt)"""
    return sosfiltfilt(butter(4, fc, btype="highpass", fs=FS, output="sos"), Xm, axis=-1)

def rings_from_hp(X1):
    rows = []
    for i, x in enumerate(X1):
        env = sosfiltfilt(LP500, np.abs(hilbert(x)))
        pk, _ = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS))
        pk = pk[pk + _SEG <= len(x)]
        if not len(pk):
            continue
        S = np.stack([x[p:p + _SEG] for p in pk])
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        fit = np.linalg.norm(np.einsum("ks,ast->kat", S, ATOMS), axis=2)
        b = fit.argmax(1)
        rows += [(i, PAR[bb, 0], PAR[bb, 1], ft) for bb, ft in zip(b, fit.max(1))]
    return np.array(rows, dtype=float).reshape(-1, 4)

# %% 신호 · 필터 · 충격 표 (fc마다)
wob = np.log10(np.sqrt((sosfiltfilt(butter(4, 20, btype="lowpass", fs=FS, output="sos"), Xc, axis=-1) ** 2).mean(1)))
HP, RG = {}, {}
for fc in FCS:
    HP[fc] = hp(Xc, fc)
    RG[fc] = rings_from_hp(HP[fc])
    print("fc", fc, "충격", len(RG[fc]), flush=True)
# fc=1000에서 final.impact_rings와 완전 일치
R_ref = final.impact_rings(Xa)
assert R_ref.shape == RG[1000].shape and np.array_equal(R_ref, RG[1000]), "fc=1000이 final.impact_rings와 다르다"
print("assert OK: fc=1000 == final.impact_rings", R_ref.shape)

# %% A1 길 끌림
rms_log = {fc: np.log10(np.sqrt((HP[fc] ** 2).mean(1))) for fc in FCS}
kurt = {fc: kurtosis(HP[fc], axis=1) for fc in FCS}
A1 = {}
for lab, m in (("train정상120", nor), ("전체190", alls)):
    A1[lab] = {"rho_rms": [float(spearmanr(rms_log[fc][m], wob[m])[0]) for fc in FCS],
               "rho_kurt": [float(spearmanr(kurt[fc][m], wob[m])[0]) for fc in FCS]}
rho = np.array(A1["train정상120"]["rho_rms"])
d_rho = np.abs(np.diff(rho))
sel_idx = next((i for i, d in enumerate(d_rho) if d < 0.05), len(FCS) - 1)
FC_SEL = FCS[sel_idx]
A1_delta = {lab: {k: np.abs(np.diff(v[k])).round(4).tolist() for k in v} for lab, v in A1.items()}
print("보충 |Δρ|", A1_delta)
print("rho_rms(train정상)", rho.round(3), "|Δρ|", d_rho.round(3), "→ 선택 fc", FC_SEL)

# %% A2 잘리는 정상 모드
fw, Pw = welch(Xc[nor], fs=FS, nperseg=8192, window="hann", noverlap=4096, detrend=False, axis=-1)
Pmed = np.median(Pw, 0)
band = (fw >= 500) & (fw <= 3000)
lp = np.log10(Pmed)
pk, pr = find_peaks(lp[band], prominence=0.2)
pk_f = fw[band][pk]
pk_prom = pr["prominences"]
print("0.5–3 kHz 봉우리(prom>=0.2 dec):", [(round(float(f)), round(float(p), 2)) for f, p in zip(pk_f, pk_prom)])
def gain(fc, f):
    sos = butter(4, fc, btype="highpass", fs=FS, output="sos")
    _, h = sosfreqz(sos, worN=np.atleast_1d(f), fs=FS)
    return np.abs(h) ** 2                                   # filtfilt = 정방향 × 역방향 → |H|²
A2 = {"peaks_Hz": [float(f) for f in pk_f], "prominence_dec": [float(p) for p in pk_prom],
      "pass": {str(fc): [float(g) for g in gain(fc, pk_f)] for fc in FCS},
      "pass_known": {str(fc): dict(zip(["730", "1310", "1600", "1880"], [float(g) for g in gain(fc, [730, 1310, 1600, 1880])])) for fc in FCS}}

# %% A3 충격 검출 안정성
def per_file(R, n):
    fi = R[:, 0].astype(int)
    nimp = np.bincount(fi, minlength=n).astype(float)
    return fi, nimp
A3 = {}
for fc in FCS:
    R = RG[fc]; fi, nimp = per_file(R, nall)
    mn = nor[fi]; mv = (fi >= ntr)
    lo2 = R[:, 1] < 2000
    A3[fc] = {"n_imp_train_normal": {"median": float(np.median(nimp[nor])), "q25": float(np.percentile(nimp[nor], 25)),
                                      "q75": float(np.percentile(nimp[nor], 75)), "zero_files": int((nimp[nor] == 0).sum()), "mean": float(nimp[nor].mean())},
              "n_imp_val": {"median": float(np.median(nimp[ntr:])), "q25": float(np.percentile(nimp[ntr:], 25)),
                            "q75": float(np.percentile(nimp[ntr:], 75)), "zero_files": int((nimp[ntr:] == 0).sum())},
              "frac_f_lt2k_train_normal": float(lo2[mn].mean()) if mn.any() else None,
              "frac_f_lt2k_val": float(lo2[mv].mean()) if mv.any() else None,
              "fit_median_train_normal": float(np.median(R[mn, 3])), "fit_median_val": float(np.median(R[mv, 3])),
              "n_impacts_train_normal": int(mn.sum())}
warn = []
for j in (sel_idx - 1, sel_idx + 1):
    if 0 <= j < len(FCS):
        a, b = A3[FC_SEL]["n_imp_train_normal"]["median"], A3[FCS[j]]["n_imp_train_normal"]["median"]
        lo, hi = min(a, b), max(a, b)
        if hi > 0 and (lo == 0 or hi / lo >= 2):
            warn.append(f"fc {FC_SEL} 정상 충격 수 중앙값 {a:g} vs 이웃 fc {FCS[j]} {b:g}: 2배 이상 차이")
print("A3 경고:", warn or "없음")

# %% B 피처 8개 (fc마다)
def feat_table(fc):
    R = RG[fc]; fi, nimp = per_file(R, nall); f, z, ft = R[:, 1], R[:, 2], R[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = lambda m: np.where(nimp > 0, np.bincount(fi, weights=m.astype(float), minlength=nall) / nimp, -1.0)
        body = frac((f >= 5500) & (f < 7750)); r24 = frac((f >= 2125) & (f < 4125)); zhi = frac(z >= 0.04)
    res = np.zeros(nall); m = (f >= 8000) & (f < 8700) & (z <= 0.02); np.maximum.at(res, fi[m], ft[m])
    crest = np.abs(HP[fc]).max(1) / np.sqrt((HP[fc] ** 2).mean(1))
    return np.column_stack([rms_log[fc], kurt[fc], crest, np.log1p(nimp), body, r24, zhi, res])
FT = {fc: feat_table(fc) for fc in FCS}
# 검증: fc=1000의 울림 몫·RES·충격수가 final.ring_table과 같다
rg_ref, res_ref, nimp_ref = final.ring_table(Xa)
assert np.allclose(FT[1000][:, 3], np.log1p(nimp_ref)) and np.allclose(FT[1000][:, 4], rg_ref[:, 1]) and np.allclose(FT[1000][:, 5], rg_ref[:, 0]) and np.allclose(FT[1000][:, 7], res_ref)
print("assert OK: fc=1000 피처 == final.ring_table")

# %% B1 단변량 AUC (train 150, 고장=1)
B1 = {str(fc): [float(roc_auc_score(y, FT[fc][:ntr, k])) for k in range(8)] for fc in FCS}

# %% B2 quick_cv (ExtraTrees make_et, 5-fold × seed 5)  /  OOF 저장 → B3
groups = np.arange(ntr)
B2, OOF = {}, {}
for fc in FCS:
    F = FT[fc][:ntr]
    pr, oofs = [], []
    for s in SEEDS:
        q = quick_cv(F, y, groups, make_et(seed=s), seeds=[s])
        pr.append(float(q["pr_auc"][0])); oofs.append(q["oof"][0])
    B2[str(fc)] = {"pr_auc_mean": float(np.mean(pr)), "pr_auc_sd": float(np.std(pr, ddof=1)), "pr_auc_seeds": pr}
    OOF[fc] = np.mean(oofs, 0)
    print("fc", fc, "PR-AUC", round(B2[str(fc)]["pr_auc_mean"], 3), "±", round(B2[str(fc)]["pr_auc_sd"], 3), flush=True)

# %% B3 파일 추적
def idx_of(nm): return names.index(nm)
B3 = {}
for nm in TRACK:
    i = idx_of(nm); B3[nm] = {}
    for fc in FCS:
        R = RG[fc]; r = R[R[:, 0] == i]
        fl = np.sort(r[:, 1]).astype(int).tolist()
        rank = int((OOF[fc] > OOF[fc][i]).sum() + 1)        # 150개 중 점수 높은 순 순위 (1 = 가장 고장스러움)
        B3[nm][str(fc)] = {"n_imp": int(len(r)), "freqs_kHz": [round(f / 1000, 2) for f in fl], "oof": float(OOF[fc][i]), "rank_of_150": rank,
                           "rank_among_normals": int((OOF[fc][:ntr][y == 0] > OOF[fc][i]).sum() + 1) if y[i] == 1 else None}
# 고장 30개의 OOF 순위 평균 (참고)
miss_rank = {str(fc): float(np.mean([(OOF[fc][:ntr][y == 0] >= OOF[fc][i]).mean() for i in np.flatnonzero(y == 1)])) for fc in FCS}

# %% 그림
BL, OR, GR, PU = "#2a78d6", "#eb6834", "#9aa0a6", "#7B1FA2"
x = np.arange(len(FCS)); lab = [str(f) for f in FCS]
# ① ρ vs fc
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, key, tt in ((ax[0], "rho_rms", "고역 log RMS ↔ 출렁임(20 Hz 저역 RMS)"), (ax[1], "rho_kurt", "고역 첨도 ↔ 출렁임")):
    a.plot(x, A1["train정상120"][key], "o-", color=BL, label="train 정상 120")
    a.plot(x, A1["전체190"][key], "s--", color=PU, label="전체 190 (train+val)")
    a.axhline(0, color="k", lw=.5); a.set_xticks(x); a.set_xticklabels(lab); a.set_xlabel("fc (Hz)"); a.set_ylabel("스피어만 ρ"); a.set_title(tt, fontsize=10); a.legend(fontsize=8)
ax[0].axvline(sel_idx, color=OR, ls=":"); ax[0].text(sel_idx, ax[0].get_ylim()[1], f" 선택 {FC_SEL}", color=OR, va="top")
fig.savefig(os.path.join(FIG, "1_길끌림_rho_vs_fc.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
# ② PSD + 필터 곡선
fig, ax = plt.subplots(figsize=(11, 5))
b2 = (fw >= 500) & (fw <= 3000)
ax.plot(fw[b2], lp[b2], color="k", lw=1.4, label="train 정상 중앙 PSD (log10)")
ax.plot(pk_f, lp[b2][pk], "v", color=OR, label="봉우리")
for f in pk_f: ax.text(f, lp[b2][pk][list(pk_f).index(f)] + .05, f"{f:.0f}", ha="center", fontsize=7, color=OR)
ax.set_xlabel("주파수 (Hz)"); ax.set_ylabel("log10 PSD (g²/Hz)")
ax2 = ax.twinx(); ff = np.linspace(500, 3000, 600)
cm = plt.cm.viridis(np.linspace(0, .9, len(FCS)))
for c, fc in zip(cm, FCS): ax2.plot(ff, gain(fc, ff), color=c, lw=1.2, label=f"fc {fc}")
ax2.set_ylabel("진폭 통과율 |H|² (filtfilt)"); ax2.set_ylim(0, 1.05)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc="lower right")
ax.set_title("정상 모드와 고역통과 곡선 (0.5–3 kHz)")
fig.savefig(os.path.join(FIG, "2_PSD_필터통과율.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
# ③ AUC · quick_cv vs fc
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
cm8 = plt.cm.tab10(np.arange(8))
for k, nm in enumerate(FEATS): ax[0].plot(x, [B1[str(fc)][k] for fc in FCS], "o-", color=cm8[k], label=nm, ms=4)
ax[0].axhline(.5, color="k", lw=.5); ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_xlabel("fc (Hz)"); ax[0].set_ylabel("단변량 AUC (고장=1, 방향 안 맞춤)"); ax[0].legend(fontsize=6.5, ncol=2); ax[0].set_title("B1 피처별 AUC", fontsize=10)
m_ = [B2[str(fc)]["pr_auc_mean"] for fc in FCS]; s_ = [B2[str(fc)]["pr_auc_sd"] for fc in FCS]
ax[1].errorbar(x, m_, yerr=s_, fmt="o-", color=BL, capsize=4); ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_xlabel("fc (Hz)"); ax[1].set_ylabel("quick_cv PR-AUC (seed 5개 평균 ± sd)"); ax[1].set_title("B2 ET 8피처", fontsize=10)
ax[1].axvline(sel_idx, color=OR, ls=":")
fig.savefig(os.path.join(FIG, "3_AUC_quickcv_vs_fc.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
# ④ 정상 충격 수 · 울림 주파수 분포
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
cnt = [per_file(RG[fc], nall)[1][nor] for fc in FCS]
ax[0].boxplot(cnt, positions=x, widths=.6, showfliers=True); ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_xlabel("fc (Hz)"); ax[0].set_ylabel("파일당 충격 수 (train 정상 120)"); ax[0].set_title("충격 수 분포")
edges = np.arange(875, 12126, 250); H = []
for fc in FCS:
    R = RG[fc]; fi = R[:, 0].astype(int); m = nor[fi]
    h, _ = np.histogram(R[m, 1], bins=edges); H.append(h / max(h.sum(), 1))
im = ax[1].imshow(np.array(H), aspect="auto", origin="lower", extent=[edges[0] / 1000, edges[-1] / 1000, -.5, len(FCS) - .5], cmap="magma_r")
ax[1].set_yticks(x); ax[1].set_yticklabels(lab); ax[1].set_xlabel("울림 주파수 (kHz)"); ax[1].set_ylabel("fc (Hz)"); ax[1].set_title("정상 충격의 울림 주파수 분포 (fc별 비율)")
fig.colorbar(im, ax=ax[1], label="충격 비율")
fig.savefig(os.path.join(FIG, "4_정상_충격수_울림분포.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% result.json
out = {"grid": FCS, "selected_fc": FC_SEL, "selection_rule": "A1 train정상 RMS ρ의 다음 격자와 |Δρ|<0.05가 처음 성립하는 fc",
       "A1": A1, "A1_abs_delta_rho_all": A1_delta, "A1_abs_delta_rho_rms_train_normal": d_rho.tolist(), "A2": A2,
       "A3": {str(k): v for k, v in A3.items()}, "A3_warnings": warn,
       "B_features": FEATS, "B1_auc": B1, "B2_quick_cv": B2, "B3": B3, "B3_mean_miss_rank_abnormal": miss_rank,
       "assert_fc1000_equals_final": True}
with open(os.path.join(HERE, "result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("완료 선택 fc", FC_SEL)
