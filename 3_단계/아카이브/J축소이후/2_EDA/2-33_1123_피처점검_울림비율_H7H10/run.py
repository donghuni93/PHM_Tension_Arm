# %% 2-33 피처 점검 ② 울림 비율 H7~H10: 충격마다 뒤 8 ms를 감쇠 사인파 사전(1–12 kHz 250 Hz × ζ 4개)에 맞춰 울림 주파수를 고르고, 파일마다 옥타브(1–2 · 2–4 · 4–8 · 8–12.5 kHz)별 충격 비율. 충격 0개 = −1. J-축소는 (k + 3·p0)/(n + 3)로 축소
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp2util import NS
from src.impacts import all_impacts, _atoms, FS
from src.final import _SEG
from src.features2 import octave_of
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")
_av = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((n for n in ["Malgun Gothic", "NanumGothic", "WenQuanYi Zen Hei"] if n in _av), "DejaVu Sans"); plt.rcParams["axes.unicode_minus"] = False
ns = NS(); y = ns.y; names = list(ns.names); nv = [n[-3:] for n in ns.nv]; ntr = len(names)
allnames = names + ["val_" + n for n in nv]; X = np.vstack([ns.X, ns.Xv]); grp = np.array(["정상" if v == 0 else "고장" for v in y] + ["val"] * len(nv))
isN, isA, isV = grp == "정상", grp == "고장", grp == "val"
IM = all_impacts(X); BANDS = ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]
n_imp = np.array([len(m["t"]) for m in IM]); R = np.full((len(X), 4), -1.0); K = np.zeros((len(X), 4))
for i, m in enumerate(IM):
    if n_imp[i]: K[i] = np.bincount(octave_of(m["rings"][:, 0]).astype(int), minlength=4)[:4]; R[i] = K[i] / n_imp[i]
chk = max(np.abs(R[:ntr] - ns.T[BANDS].values).max(), np.abs(R[ntr:] - ns.Tv[BANDS].values).max())
# 축소 (J-축소와 같은 식, p0 = 학습 정상의 모든 충격에서 대역 비율)
m0 = isN[:ntr] & (n_imp[:ntr] > 0); p0 = K[:ntr][m0].sum(0) / n_imp[:ntr][m0].sum()
S = R.copy(); h = n_imp > 0; S[h] = (K[h] + 3.0 * p0) / (n_imp[h, None] + 3.0)

# %% 그림 1: 정의 — 가장 큰 충격 8 ms와 맞춘 감쇠 사인파
atoms, par = _atoms(); ex = [("nor_006", "정상"), ("nor_025", "정상 (오경보)"), ("nor_116", "정상 (오경보)"), ("abn_010", "고장 (8.3k)"), ("abn_022", "고장 (미탐, 2 kHz)"), ("val_004", "val 004 (J: 고장)")]
ex = [(n, t) for n, t in ex if n in allnames]
fig, axs = plt.subplots(len(ex), 2, figsize=(14, 2.2 * len(ex)), gridspec_kw=dict(width_ratios=[1.2, 1]))
for row, (n, t) in zip(axs, ex):
    i = allnames.index(n); m = IM[i]; j = int(np.argmax(m["height_g"])); p = m["idx"][j]; seg = m["hp"][p:p + _SEG]; seg = seg / (np.linalg.norm(seg) + 1e-12)
    fit = np.linalg.norm(np.einsum("s,ast->at", seg, atoms), axis=1); a = int(fit.argmax()); rec = atoms[a] @ (atoms[a].T @ seg); tt = np.arange(_SEG) / FS * 1000
    row[0].plot(tt, seg, color="#3b7dd8", lw=1, label="충격 뒤 8 ms (고역통과, 정규화)"); row[0].plot(tt, rec, color="#e8743b", lw=1.2, label=f"맞춘 감쇠 사인파 {par[a, 0]/1000:.2f} kHz · ζ {par[a, 1]} · fit {fit[a]:.2f}")
    row[0].set_title(f"{n} — {t} · 가장 큰 충격 ({m['t'][j]:.2f}s)", fontsize=9, loc="left"); row[0].legend(fontsize=7, loc="upper right"); row[0].set_xlabel("ms", fontsize=8)
    f_all = m["rings"][:, 0]; row[1].hist(f_all / 1000, bins=np.arange(1, 12.6, .5), color="#9bb7e0"); [row[1].axvline(v, color="k", ls=":", lw=.7) for v in (2, 4, 8)]
    row[1].set_title(f"이 파일 충격 {len(f_all)}개의 울림 주파수 (점선 = 옥타브 경계) · H7~H10 = " + ", ".join(f"{v:.2f}" for v in R[i]), fontsize=8, loc="left"); row[1].set_xlabel("kHz", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_정의_예시.png"), dpi=110); plt.close(fig)

# %% 그림 2: 충격 단위 울림 주파수 분포 · 맞춤도(fit) 분포 (그룹별)
C = {"정상": "#3b7dd8", "고장": "#e8743b", "val": "#2ca02c"}; P = {g: np.vstack([IM[i]["rings"] for i in np.flatnonzero(grp == g) if n_imp[i]]) for g in C}
fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
for g in C:
    axs[0].hist(P[g][:, 0] / 1000, bins=np.arange(1, 12.6, .5), density=True, alpha=.5, color=C[g], label=f"{g} (충격 {len(P[g])}개)"); axs[1].hist(P[g][:, 2], bins=np.linspace(0, 1, 26), density=True, alpha=.5, color=C[g], label=g)
axs[0].set_title("충격 단위 울림 주파수 (kHz)"); [axs[0].axvline(v, color="k", ls=":", lw=.7) for v in (2, 4, 8)]; axs[0].legend(fontsize=8); axs[1].set_title("충격 단위 맞춤도 fit (1 = 사전과 완전 일치)"); axs[1].legend(fontsize=8)
zt = {g: [np.mean(P[g][:, 1] == z) for z in (0.01, 0.02, 0.04, 0.08)] for g in C}; w = .27
for k, g in enumerate(C): axs[2].bar(np.arange(4) + (k - 1) * w, zt[g], w, color=C[g], alpha=.7, label=g)
axs[2].set_xticks(range(4)); axs[2].set_xticklabels(["0.01", "0.02", "0.04", "0.08"]); axs[2].set_title("고른 감쇠비 ζ 비율"); axs[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_충격단위_분포.png"), dpi=110); plt.close(fig)

# %% 그림 3: 파일 단위 H7~H10 (충격 있는 파일만) 분포
fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
for b, ax in enumerate(axs):
    for g in C: m = (grp == g) & (n_imp > 0); ax.hist(R[m, b], bins=np.linspace(0, 1, 11), density=True, alpha=.5, color=C[g], label=g)
    ax.set_title(BANDS[b]); ax.grid(alpha=.2)
axs[0].legend(fontsize=8); fig.suptitle("파일 단위 울림 비율 (충격 ≥ 1개인 파일만). 0과 1에 몰리는 것은 충격이 적어서", x=.01, ha="left", fontsize=9.5); fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_파일단위_분포.png"), dpi=110); plt.close(fig)

# %% 표
auc = lambda v, a, b: roc_auc_score(np.r_[np.zeros(a.sum()), np.ones(b.sum())], np.r_[v[a], v[b]])
def stat(v, mask=None):
    mk = np.ones(len(v), bool) if mask is None else mask; a = auc(v, isN & mk, isA & mk); s = auc(v, isN & mk, isV & mk)
    return dict(분리도=round(abs(a - .5) * 2, 2), 이동도=round(abs(s - .5) * 2, 2), 정상중앙=round(float(np.median(v[isN & mk])), 2), 고장중앙=round(float(np.median(v[isA & mk])), 2), val중앙=round(float(np.median(v[isV & mk])), 2))
rows = []
for b, c in enumerate(BANDS):
    rows.append(dict(피처=c, 처리="원값(−1 포함, 모델 입력 그대로)", **stat(R[:, b]))); rows.append(dict(피처=c, 처리="충격 ≥ 1개 파일만", **stat(R[:, b], n_imp > 0))); rows.append(dict(피처=c, 처리="충격 ≥ 4개 파일만", **stat(R[:, b], n_imp >= 4))); rows.append(dict(피처=c, 처리="축소 후(α=3, J-축소 입력)", **stat(S[:, b])))
T1 = pd.DataFrame(rows)
# n 층별 H9 (4–8 kHz) 중앙값: val 이동이 충격 수 때문인지
strata = [("n=1", n_imp == 1), ("n=2~3", (n_imp >= 2) & (n_imp <= 3)), ("n=4~6", (n_imp >= 4) & (n_imp <= 6)), ("n≥7", n_imp >= 7)]
T2 = pd.DataFrame([dict(층=s, **{f"{g} 파일수": int(((grp == g) & m).sum()) for g in C}, **{f"{g} H9 중앙": (round(float(np.median(R[(grp == g) & m, 2])), 2) if ((grp == g) & m).sum() else None) for g in C}) for s, m in strata])
fitq = pd.DataFrame([dict(그룹=g, 충격수=len(P[g]), fit중앙=round(float(np.median(P[g][:, 2])), 2), fit_0_5미만=round(float((P[g][:, 2] < .5).mean()), 2), ζ0_01=round(zt[g][0], 2), ζ0_08=round(zt[g][3], 2)) for g in C])
B = [("nor_025", "오경보"), ("nor_055", "오경보"), ("nor_059", "오경보"), ("nor_065", "오경보"), ("nor_116", "오경보"), ("abn_022", "미탐"), ("abn_005", "경계"), ("abn_002", "8.3k규칙"), ("abn_029", "8.3k규칙"), ("val_025", "val경계"), ("val_018", "val경계"), ("val_021", "val경계"), ("val_012", "val경계"), ("val_027", "val경계"), ("val_001", "val(8.3k만)")]
pct = lambda v, i: float((v[isN] <= v[i]).mean() * 100)
T3 = pd.DataFrame([dict(파일=n, 구분=t, 충격수=int(n_imp[allnames.index(n)]), **{c[:3]: f"{R[allnames.index(n), b]:.2f} ({pct(R[:, b], allnames.index(n)):.0f})" for b, c in enumerate(BANDS)}) for n, t in B if n in allnames])
D = pd.DataFrame(R[:ntr], columns=[c[:3] for c in BANDS]); D["n"] = n_imp[:ntr]
for c in ["H1_nimp", "H3_kurt_1-2k", "H5_kurt_4-8k", "H12_share_1-2k", "H13_share_2-4k", "H14_share_4-8k", "H15_share_8-12.5k", "H18_hfl", "L1_S"]: D[c[:3]] = ns.T[c].values
Cr = D.corr(method="spearman").loc[["H7_", "H8_", "H9_", "H10"], ["H7_", "H8_", "H9_", "H10", "n", "H1_", "H3_", "H5_", "H12", "H13", "H14", "H15", "H18", "L1_"]].round(2)
md = lambda df: "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join("" if x is None else str(x) for x in r_) + " |" for r_ in df.values)
txt = [f"정의 검증(모델 H7~H10과 최대 차이): {chk:.1e}, 축소 p0 = {np.round(p0, 3).tolist()}\n", "분리도 · 이동도\n" + md(T1), "\n충격 수 층별 H9(4–8k) 중앙값\n" + md(T2), "\n충격 단위 맞춤도\n" + md(fitq), "\n경계 파일: 값 (정상 내 백분위)\n" + md(T3), "\n학습 150 스피어만 상관\n" + md(Cr.reset_index())]
open(os.path.join(HERE, "결과.md"), "w", encoding="utf-8").write("\n".join(txt)); print("\n".join(txt))
