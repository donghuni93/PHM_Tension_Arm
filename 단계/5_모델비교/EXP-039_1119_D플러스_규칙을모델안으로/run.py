# %% [markdown]
# EXP-039 D+ 피처로 규칙을 모델 안으로 (AE + ET(D+))
# - 가설: H2′(8.3 kHz 모드) · H1(유격). 8.3 kHz 규칙(P83·W83)과 유격 반복 규칙(K1·K2·K3)이 쓰던 연속 수치를 ET 입력에 넣으면 규칙(사후 문턱) 없이도 002·014류를 가르는가
# - 바꾼 것: ET 입력 ext150 → D+ 81열(대표 62 + 울림 4 + 물리 10 + 규칙 기반 5, src/dplus.py), 규칙 유무 비교. AE 전문가·로지스틱 결합·중첩 CV는 EXP-025와 같다
# 사용: python "단계/5_모델비교/EXP-039_1119_D플러스_규칙을모델안으로/run.py"   (AE 학습이 많아 10분 안팎)
# 기록 후에는 이 파일을 고치지 않는다. 바꾸려면 새 EXP 폴더.

# %% 설정
import os, sys, warnings
warnings.filterwarnings("ignore")
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from sklearn.model_selection import StratifiedKFold
from src.exp import Experiment, load_train, cached_table, feature_table_ext
from src import viz
from src.data import load_folder
from src.evaluate import nested_cv, best_f2_threshold
from src.dplus import DPlus, AEET, rule_flags
from src.final import _fit_ae, _ae_score
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):      # src 임포트 뒤에 글꼴 설정 (viz가 덮어씀)
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
BASE, MAIN = "기준 EXP-025 재현 (AE+ET ext150 + 규칙)", "AE+ET(D+ 81열) 규칙 없음 (새 구성)"
ex = Experiment(__file__, "EXP-039", "D+ 피처로 규칙을 모델 안으로 (AE + ET(D+))", ["H1", "H2′"],
                member="이동훈", model="AE + ET(D+ 81열) → 로지스틱 (규칙 없음 / + 규칙)",
                changes="ET 입력을 ext150 → D+(대표 62 + 울림 4 + 물리 10 + 규칙 기반 5)로 교체, 규칙 유무 비교",
                focus=("abn_002", "abn_017", "abn_022", "nor_116", "nor_055"))
X, y, names = load_train(); E = cached_table("ext", feature_table_ext, X); g = np.arange(len(y))
dp = DPlus().fit(X, E); D = dp.D_; D76 = D[:, :76]
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val")); Dv, Tv = dp.transform(Xv)
Ev = Tv["E"]; assert Ev.shape[1] == E.shape[1], "ext150 열 수가 다르다"
CFG = {BASE: (np.c_[g, E.values], np.c_[g[:0], Ev] if False else Ev, True),
       "AE+ET(ext150) 규칙 없음 (EXP-014 재현)": (np.c_[g, E.values], Ev, False),
       "AE+ET(D 76열) + 규칙": (np.c_[g, D76], Dv[:, :76], True),
       "AE+ET(D+ 81열) + 규칙": (np.c_[g, D], Dv, True),
       MAIN: (np.c_[g, D], Dv, False)}

# %% 중첩 CV (바깥 5 × 안쪽 4, seed 5개)
R = {}
for nm, (F_, _fv, rl) in CFG.items():
    R[nm] = nested_cv(F_, y, g, lambda rl=rl: AEET(rules=rl), n_jobs=-1)
    ex.record(nm, R[nm], names, main=(nm == MAIN))
    th = np.asarray(R[nm]["thresholds"]); ex.note(f"임계값 {nm}", f"바깥 fold 25개: 중앙값 {np.median(th):.2f}, 사분위 {np.percentile(th, 25):.2f}–{np.percentile(th, 75):.2f}, 범위 {th.min():.2f}–{th.max():.2f}")

# %% OOF(5겹 seed 0) 전문가 점수 · 임계값 · val 점수 (구성마다)
def oof(F_, rl):
    S = np.zeros((len(y), 2)); sc = np.zeros(len(y)); on = np.zeros(len(y), bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F_, y):
        m = AEET(rules=rl).fit(F_[tr], y[tr]); S[te], on[te] = m.expert_scores(F_[te]); sc[te] = m.st_.decision_function(m.sc_.transform(S[te])) + 100.0 * on[te]
    return S, sc, on
ae_final = _fit_ae(dp.T_["LP"][y == 0]); ae_v = _ae_score(ae_final, Tv["LP"]); s83v, knv = rule_flags(Tv, dp.T_, y); ruv = s83v | knv
P = {}
for nm, (F_, fv, rl) in CFG.items():
    S, sc, on = oof(F_, rl); thr = best_f2_threshold(y, sc); fin = AEET(rules=rl).fit(F_, y)
    Sv = np.c_[ae_v, fin._et(fin.X_, y, fv)]; sv = fin.st_.decision_function(fin.sc_.transform(Sv)) + 100.0 * (ruv if rl else 0)
    P[nm] = dict(S=S, sc=sc, on=on, thr=thr, fin=fin, Sv=Sv, sv=sv, pv=sv >= thr)
    p = sc >= thr; ex.note(f"OOF {nm}", f"임계값 {thr:.2f}, 재현율 {int((p & (y == 1)).sum())}/30, 오경보 {int((p & (y == 0)).sum())}/120, val 고장 판정 {int((sv >= thr).sum())}/40")
    print(f"{nm:40s} OOF 임계값 {thr:.2f} 재현율 {int((p & (y == 1)).sum())}/30 오경보 {int((p & (y == 0)).sum())}/120 | val 고장 판정 {int((sv >= thr).sum())}/40", flush=True)
pd.DataFrame({"file": nv, **{nm: P[nm]["pv"].astype(int) for nm in CFG}, **{f"{nm} 점수": P[nm]["sv"] for nm in CFG}}).round(3).to_csv(os.path.join(ex.dir, "val_flags.csv"), index=False, encoding="utf-8-sig")

# %% 그림 1: 놓친 파일 vs 정상 5–95% (기본)
f, Lp = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, Lp, y, names, ex.focus), "psd_focus", "놓친 고장(002·017·022)과 오경보 정상(116·055)의 PSD가 정상 범위(띠) 밖으로 나가는 주파수가 있는가")

# %% 그림 2: t-SNE (기본, 입력 공간별 + 이상 점수 색)
fig, _ = viz.tsne({"ext150 피처": E.values, "D+ 81열": D}, y, names, mark=ex.focus, score=P[MAIN]["sc"])
ex.save_fig(fig, "tsne", "ext150과 D+ 입력 공간에서 고장이 정상과 따로 모이는가. 아래 줄 = 새 구성의 OOF 이상 점수")

# %% 그림 3: 핵심 — logistic_plane (AE z × ET z, OOF seed 0). 기준 | 새 구성, × = 자주 틀린 파일(5 seed 중 3회+)
PUR, LIGHT, DARK = "#7B1FA2", "#CE93D8", "#4A148C"
def plane(a, nm, val=False, title=""):
    p = P[nm]; S, sc, on, thr, fin = p["S"], p["sc"], p["on"], p["thr"], p["fin"]
    mu, sd = S[y == 0].mean(0), S[y == 0].std(0); Z = (S - mu) / sd
    w = fin.st_.coef_[0] / fin.sc_.scale_; b = fin.st_.intercept_[0] - (fin.st_.coef_[0] * fin.sc_.mean_ / fin.sc_.scale_).sum()
    gx = np.linspace(-3, 5.5, 90); GX, GY = np.meshgrid(gx, gx); Zg = w[0] * (GX * sd[0] + mu[0]) + w[1] * (GY * sd[1] + mu[1]) + b
    a.contourf(GX, GY, Zg, levels=[thr, 1e9], colors=["#FFE0B2"], alpha=0.5); a.contour(GX, GY, Zg, levels=[thr], colors=["k"], linewidths=1.6); a.contour(GX, GY, Zg, levels=[thr - 1, thr + 1], colors=["k"], linewidths=0.7, linestyles=":")
    show = ~(on & (y == 1))                                                          # 규칙으로 잡힌 고장은 평면에서 뺀다 (상자에 적음)
    a.scatter(Z[show & (y == 0), 0], Z[show & (y == 0), 1], s=22, color="#2a78d6", alpha=.55, label="정상 (OOF)"); a.scatter(Z[show & (y == 1), 0], Z[show & (y == 1), 1], s=36, color="#eb6834", alpha=.9, label="고장 (OOF)")
    R_ = R[nm]; bad = np.where((R_["miss_counts"] >= 3) | (R_["fp_counts"] >= 3))[0]
    a.scatter(Z[bad, 0], Z[bad, 1], marker="x", s=90, color="k", lw=1.4, label="× 자주 틀림")
    for i in bad: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), Z[i], fontsize=8, xytext=(4, 4), textcoords="offset points")
    hid = [names[i][-3:] for i in np.where(on & (y == 1))[0]]
    if hid: a.text(0.02, 0.02, "규칙으로 고장 판정(평면에서 뺌):\n" + " ".join(hid), transform=a.transAxes, fontsize=7, va="bottom", bbox=dict(fc="white", ec="gray", alpha=.8))
    if val:
        Zv_ = (p["Sv"] - mu) / sd; a.scatter(Zv_[:, 0], Zv_[:, 1], c=np.where(p["pv"], DARK, LIGHT), s=95, edgecolor=DARK, lw=1.4, zorder=5, label="val: 고장 판정 (진한 보라)")
        a.scatter([], [], c=LIGHT, s=95, edgecolor=DARK, label="val: 정상 판정 (연보라)")
        for i in np.where(p["pv"])[0]: a.annotate(nv[i][-3:], Zv_[i], fontsize=8, color=DARK, weight="bold", xytext=(5, 5), textcoords="offset points")
    a.set_xlim(-3, 5.5); a.set_ylim(-3, 5.5); a.set_xlabel("AE z (스펙트럼 모양 이상)"); a.set_ylabel("ET z (충격 피처 이상)"); a.legend(fontsize=7, loc="lower right"); a.set_title(title or nm, fontsize=9)
fig, ax = plt.subplots(1, 2, figsize=(15, 6.8)); plane(ax[0], BASE, title=f"기준 EXP-025 재현 (F2 {ex.res['configs'][BASE]['f2']:.3f})"); plane(ax[1], MAIN, title=f"새 구성: AE + ET(D+), 규칙 없음 (F2 {ex.res['configs'][MAIN]['f2']:.3f})")
ex.save_fig(fig, "logistic_plane", "AE z × ET z 평면의 로지스틱 경계(굵은 선 = 임계값, 점선 ±1). 기준은 규칙으로 잡힌 고장을 뺐고 새 구성은 규칙 없이 전부 보인다. ×는 5 seed 중 3번 이상 틀린 파일", key=True)

# %% 그림 4: 핵심 — 새 구성 평면 + val(보라)
fig, ax = plt.subplots(1, 2, figsize=(15, 6.8)); plane(ax[0], MAIN, val=True, title="새 구성(규칙 없음) + val (보라)"); plane(ax[1], "AE+ET(D+ 81열) + 규칙", val=True, title="새 구성 + 규칙 + val (보라)")
ex.save_fig(fig, "val_plane", "새 구성에 val 40개를 보라색으로 겹침. 진한 보라 = 고장 판정. 왼쪽은 규칙 없이, 오른쪽은 규칙을 더했을 때", key=True)

# %% 그림 5: 어려운 파일 (구성별 틀린 횟수)
bad = sorted({i for r in R.values() for i in np.where((r["miss_counts"] >= 3) | (r["fp_counts"] >= 3))[0]}); cn = list(CFG)
M = np.array([[R[c]["miss_counts"][i] if y[i] else R[c]["fp_counts"][i] for c in cn] for i in bad])
fig, a = plt.subplots(figsize=(10, 0.45 * len(bad) + 2)); im = a.imshow(M, cmap="Reds", vmin=0, vmax=5, aspect="auto")
a.set_xticks(range(len(cn))); a.set_xticklabels([c.replace(" (", "\n(") for c in cn], fontsize=7); a.set_yticks(range(len(bad))); a.set_yticklabels([f"{names[i]} ({'고장 놓침' if y[i] else '정상 오경보'})" for i in bad], fontsize=8)
for r_ in range(M.shape[0]):
    for c_ in range(M.shape[1]): a.text(c_, r_, int(M[r_, c_]), ha="center", va="center", fontsize=8)
fig.colorbar(im, label="5 seed 중 틀린 횟수"); ex.save_fig(fig, "hard_files", "구성마다 어떤 파일을 몇 번 틀렸는가 (고장 = 놓침, 정상 = 오경보)")

# %% 그림 6: 오늘 정리 — F2 비교 + val 판정 매트릭스
fig, ax = plt.subplots(1, 2, figsize=(18, 6.6)); a = ax[0]; cf = ex.res["configs"]
vals = [cf[c]["f2"] for c in cn]; sds = [cf[c]["f2_sd"] for c in cn]; cols_ = ["#9E9E9E", "#9E9E9E", "#B39DDB", "#9575CD", PUR]
a.barh(range(len(cn)), vals, xerr=sds, color=cols_, capsize=3)
for xv, lab in ((0.857, "EXP-014 0.857"), (0.942, "EXP-025 0.942"), (0.955, "EXP-034 c 0.955 (폐기)")): a.axvline(xv, color="k", ls=":", lw=0.8); a.text(xv, len(cn) - 0.45, lab, rotation=90, fontsize=7, va="top", ha="right")
a.set_yticks(range(len(cn))); a.set_yticklabels([c.replace(" (", "\n(") for c in cn], fontsize=8); a.set_xlim(0.78, 1.0); a.set_xlabel("중첩 F₂ (seed 5 × 바깥 5 × 안쪽 4)"); a.invert_yaxis(); a.set_title("구성별 F₂ (보라 = 새 D+ 구성)")
for k in range(len(cn)): a.text(vals[k] + sds[k] + 0.003, k, f"{vals[k]:.3f}  R {cf[cn[k]]['recall']:.2f} · 오경보 {cf[cn[k]]['fpr']:.3f}", va="center", fontsize=7)
a = ax[1]; Fm = np.array([P[c]["pv"] for c in cn]).T.astype(int); order = np.argsort(-Fm.sum(1), kind="stable"); keep = [i for i in order if Fm[i].sum() > 0]
a.imshow(Fm[keep], cmap="Purples", vmin=0, vmax=1.4, aspect="auto"); a.set_xticks(range(len(cn))); a.set_xticklabels([c.replace(" (", "\n(") for c in cn], fontsize=7, rotation=0)
a.set_yticks(range(len(keep))); a.set_yticklabels([nv[i] for i in keep], fontsize=8)
for r_, i in enumerate(keep): a.text(len(cn) - 0.45, r_, f"{Fm[i].sum()}/5", va="center", fontsize=7)
a.set_title("val 40개 중 하나라도 고장 판정된 파일 (진한 칸 = 고장 판정)")
fig.tight_layout(); ex.save_fig(fig, "summary", "구성별 F₂ 비교와 val 판정 일치도 (한 장 요약)", key=True)
print("완료:", ex.dir)
