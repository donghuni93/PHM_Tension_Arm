# %% [markdown]
# EXP-025 README2 논의용 그림 — "고장과 정상을 확실히 가르는 선이 있는가", 비선형(SVM RBF) 경계와 간격
# 1. svm_plane: 기준 결합의 두 전문가 z 평면(AE z, ET z, OOF seed 0)에서
#    (a) 로지스틱(선형) (b) RBF SVM 보통(C=1) (c) RBF SVM 학습 100% 분리(C=1000, γ=20) 의 경계·여백(±1)
#    각 패널 제목: 학습 데이터에서 틀린 수(자기 자신으로 채점) vs 5-fold OOF에서 틀린 수 (결정값 0 기준)
# 2. gap: 최종 EXP-025의 (점수 − 임계값) seed 5번 평균(fig_errors의 errors_margin.npy) — 고장·정상 사이 간격이 있나
# 3. 참고 quick_cv: 두 전문가 z + 규칙 축(r, 2위/1위, 8.3k 돌출·폭)을 RBF SVM / 로지스틱에 넣으면 (탐색, EXP 아님)
#    주의: z는 바깥 seed 0 분할의 OOF 값을 고정 입력으로 쓴다(약한 누수) → 비교용
# 실행: conda run -n qe-phm python 단계/5_모델비교/유의미/EXP-025_유격반복규칙/fig_svm.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv
from src.features import feature_table_knock
from src.experts import ScoreStack, KnockRule, Rule83, peak83_table, ae_mean, et_logit
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y, names = load_train(); g = np.arange(len(y)); F = g[:, None]
NOR, ABN = "#2a78d6", "#eb6834"
s = lambda n: n.replace("abn_", "A").replace("nor_", "N")
LOG = []
def log(*a): LOG.append(" ".join(str(v) for v in a)); print(*a, flush=True)

# %% 두 전문가 z (OOF, 바깥 seed 0)
Z = np.empty((len(y), 2))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    Z[te] = ScoreStack(experts=(ae_mean, et_logit), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
fired = Rule83(0.75, 150)(g) | KnockRule(2.0, 0.7).fit(g)(g)          # 최종 모델에서 규칙이 맡는 파일
np.save(os.path.join(HERE, "figs", "svm_z.npy"), Z)

# %% 1. 세 경계: 학습 오류 vs OOF 오류
MODELS = [("(a) 로지스틱 = 직선", lambda: LogisticRegression(C=1.0, class_weight="balanced")),
          ("(b) RBF SVM 보통 (C=1, γ=scale)", lambda: SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")),
          ("(c) RBF SVM 학습 100% 분리 (C=1000, γ=20)", lambda: SVC(C=1000.0, kernel="rbf", gamma=20.0, class_weight="balanced"))]
Zc = np.clip(Z, -4, 12)
xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 400), np.linspace(-3, 4.5, 400))
G = np.c_[xx.ravel(), yy_.ravel()]
fig, axs = plt.subplots(1, 3, figsize=(21, 7.4))
log("[1] z 평면 경계: 학습(자기 채점) 틀린 수 vs 5-fold OOF 틀린 수 (결정값 0 기준, seed 3 평균) · 그 OOF에서 틀린 파일")
for ax, (nm, mk) in zip(axs, MODELS):
    m = mk().fit(Zc, y)
    tr_err = ((m.decision_function(Zc) >= 0).astype(int) != y)
    oof_wrong = []
    for sd in range(3):
        o = np.empty(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(g, y, g):
            o[te] = mk().fit(Zc[tr], y[tr]).decision_function(Zc[te])
        oof_wrong.append(((o >= 0).astype(int) != y))
    oof_wrong = np.array(oof_wrong)
    often = [names[i] for i in np.flatnonzero(oof_wrong.sum(0) >= 2)]
    log(f"  {nm}: 학습 {tr_err.sum()}개 틀림 · OOF 평균 {oof_wrong.sum(1).mean():.1f}개 틀림 (고장 {oof_wrong[:, y == 1].sum(1).mean():.1f} · 정상 {oof_wrong[:, y == 0].sum(1).mean():.1f}) · "
        f"OOF 2/3+: {' '.join(s(n) for n in often)}")
    D = m.decision_function(G).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.6)
    ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.8, linestyles="--")
    for lab, sel, c, mkr in (("정상", y == 0, NOR, "s"), ("고장", y == 1, ABN, "o")):
        ax.scatter(Zc[sel, 0], Zc[sel, 1], color=c, s=22, lw=0, marker=mkr, label=lab, alpha=.85)
    ax.scatter(Zc[fired, 0], Zc[fired, 1], s=70, facecolor="none", edgecolor="#7a3db8", lw=1.1, label="규칙이 맡는 고장 (8.3k·유격)")
    for n in ("abn_022", "abn_005", "abn_017", "abn_014", "abn_002", "nor_025", "nor_116", "nor_055", "nor_059"):
        i = names.index(n)
        ax.annotate(s(n), Zc[i], xytext=(4, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.grid(alpha=.3); ax.set_aspect("equal")
    ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (충격 피처 이상)")
    ax.set_title(f"{nm}\n학습 {tr_err.sum()}개 틀림 → 새 파일(OOF) {oof_wrong.sum(1).mean():.1f}개 틀림", fontsize=10)
axs[0].legend(fontsize=8, loc="upper right")
fig.suptitle("실선 = 경계(결정값 0), 점선 = 여백(±1). 경계를 휘게 할수록 학습 오류는 0에 가까워지지만 새 파일 오류는 늘어난다", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "svm_plane.png"), dpi=80); plt.close(fig)

# %% 2. 간격: 최종 모델의 (점수 − 임계값) seed 평균
M = np.load(os.path.join(HERE, "figs", "errors_margin.npy"))
mm = np.clip(M, -6, 6).mean(0); mm = np.where(fired, 6.5, mm)
lo_f = mm[(y == 1) & (mm < 6)]; hi_n = mm[y == 0]
gray_lo, gray_hi = np.sort(lo_f)[1], hi_n.max()                         # 022(가장 낮은 고장)를 뺀 다음 고장 ~ 가장 높은 정상
ing = (mm >= gray_lo) & (mm <= gray_hi)
log(f"\n[2] 최종 모델 여유(seed 평균): 가장 낮은 고장 {s(names[int(np.argmin(np.where(y == 1, mm, 99)))])} {lo_f.min():+.2f} · "
    f"다음 고장 {gray_lo:+.2f} · 가장 높은 정상 {s(names[int(np.argmax(np.where(y == 0, mm, -99)))])} {gray_hi:+.2f}")
log(f"  겹치는 구간 [{gray_lo:+.2f}, {gray_hi:+.2f}] 안: 고장 {' '.join(s(names[i]) for i in np.flatnonzero(ing & (y == 1)))} · "
    f"정상 {' '.join(s(names[i]) for i in np.flatnonzero(ing & (y == 0)))}")
fig, ax = plt.subplots(figsize=(16, 4.8))
rng = np.random.default_rng(1)
for lab, sel, c, yv in (("정상", y == 0, NOR, 0), ("고장", y == 1, ABN, 1)):
    ax.scatter(mm[sel], yv + rng.uniform(-.25, .25, sel.sum()), color=c, s=24, alpha=.75, lw=0, label=lab)
ax.axvspan(gray_lo, gray_hi, color="#f5d76e", alpha=.35, label="겹치는 구간 (022 제외)")
ax.axvline(0, color="k", lw=1.2); ax.axvspan(6.1, 6.9, color="#eee", zorder=0)
ax.text(6.5, 1.42, "규칙 +100", ha="center", fontsize=8)
for i in np.flatnonzero(ing | ((y == 1) & (mm < gray_lo))):
    ax.annotate(s(names[i]), (mm[i], y[i]), xytext=(0, 9 if y[i] else -14), textcoords="offset points", fontsize=8,
                ha="center", color=ABN if y[i] else NOR)
ax.set_yticks([0, 1]); ax.set_yticklabels(["정상", "고장"]); ax.set_xlim(-6.3, 7); ax.set_ylim(-.6, 1.6)
ax.set_xlabel("최종 모델 여유 = 점수 - 임계값, seed 5번 평균 (0 오른쪽 = 고장 판정, -6 아래는 -6에 모음)")
ax.set_title("고장과 정상 사이에 빈 간격이 있는가: 노란 구간에 두 라벨이 섞여 있다 = 확실한 선은 없다", fontsize=10)
ax.grid(alpha=.3, axis="x"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "gap.png"), dpi=80); plt.close(fig)

# %% 3. 참고 quick_cv: z + 규칙 축을 한 공간에 넣으면
K = cached_table("knock", feature_table_knock, X)
P83, W83, _ = peak83_table()
r = KnockRule(2.0, 0.7).fit(g).r(g)
SP = {"z 2축": Zc, "z 2축 + 규칙 축 4개(r, 2위/1위, 8.3k 돌출·폭)": np.column_stack([Zc, r, K["K3_top2_ratio"].values, P83, W83 / 100])}
log("\n[3] 참고 quick_cv (5-fold × seed 3, 규칙 없이 한 모델로): PR-AUC (최종 EXP-025 = 0.973)")
for sn, A in SP.items():
    for mn, mk in (("로지스틱", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced"))),
                   ("RBF SVM", lambda: make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", class_weight="balanced")))):
        q = quick_cv(A, y, g, mk)
        nn = q["oof"][:, y == 0]
        risk = {n: np.mean([(nn[k] >= q["oof"][k, names.index(n)]).mean() for k in range(3)]) for n in ("abn_014", "abn_017", "abn_022", "abn_002")}
        log(f"  {sn:40s} {mn:8s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · 위험(위 정상 비율) " + " ".join(f"{s(k)} {v:.2f}" for k, v in risk.items()))

open(os.path.join(HERE, "figs", "svm.txt"), "w", encoding="utf-8").write("\n".join(LOG))
print("done")
