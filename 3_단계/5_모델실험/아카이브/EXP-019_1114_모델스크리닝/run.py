# %% [markdown]
# EXP-019 후보 모델 1회 스크리닝 — 회색지대 정보를 스스로 뽑는가
# - 배경: EXP-004·015에서 모델만 바꿔도 같은 고장을 놓쳤다 → 병목은 경계 모양이 아니라 입력 정보(피처).
#   2-11에서 본 회색지대: 정상 N-A(큰 사건 한 번) · N-B(6–8 kHz 모양) / 고장 002·029(8 kHz 좁은 대역, 조용함),
#   005·014·025(반복 충격인데 AE 약함), 017·022(정보 없음)
# - 순서 (결정.md '모델 탐색'): MiniRocket → 1D-CNN(창 MIL) → 합성곱 AE → Deep SAD → 2D-CNN → AR 잔차 → TabPFN → MSPC → CatBoost
# - 모델마다 셀 하나. 결과를 설명·확인한 뒤 다음 셀을 추가한다 (스크리닝 기록이라 셀 추가는 허용, 기존 셀은 고치지 않음)
# - 측정 (quick_cv 5-fold × seed 3, 임계값 없음)
#   (a) 단독 PR-AUC
#   (b) 회색지대 순위: 고장 7개·정상 10개의 정상 중 백분위 (pct = 자기보다 낮은 정상 비율)
#   (c) 보완성: EXP-014 결합(AE + ET → 로지스틱)에 새 전문가를 한 열 더한 PR-AUC vs 기준
# - 통과 기준 (실행 전 고정, 하나라도 만족하면 정식 실험 EXP-020+에서 중첩 CV)
#   ① (c) PR-AUC ≥ 기준 평균 + 2 × 기준 seed sd
#   ② (c)에서 기준 pct < 0.90이던 회색 고장 2개 이상이 pct ≥ 0.90으로 오르고, 회색 정상 평균 pct가 기준보다 오르지 않음
# - 하이퍼파라미터는 문헌 기본값 고정. 튜닝하지 않는다
# 실행: conda run -n qe-phm python 3_단계/5_모델비교/스크리닝/EXP-019_1114_모델스크리닝/run.py

# %% 설정
import os, sys, time, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv
from src.experts import ScoreStack, ae_mean, et_logit, ae_segerr, et_trees, warm_cache
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ex = Experiment(__file__, "EXP-019", "후보 모델 1회 스크리닝 (회색지대 정보)", ["H1", "H2", "H3"],
                member="이동훈", model="스크리닝: MiniRocket …",
                changes="EXP-014 결합에 후보 모델 점수를 한 열 더해 보완성·회색지대 순위 확인", focus=())
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
nor = y == 0
GRAY_A = ["abn_002", "abn_005", "abn_014", "abn_017", "abn_022", "abn_025", "abn_029"]
GRAY_N = ["nor_025", "nor_027", "nor_055", "nor_059", "nor_060", "nor_062", "nor_073", "nor_086", "nor_106", "nor_116"]
GRAY = GRAY_A + GRAY_N
ia, in_ = [names.index(n) for n in GRAY_A], [names.index(n) for n in GRAY_N]

warm_cache([ae_segerr, et_trees], seeds=SEEDS)


def qcv(experts, **kw):
    q = quick_cv(g[:, None], y, g, lambda: ScoreStack(experts, **kw), seeds=SEEDS)
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    return q


BASE = qcv((ae_mean, et_logit), combine="lr")
BAR = BASE["pr_auc"].mean() + 2 * BASE["pr_auc"].std()
R = {"기준 EXP-014 (AE+ET)": BASE}
SCREEN = ex.res.setdefault("screen", {})
print(f"기준 PR-AUC {BASE['pr_auc'].mean():.3f} ± {BASE['pr_auc'].std():.3f} → 기준선 ① {BAR:.3f}", flush=True)


AE_ALONE = qcv((ae_mean,), combine="col0")
ET_ALONE = qcv((et_logit,), combine="col0")
print(f"참고: AE 단독 {AE_ALONE['pr_auc'].mean():.3f} · ET 단독 {ET_ALONE['pr_auc'].mean():.3f}", flush=True)


def screen(name, fn):
    """후보 하나: (a) 단독 (b) 회색지대 순위 (c) 결합 보완성 + 통과 판정
    (d) 교체 (11-14 추가, 사용자 요청 'EXP-014를 베이스로 교체·추가 점검'): AE+후보(ET 자리 교체), ET+후보(AE 자리 교체)"""
    t0 = time.time()
    a = qcv((fn,), combine="col0")
    c = qcv((ae_mean, et_logit, fn), combine="lr")
    rE, rA = qcv((ae_mean, fn), combine="lr"), qcv((et_logit, fn), combine="lr")
    R[f"{name} 단독"], R[f"{name} + 기준"] = a, c
    lifted = [GRAY_A[k] for k, i in enumerate(ia) if BASE["pct"][i] < .9 <= c["pct"][i]]
    dn = c["pct"][in_].mean() - BASE["pct"][in_].mean()
    p1 = c["pr_auc"].mean() >= BAR
    p2 = len(lifted) >= 2 and dn <= 0
    SCREEN[name] = dict(alone=[round(float(a["pr_auc"].mean()), 3), round(float(a["pr_auc"].std()), 3)],
                        stack=[round(float(c["pr_auc"].mean()), 3), round(float(c["pr_auc"].std()), 3)],
                        lifted=lifted, gray_n_delta=round(float(dn), 3), pass1=bool(p1), pass2=bool(p2),
                        verdict="통과" if (p1 or p2) else "탈락", sec=round(time.time() - t0),
                        swap_et=[round(float(rE["pr_auc"].mean()), 3), round(float(rE["pr_auc"].std()), 3)],
                        swap_ae=[round(float(rA["pr_auc"].mean()), 3), round(float(rA["pr_auc"].std()), 3)])
    s = SCREEN[name]
    line = (f"{name}: 단독 {s['alone'][0]:.3f} ± {s['alone'][1]:.3f} · +기준 {s['stack'][0]:.3f} ± {s['stack'][1]:.3f} "
            f"(기준 {BASE['pr_auc'].mean():.3f}, ① 선 {BAR:.3f}) · 올라온 회색 고장 {' '.join(lifted) or '없음'} · "
            f"회색 정상 평균 pct {dn:+.3f} → {s['verdict']} ({s['sec']}초)\n"
            f"  교체: AE + {name} (ET 자리) {s['swap_et'][0]:.3f} ± {s['swap_et'][1]:.3f} · "
            f"ET + {name} (AE 자리) {s['swap_ae'][0]:.3f} ± {s['swap_ae'][1]:.3f}")
    print(line, flush=True)
    ex.note(f"스크리닝: {name}", line + "\n\n파일      기준   단독   +기준\n" + "\n".join(
        f"{n:9s} {BASE['pct'][names.index(n)]:.2f}   {a['pct'][names.index(n)]:.2f}   {c['pct'][names.index(n)]:.2f}" for n in GRAY))
    _heatmap()


def _heatmap():
    """회색지대 위험도: 고장 = 자기보다 높은 정상 비율(1 − pct), 정상 = pct. 0 = 잘 구분, 1 = 정반대"""
    cols = list(R)
    M = np.array([[1 - R[k]["pct"][names.index(n)] if n.startswith("abn") else R[k]["pct"][names.index(n)] for k in cols] for n in GRAY])
    fig, ax = plt.subplots(figsize=(2.2 + 1.3 * len(cols), 8))
    im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if M[i, j] < .7 else "w")
    ax.axhline(len(GRAY_A) - .5, color="k", lw=.8)
    ax.set_yticks(range(len(GRAY))); ax.set_yticklabels(GRAY, fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([k.replace(" ", "\n", 1) for k in cols], fontsize=7.5)
    ax.set_title("회색지대 위험도 (quick_cv seed 3 평균). 위 = 고장(1 − pct), 아래 = 정상(pct). 낮을수록 좋음", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
    ex.save_fig(fig, "gray_ranks", "후보 모델이 기준(EXP-014)이 헷갈리던 파일을 다르게 보는가. 같은 열의 고장은 낮게, 정상도 낮게 나와야 한다")


# %% 1. MiniRocket + 릿지 (원신호, 평균 제거·200 Hz 고역통과, 커널 10,000개)
# 노리는 것: 회색 정상 N-A(큰 사건 한 번)과 반복 충격 고장(005·014·025)을 패턴 "출현 비율"(PPV)로 구분
import traceback
import src.candidates as C


def run(name, fn, n_jobs):
    """캐시를 채우고 스크리닝. 한 모델의 오류가 나머지를 막지 않게 (오류는 기록)"""
    try:
        warm_cache([fn], seeds=SEEDS, n_jobs=n_jobs)
        screen(name, fn)
    except Exception:
        tb = traceback.format_exc(); print(f"{name} 오류\n{tb}", flush=True)
        ex.note(f"스크리닝: {name}", "오류\n" + tb[-1500:])


run("MiniRocket", C.minirocket_ridge, 8)

# %% 2. 1D-CNN + 창 MIL (원신호 0.2초 창, 파일 = 상위 3창 평균). 노리는 것: 반복 충격(N-A vs 005·014·025)
run("1D-CNN", C.cnn1d_mil, 1)

# %% 3. 합성곱 AE (log STFT 0.33초 × 48.8 Hz, 정상만). 노리는 것: N-B(6–8 kHz) vs 002·029(8 kHz), 시간 국소 이상
run("합성곱AE", C.convae_stft, 1)

# %% 4. Deep SAD (같은 입력, 고장 30개도 사용). 노리는 것: 3번 공간에서 고장 쪽으로 경계를 밀어냄
run("DeepSAD", C.deepsad_stft, 1)

# %% 5. 2D-CNN + 구간 MIL (같은 입력, 지도 학습)
run("2D-CNN", C.cnn2d_mil, 1)

# %% 6. AR 예측 잔차 (정상 AR(32) → 잔차 충격 피처 5개 → 로지스틱). 노리는 것: N-A (가벼운 물리 기준)
run("AR잔차", C.ar_resid_lr, 8)

# %% 7. TabPFN — 제외 (11-14 사용자 결정: 가중치에 외부 계정·API 키가 필요한 모델은 쓰지 않음)
ex.note("스크리닝: TabPFN", "제외: 가중치 다운로드에 외부 라이선스 동의·API 키 필요 → 사용하지 않음 (사용자 결정 11-14)")

# %% 8. PCA-MSPC (ext150, 정상 PCA 분산 90%, T² + SPE). 품질공학 관리도 기준선
run("MSPC", C.mspc_ext, 8)

# %% 9. CatBoost (ext150, 500회, 클래스 균형). 경계만 바꾸는 비교군
run("CatBoost", C.catboost_ext, 8)

# %% 요약표
rows = ["후보         단독          +기준(추가)    AE+후보(ET교체)  ET+후보(AE교체)  올라온 회색 고장       정상Δ   판정"]
rows.append(f"{'기준':10s}  {'—':12s}  {BASE['pr_auc'].mean():.3f} ± {BASE['pr_auc'].std():.3f}  "
            f"(AE 단독 {AE_ALONE['pr_auc'].mean():.3f} · ET 단독 {ET_ALONE['pr_auc'].mean():.3f})")
for k, s in SCREEN.items():
    rows.append(f"{k:10s}  {s['alone'][0]:.3f} ± {s['alone'][1]:.3f}  {s['stack'][0]:.3f} ± {s['stack'][1]:.3f}  "
                f"{s['swap_et'][0]:.3f} ± {s['swap_et'][1]:.3f}    {s['swap_ae'][0]:.3f} ± {s['swap_ae'][1]:.3f}    "
                f"{' '.join(x.replace('abn_', '') for x in s['lifted']) or '—':20s}  {s['gray_n_delta']:+.3f}  {s['verdict']}")
ex.note("요약 (quick_cv PR-AUC, seed 3)", "\n".join(rows))
print("\n".join(rows), flush=True)


# %% 분리 판정 (11-14 사용자: 수치보다 '고장을 확실히 가르는가'. PR-AUC는 참고)
# OOF 점수, seed 3개 모두에서
#   확실한 고장 = 모든 정상보다 높음 / 확실한 정상 = 모든 고장보다 낮음 / 회색지대 = 나머지(겹치는 구간)
#   여유(마진) = 정상 99%선과 고장 1%선의 간격을 정상 IQR로 나눈 값 (양수면 분리, 클수록 안정)
def separation(q):
    sure_a, sure_n = np.ones(len(y), bool), np.ones(len(y), bool)
    margins = []
    for o in q["oof"]:
        n, a = o[y == 0], o[y == 1]
        sure_a &= (y == 1) & (o > n.max())
        sure_n &= (y == 0) & (o < a.min())
        iqr = np.subtract(*np.percentile(n, [75, 25])) + 1e-12
        margins.append((np.percentile(a, 1) - np.percentile(n, 99)) / iqr)
    gray = ~(sure_a | sure_n)
    return dict(sure_a=int(sure_a.sum()), sure_n=int(sure_n.sum()), gray_a=[names[i] for i in np.flatnonzero(gray & (y == 1))],
                gray_n=[names[i] for i in np.flatnonzero(gray & (y == 0))], margin=float(np.mean(margins)))


S0 = separation(BASE)
rows = [f"{'구성':22s} 확실한 고장/30  확실한 정상/120  회색(고장+정상)  여유   고장 회색 (기준 대비 +새로 들어감 −빠짐)"]
for k, q in R.items():
    s = separation(q)
    ga = set(s["gray_a"])
    diff = " ".join([f"+{x[4:]}" for x in sorted(ga - set(S0["gray_a"]))] + [f"−{x[4:]}" for x in sorted(set(S0["gray_a"]) - ga)])
    rows.append(f"{k:22s} {s['sure_a']:4d}            {s['sure_n']:4d}             {len(s['gray_a']):3d}+{len(s['gray_n']):3d}       "
                f"{s['margin']:+.2f}  {' '.join(x[4:] for x in s['gray_a'])}  [{diff or '같음'}]")
ex.note("분리 판정 (OOF, seed 3 모두)", "\n".join(rows))
print("\n".join(rows), flush=True)
print("done", flush=True)
