"""평가: 탐색은 quick_cv(5-fold, PR-AUC), 최종 후보 확인은 중첩 교차검증 F₂.

- 바깥 fold: 점수를 내는 파일. 학습(스케일러·모델·임계값)에 쓰지 않는다
- 안쪽 fold: 바깥 학습 파일 안에서 다시 CV → OOF 점수로 F₂ 최대 임계값 선택
규칙 근거: 단계/1_문제·평가/README.md (폴드 안에서 fit, 임계값, 평가·보고)
"""
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold


def _scores(model, F):
    """이상 점수: 클수록 고장. decision_function이 있으면 그것을, 없으면 고장 확률"""
    if hasattr(model, "decision_function"):
        return model.decision_function(F)
    return model.predict_proba(F)[:, 1]


def best_f2_threshold(y, scores):
    """score >= t 를 고장으로 볼 때 F₂가 최대인 t.

    후보는 점수 값 자체. F₂가 같으면 가장 낮은 t (재현율 우선)
    """
    y = np.asarray(y)
    candidates = np.unique(scores)          # 오름차순
    f2 = [fbeta_score(y, (scores >= t).astype(int), beta=2, zero_division=0) for t in candidates]
    return candidates[int(np.argmax(f2))]   # argmax는 첫 번째(가장 낮은 t)를 고른다


def _oof_scores(F, y, groups, make_model, cv):
    """cv로 나눠 각 fold를 나머지로 학습한 모델로 점수 → 모든 행의 OOF 점수"""
    oof = np.empty(len(y))
    for tr, te in cv.split(F, y, groups):
        model = make_model().fit(F[tr], y[tr])
        oof[te] = _scores(model, F[te])
    return oof


def _run_tasks(task, jobs, n_jobs):
    """(seed, 바깥 fold) 작업들을 실행해 jobs 순서대로 결과를 돌려준다.
    n_jobs=1이면 순서대로, 아니면 프로세스 n_jobs개로 병렬(-1 = 모든 코어).
    작업자마다 BLAS 스레드는 1개 (프로세스 수 × BLAS 스레드로 코어를 과하게 잡지 않게)"""
    if n_jobs == 1:
        return [task(*j) for j in jobs]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=n_jobs)(delayed(task)(*j) for j in jobs)


def _collect(results, y, seeds, n_outer):
    """작업 결과 [(te, pred_te, t), ...] (seed 순 → 바깥 fold 순)를 seed별 지표로 모은다"""
    out = {"f2": [], "recall": [], "precision": [], "fpr": [], "thresholds": []}
    miss = np.zeros(len(y), dtype=int)
    fp = np.zeros(len(y), dtype=int)
    for i, _ in enumerate(seeds):
        pred = np.empty(len(y), dtype=int)
        for te, p, t in results[i * n_outer:(i + 1) * n_outer]:
            pred[te] = p
            out["thresholds"].append(t)
        out["f2"].append(fbeta_score(y, pred, beta=2))
        out["recall"].append(recall_score(y, pred))
        out["precision"].append(precision_score(y, pred, zero_division=0))
        out["fpr"].append(float(pred[y == 0].mean()))
        miss += (y == 1) & (pred == 0)
        fp += (y == 0) & (pred == 1)

    out = {k: np.array(v) for k, v in out.items()}
    out["miss_counts"], out["fp_counts"] = miss, fp
    return out


def _outer_fold(F, y, groups, make_model, n_inner, seed, tr, te):
    """바깥 fold 하나: 안쪽 OOF로 임계값 선택 → 바깥 학습 전체로 재학습 → 바깥 테스트 예측"""
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    oof = _oof_scores(F[tr], y[tr], groups[tr], make_model, inner)
    t = best_f2_threshold(y[tr], oof)            # 안쪽 OOF로만 임계값 선택
    model = make_model().fit(F[tr], y[tr])        # 바깥 학습 파일 전체로 다시 학습
    return te, (_scores(model, F[te]) >= t).astype(int), t


def nested_cv(F, y, groups, make_model, n_outer=5, n_inner=4, seeds=range(5), n_jobs=1):
    """중첩 CV. make_model: 호출할 때마다 새 (학습 전) 모델을 돌려주는 함수

    n_jobs: (seed, 바깥 fold) 작업을 동시에 돌릴 프로세스 수. 1 = 순서대로, -1 = 모든 코어.
            결과는 n_jobs와 무관하게 같다. 이때 모델 자체는 n_jobs=1로 (make_et 기본값)

    반환 dict
    - f2, recall, precision: seed별 값 배열 (바깥 fold 예측을 모아 계산)
    - fpr: seed별 오경보율 = FP / 정상 수 (고장 비율과 무관. 1-2 조각 4)
    - miss_counts, fp_counts: 파일별로 seed 중 몇 번 놓쳤는지 / 오경보였는지
    - thresholds: 바깥 fold마다 고른 임계값
    """
    F, y, groups = np.asarray(F, dtype=float), np.asarray(y), np.asarray(groups)
    seeds = list(seeds)
    jobs = [(F, y, groups, make_model, n_inner, seed, tr, te)
            for seed in seeds
            for tr, te in StratifiedGroupKFold(n_splits=n_outer, shuffle=True,
                                               random_state=seed).split(F, y, groups)]
    return _collect(_run_tasks(_outer_fold, jobs, n_jobs), y, seeds, n_outer)


# ---------------------------------------------------------------------------
# 탐색용 5-fold (11-14 결정): 임계값을 고르지 않는 지표로만 비교
# ---------------------------------------------------------------------------
def quick_cv(F, y, groups, make_model, n_splits=5, seeds=range(3), top_k=None):
    """탐색 단계 비교용. 파일 단위 5-fold OOF 점수 → PR-AUC·ROC-AUC (임계값 선택 없음)

    중첩 CV는 최종 후보 확인에만 쓴다. 여기서 F₂를 내지 않는 이유:
    같은 fold로 임계값을 고르면 F₂가 +0.03–0.045 부풀고 순위가 뒤집힘 (1-2 조각 3)

    반환 dict
    - pr_auc, roc_auc: seed별 값 배열
    - oof: (seed 수, 파일 수) OOF 이상 점수
    - miss_rank: 고장 파일별 평균 순위 백분위 (정상 점수 중 그 고장보다 높은 비율, 0 = 가장 잘 잡힘)
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    F, y, groups = np.asarray(F, dtype=float), np.asarray(y), np.asarray(groups)
    seeds = list(seeds)
    oof = np.array([_oof_scores(F, y, groups, make_model,
                                StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s))
                    for s in seeds])
    normal = oof[:, y == 0]
    rank = np.array([(normal >= oof[:, [i]]).mean(axis=1).mean() for i in np.flatnonzero(y == 1)])
    return {
        "pr_auc": np.array([average_precision_score(y, o) for o in oof]),
        "roc_auc": np.array([roc_auc_score(y, o) for o in oof]),
        "oof": oof,
        "miss_rank": dict(zip(np.flatnonzero(y == 1).tolist(), rank.round(3).tolist())),
    }


def separation(oof, y, names, low_q=10):
    """분리 판정 (11-14 사용자: 수치보다 '고장을 확실히 가르는가'). oof = quick_cv의 (seed, 파일) OOF 점수
    seed 모두에서
    - 확실한 고장: 모든 정상보다 높음 / 확실한 정상: 모든 고장보다 낮음 / 회색 = 나머지
    - 위험 정상: 고장 하위 low_q% 선보다 높은 정상 수 (seed 평균). 가장 약한 고장 하나에 끌리지 않는 오경보 지표
    - 여유: (고장 1%선 − 정상 99%선) / 정상 IQR (양수면 분리)"""
    y = np.asarray(y)
    sa, sn, risk, margin = np.ones(len(y), bool), np.ones(len(y), bool), [], []
    for o in np.atleast_2d(oof):
        n, a = o[y == 0], o[y == 1]
        sa &= (y == 1) & (o > n.max())
        sn &= (y == 0) & (o < a.min())
        risk.append(int((n > np.percentile(a, low_q)).sum()))
        margin.append((np.percentile(a, 1) - np.percentile(n, 99)) / (np.subtract(*np.percentile(n, [75, 25])) + 1e-12))
    gray = ~(sa | sn)
    return dict(sure_a=int(sa.sum()), sure_n=int(sn.sum()), risk_n=float(np.mean(risk)), margin=float(np.mean(margin)),
                gray_a=[names[i] for i in np.flatnonzero(gray & (y == 1))],
                gray_n=[names[i] for i in np.flatnonzero(gray & (y == 0))])


# ---------------------------------------------------------------------------
# 슬라이싱 버전 (EXP-003): 조각으로 학습, 파일 단위로 집계해 평가
# ---------------------------------------------------------------------------
def _rows(file_idx, files):
    """files에 속한 조각의 행 번호"""
    return np.flatnonzero(np.isin(file_idx, files))


def _outer_fold_sliced(F, file_idx, y, make_model, agg, n_inner, seed, tr, te):
    """nested_cv_sliced의 바깥 fold 하나 (파일 단위 임계값, 조각으로 학습)"""
    files = np.arange(len(y))
    groups = files                                    # 파일 하나 = 그룹 하나 (nested_cv와 같음)

    def fit_on(train_files):
        r = _rows(file_idx, train_files)
        return make_model().fit(F[r], y[file_idx[r]])

    def score_files(model, test_files):
        r = _rows(file_idx, test_files)
        s, fi = _scores(model, F[r]), file_idx[r]
        return np.array([agg(s[fi == f]) for f in test_files])

    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    oof = np.empty(len(tr))
    for itr, ite in inner.split(files[tr], y[tr], groups[tr]):
        oof[ite] = score_files(fit_on(files[tr][itr]), files[tr][ite])
    t = best_f2_threshold(y[tr], oof)             # 안쪽 OOF(파일 단위)로만 임계값 선택
    model = fit_on(files[tr])                     # 바깥 학습 파일의 모든 조각으로 다시 학습
    return te, (score_files(model, files[te]) >= t).astype(int), t


def nested_cv_sliced(F, file_idx, y, make_model, agg=np.max,
                     n_outer=5, n_inner=4, seeds=range(5), n_jobs=1):
    """조각 피처로 학습하고 파일 단위로 평가하는 중첩 CV

    - F: (조각 수, 피처 수), file_idx: 조각 → 파일 번호, y: (파일 수,) 파일 라벨
    - 폴드는 파일 단위로 나눈다. nested_cv와 같은 y·groups·seed로 split하므로 파일 분할이 같다
    - 조각 라벨 = 소속 파일 라벨. 파일 점수 = 그 파일 조각 점수의 agg
    - 임계값·F₂는 파일 단위. 반환 dict 형태는 nested_cv와 같다. n_jobs도 nested_cv와 같다
    """
    F, file_idx, y = np.asarray(F, dtype=float), np.asarray(file_idx), np.asarray(y)
    files = np.arange(len(y))
    seeds = list(seeds)
    jobs = [(F, file_idx, y, make_model, agg, n_inner, seed, tr, te)
            for seed in seeds
            for tr, te in StratifiedGroupKFold(n_splits=n_outer, shuffle=True,
                                               random_state=seed).split(files, y, files)]
    return _collect(_run_tasks(_outer_fold_sliced, jobs, n_jobs), y, seeds, n_outer)
