"""평가: 중첩 교차검증으로 F₂를 정직하게 계산한다.

- 바깥 fold: 점수를 내는 파일. 학습(스케일러·모델·임계값)에 쓰지 않는다
- 안쪽 fold: 바깥 학습 파일 안에서 다시 CV → OOF 점수로 F₂ 최대 임계값 선택
규칙 근거: vault/10_스펙/pipeline.md "폴드 안에서 fit", "임계값", "평가·보고"
"""
import numpy as np
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


def nested_cv(F, y, groups, make_model, n_outer=5, n_inner=4, seeds=range(5)):
    """중첩 CV. make_model: 호출할 때마다 새 (학습 전) 모델을 돌려주는 함수

    반환 dict
    - f2, recall, precision: seed별 값 배열 (바깥 fold 예측을 모아 계산)
    - miss_counts, fp_counts: 파일별로 seed 중 몇 번 놓쳤는지 / 오경보였는지
    - thresholds: 바깥 fold마다 고른 임계값
    """
    F, y, groups = np.asarray(F, dtype=float), np.asarray(y), np.asarray(groups)
    out = {"f2": [], "recall": [], "precision": [], "thresholds": []}
    miss = np.zeros(len(y), dtype=int)
    fp = np.zeros(len(y), dtype=int)

    for seed in seeds:
        pred = np.empty(len(y), dtype=int)
        outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        for tr, te in outer.split(F, y, groups):
            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
            oof = _oof_scores(F[tr], y[tr], groups[tr], make_model, inner)
            t = best_f2_threshold(y[tr], oof)            # 안쪽 OOF로만 임계값 선택
            model = make_model().fit(F[tr], y[tr])        # 바깥 학습 파일 전체로 다시 학습
            pred[te] = (_scores(model, F[te]) >= t).astype(int)
            out["thresholds"].append(t)
        out["f2"].append(fbeta_score(y, pred, beta=2))
        out["recall"].append(recall_score(y, pred))
        out["precision"].append(precision_score(y, pred, zero_division=0))
        miss += (y == 1) & (pred == 0)
        fp += (y == 0) & (pred == 1)

    out = {k: np.array(v) for k, v in out.items()}
    out["miss_counts"], out["fp_counts"] = miss, fp
    return out


# ---------------------------------------------------------------------------
# 슬라이싱 버전 (EXP-003): 조각으로 학습, 파일 단위로 집계해 평가
# ---------------------------------------------------------------------------
def _rows(file_idx, files):
    """files에 속한 조각의 행 번호"""
    return np.flatnonzero(np.isin(file_idx, files))


def nested_cv_sliced(F, file_idx, y, make_model, agg=np.max,
                     n_outer=5, n_inner=4, seeds=range(5)):
    """조각 피처로 학습하고 파일 단위로 평가하는 중첩 CV

    - F: (조각 수, 피처 수), file_idx: 조각 → 파일 번호, y: (파일 수,) 파일 라벨
    - 폴드는 파일 단위로 나눈다. nested_cv와 같은 y·groups·seed로 split하므로 파일 분할이 같다
    - 조각 라벨 = 소속 파일 라벨. 파일 점수 = 그 파일 조각 점수의 agg
    - 임계값·F₂는 파일 단위. 반환 dict 형태는 nested_cv와 같다
    """
    F, file_idx, y = np.asarray(F, dtype=float), np.asarray(file_idx), np.asarray(y)
    files = np.arange(len(y))
    groups = files                                    # 파일 하나 = 그룹 하나 (nested_cv와 같음)
    out = {"f2": [], "recall": [], "precision": [], "thresholds": []}
    miss = np.zeros(len(y), dtype=int)
    fp = np.zeros(len(y), dtype=int)

    def fit_on(train_files):
        r = _rows(file_idx, train_files)
        return make_model().fit(F[r], y[file_idx[r]])

    def score_files(model, test_files):
        r = _rows(file_idx, test_files)
        s, fi = _scores(model, F[r]), file_idx[r]
        return np.array([agg(s[fi == f]) for f in test_files])

    for seed in seeds:
        pred = np.empty(len(y), dtype=int)
        outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        for tr, te in outer.split(files, y, groups):
            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
            oof = np.empty(len(tr))
            for itr, ite in inner.split(files[tr], y[tr], groups[tr]):
                oof[ite] = score_files(fit_on(files[tr][itr]), files[tr][ite])
            t = best_f2_threshold(y[tr], oof)             # 안쪽 OOF(파일 단위)로만 임계값 선택
            model = fit_on(files[tr])                     # 바깥 학습 파일의 모든 조각으로 다시 학습
            pred[te] = (score_files(model, files[te]) >= t).astype(int)
            out["thresholds"].append(t)
        out["f2"].append(fbeta_score(y, pred, beta=2))
        out["recall"].append(recall_score(y, pred))
        out["precision"].append(precision_score(y, pred, zero_division=0))
        miss += (y == 1) & (pred == 0)
        fp += (y == 0) & (pred == 1)

    out = {k: np.array(v) for k, v in out.items()}
    out["miss_counts"], out["fp_counts"] = miss, fp
    return out
