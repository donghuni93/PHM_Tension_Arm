"""실험 공용: 데이터·피처 캐시, 모델, 실험 기록(Experiment)

실험 폴더(단계/N_이름/EXP-nnn_이름/run.py)에서:
    from src.exp import *
    ex = Experiment(__file__, "EXP-007", "제목", ["H2", "H3"],
                    member="이동훈", model="ExtraTrees", changes="이전 대비 바꾼 것 한 줄")
    ex.save_fig(fig, "psd_missed")                          # → figs/psd_missed.png + README 그림 목록
    ex.record("구성 이름", nested_cv(...), names, main=True)  # → result.json + README 결과 표
README 맨 위 '노션 카드' 표는 노션 실험 DB 속성(팀원·모델·F2 점수·변경 사항·메모)과 같다.
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.data import load_dataset
from src.features import feature_table, feature_table_ext, feature_table_phys
from src.evaluate import nested_cv
from src import viz
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier

CACHE = os.path.join(ROOT, "outputs", "cache")
DATA = os.path.join(ROOT, "data", "TeamProject_train_변경")
os.makedirs(CACHE, exist_ok=True)

__all__ = ["ROOT", "CACHE", "np", "pd", "viz", "nested_cv", "load_train", "cached_table",
           "feature_table", "feature_table_ext", "feature_table_phys",
           "make_et", "ColModel", "RankAvg", "CollinearityFilter", "vif", "Experiment"]


# ---------- 데이터 ----------
def load_train():
    """X, y, names (캐시: outputs/cache/train_X.npz)"""
    p = os.path.join(CACHE, "train_X.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return z["X"], z["y"], list(z["names"])
    X, y, names = load_dataset(DATA)
    np.savez(p, X=X, y=y, names=np.array(names))
    return X, y, names


def cached_table(name, fn, X):
    """피처 표 캐시 (outputs/cache/feat_{name}.csv). inf·NaN은 0.
    피처 함수를 고쳤으면 해당 csv를 지우고 다시 돌린다"""
    p = os.path.join(CACHE, f"feat_{name}.csv")
    if os.path.exists(p):
        T = pd.read_csv(p)
    else:
        T = fn(X)
        T.to_csv(p, index=False)
    return T.replace([np.inf, -np.inf], np.nan).fillna(0)


def _f2_at(recall, fpr, p):
    """재현율·오경보율이 그대로일 때 고장 비율 p에서의 F2 (1-2 조각 4 공식 환산)"""
    tp, fp, fn = p * recall, (1 - p) * fpr, p * (1 - recall)
    return float(5 * tp / (5 * tp + 4 * fn + fp))


# ---------- 모델 ----------
def make_et(n=300, seed=0, n_jobs=1):
    """n_jobs=1이 기본: 파일 ~120개에서는 나무 단위 병렬이 오히려 느리다 (-1이 약 40% 느림).
    병렬은 nested_cv(n_jobs=...)에서 (seed, 바깥 fold) 단위로 한다"""
    return lambda: ExtraTreesClassifier(n, class_weight="balanced_subsample", min_samples_leaf=2,
                                        max_features="sqrt", random_state=seed, n_jobs=n_jobs)


class ColModel(BaseEstimator):
    """열 일부만 쓰는 모델"""
    def __init__(self, cols, make): self.cols, self.make = cols, make
    def fit(self, X, y): self.m_ = self.make().fit(X[:, self.cols], y); return self
    def predict_proba(self, X): return self.m_.predict_proba(X[:, self.cols])


def vif(F):
    """열별 VIF = 상관행렬 역행렬의 대각. 특이행렬 대비 아주 작은 릿지(1e-8)"""
    R = np.corrcoef(np.asarray(F, dtype=float), rowvar=False)
    return np.diag(np.linalg.inv(R + 1e-8 * np.eye(len(R))))


class CollinearityFilter(BaseEstimator):
    """다중공선성 제거 (라벨 안 씀, fit한 데이터의 피처만 봄 → 폴드 안에서 fit).
    1) 상수 열 제거 2) |r| > r_max 인 쌍에서 뒤쪽 열 제거 (열 순서 = 피처 id 순, core 13이 앞이라 먼저 남음)
    3) vif_max가 있으면 VIF가 가장 큰 열을 하나씩 빼서 모두 vif_max 이하가 될 때까지"""
    def __init__(self, r_max=0.9, vif_max=None): self.r_max, self.vif_max = r_max, vif_max
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        keep = [j for j in range(X.shape[1]) if X[:, j].std() > 0]
        R = np.abs(np.corrcoef(X[:, keep], rowvar=False))
        sel = []
        for a in range(len(keep)):
            if all(R[a, b] <= self.r_max for b in sel):
                sel.append(a)
        cols = [keep[a] for a in sel]
        if self.vif_max is not None:
            while len(cols) > 2:
                v = vif(X[:, cols])
                if v.max() <= self.vif_max:
                    break
                cols.pop(int(np.argmax(v)))
        self.cols_ = np.array(cols)
        return self
    def transform(self, X): return np.asarray(X, dtype=float)[:, self.cols_]


class RankAvg(BaseEstimator):
    """여러 모델의 고장 확률을 학습 데이터 점수 분포 기준 백분위로 바꿔 평균"""
    def __init__(self, makers): self.makers = makers
    def fit(self, X, y):
        self.ms_ = [m().fit(X, y) for m in self.makers]
        self.ref_ = [np.sort(m.predict_proba(X)[:, 1]) for m in self.ms_]
        return self
    def decision_function(self, X):
        s = [np.searchsorted(r, m.predict_proba(X)[:, 1], side="right") / len(r) for m, r in zip(self.ms_, self.ref_)]
        return np.mean(s, axis=0)


# ---------- 실험 기록 ----------
def _git_hash():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return h + ("+수정중" if dirty else "")
    except Exception:
        return "미확인"


class Experiment:
    """실험 폴더 하나 = 코드(run.py) + figs/ + result.json + README.md

    README 맨 위 '노션 카드' = 노션 실험 DB 속성 (팀원·모델·F2 점수·변경 사항·메모).
    F2 점수는 record(..., main=True)로 지정한 대표 구성(없으면 처음 기록한 구성).
    README의 '## 해석'(결과 표 아래)은 손으로 쓰는 곳이라 다시 실행해도 보존한다. 카드의 메모 칸은 그 첫 문장. 수치 메모는 result.json에만."""

    def __init__(self, run_file, exp_id, title, hypotheses=(), member="", model="", changes="",
                 focus=("abn_002", "abn_022", "abn_029")):
        self.dir = os.path.dirname(os.path.abspath(run_file))
        self.figs = os.path.join(self.dir, "figs"); os.makedirs(self.figs, exist_ok=True)
        self.id, self.title, self.hyp, self.focus = exp_id, title, list(hypotheses), list(focus)
        self.json = os.path.join(self.dir, "result.json")
        self.res = json.load(open(self.json, encoding="utf-8")) if os.path.exists(self.json) else {}
        self.res.update(id=exp_id, title=title, hypotheses=self.hyp, member=member, model=model,
                        changes=changes, git=_git_hash(),
                        date=datetime.date.today().isoformat())
        self.res.setdefault("configs", {}); self.res.setdefault("figs", {})
        self._write()

    def save_fig(self, fig, name, caption=""):
        """figs/{name}.png 저장. caption은 README에 '무엇을 보는 그림인지' 한 줄"""
        import matplotlib.pyplot as plt
        fig.savefig(os.path.join(self.figs, f"{name}.png"), dpi=80)
        self.res["figs"][name] = caption
        self._write()
        if "inline" not in plt.get_backend().lower():
            plt.close(fig)
        return fig

    def note(self, key, text):
        """result.json에 남길 짧은 수치 메모 (예: 피처 AUC 표). README에는 쓰지 않는다 — 필요한 수치는 해석에 옮겨 쓴다"""
        self.res.setdefault("notes", {})[key] = text
        self._write()

    def record(self, name, r, names, main=False):
        """nested_cv 결과 한 구성을 기록하고 한 줄 출력. main=True면 노션 카드의 F2 점수로 쓴다"""
        miss = [names[i] for i in np.flatnonzero(r["miss_counts"] >= 3)]
        row = dict(f2=round(float(r["f2"].mean()), 3), f2_sd=round(float(r["f2"].std()), 3),
                   recall=round(float(r["recall"].mean()), 3), precision=round(float(r["precision"].mean()), 3),
                   fpr=round(float(r["fpr"].mean()), 3), fpr_sd=round(float(r["fpr"].std()), 3),
                   f2_p10=round(_f2_at(r["recall"].mean(), r["fpr"].mean(), 0.1), 3),
                   f2_p40=round(_f2_at(r["recall"].mean(), r["fpr"].mean(), 0.4), 3),
                   missed=[m.replace("abn_", "") for m in miss], fp3=int((r["fp_counts"] >= 3).sum()),
                   miss_counts={names[i]: int(c) for i, c in enumerate(r["miss_counts"]) if c},
                   fp_counts={names[i]: int(c) for i, c in enumerate(r["fp_counts"]) if c})
        self.res["configs"][name] = row
        if main or "main" not in self.res:
            self.res["main"] = name
        self._write()
        print(f"{name:40s} F2 {row['f2']:.3f} ± {row['f2_sd']:.3f} | R {row['recall']:.3f} FPR {row['fpr']:.3f} P {row['precision']:.3f}"
              f" | 자주 놓침 {' '.join(row['missed'])} | 오경보(3회+) {row['fp3']}", flush=True)
        return row

    @staticmethod
    def _read_interp(txt):
        """README에서 손으로 쓴 해석 본문을 꺼낸다. 새 형식 = '## 해석' ~ '## 그림' 사이,
        옛 형식 = '## 메모' 뒤 전부 (안내문 '(결론·해석을 …)'은 버림)"""
        if "\n## 해석\n" in txt:
            body = txt.split("\n## 해석\n", 1)[1].split("\n## 그림", 1)[0]
        elif "## 메모" in txt:
            body = txt.split("## 메모", 1)[1]
        else:
            return ""
        lines = [l for l in body.splitlines() if not l.startswith("(결론·해석을")]
        return "\n".join(lines).strip("\n")

    def _write(self):
        """README = 노션 카드 → 결과 표 → 해석(손으로 씀, 다시 실행해도 유지) → 그림(psd_missed가 맨 끝).
        note()로 남긴 수치 메모는 result.json에만 두고 README에는 쓰지 않는다 (11-14 사용자 결정)"""
        json.dump(self.res, open(self.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        readme = os.path.join(self.dir, "README.md")
        interp = self._read_interp(open(readme, encoding="utf-8").read()) if os.path.exists(readme) else ""
        r = self.res
        notes = os.path.relpath(os.path.join(ROOT, "notes", "물리가설.md"), self.dir).replace("\\", "/")
        first = next((l.strip() for l in interp.splitlines() if l.strip() and not l.startswith(("#", "|", "-", "!"))), "")
        m = r["configs"].get(r.get("main"), {})
        f2 = f"{m['f2']:.3f} ± {m['f2_sd']:.3f}" if m else ""
        hyp = f" ({', '.join(r['hypotheses'])})" if r["hypotheses"] else ""
        s = [f"# {r['id']} {r['title']}\n",
             "노션 카드 (실험 DB에 그대로 옮긴다)\n",
             "| 팀원 | 모델 | F2 점수 | 변경 사항 | 메모 |", "|---|---|---|---|---|",
             f"| {r.get('member', '')} | {r.get('model', '')} | {f2} | {r.get('changes', '')}{hyp} | {first} |\n",
             f"- 가설: {', '.join(r['hypotheses']) or '—'} ([물리가설]({notes}))",
             f"- 실행: {r['date']} · git `{r['git']}` · 코드 [run.py](run.py)"]
        if r["configs"]:
            s += ["- 평가: 중첩 CV (바깥 5 × 안쪽 4, seed 5개). 자주 놓침 = seed 5번 중 3번 이상\n",
                  "## 결과\n", "| 구성 | F2 | 재현율 | 오경보율 | 정밀도 | F2 (고장 10%·40%) | 자주 놓침 | 오경보 3회+ |", "|---|---|---|---|---|---|---|---|"]
        for k, v in r["configs"].items():
            s.append(f"| {k.replace('|', chr(92) + '|')} | {v['f2']:.3f} ± {v['f2_sd']:.3f} | {v['recall']:.3f} | {v.get('fpr', float('nan')):.3f} | {v['precision']:.3f} "
                     f"| {v.get('f2_p10', float('nan')):.3f} · {v.get('f2_p40', float('nan')):.3f} "
                     f"| {' '.join(v['missed']) or '—'} | {v['fp3']} |")
        if interp:
            s.append("\n## 해석\n\n" + interp + "\n")
        s.append("\n## 그림\n")
        figs = sorted(r["figs"].items(), key=lambda kv: kv[0].startswith("psd"))   # psd 그림은 맨 끝
        for k, cap in figs:
            s.append(f"### {k}\n{cap}\n\n![{k}](figs/{k}.png)\n")
        open(readme, "w", encoding="utf-8").write("\n".join(s))
