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
from src.evaluate import nested_cv, MODE, ET_N, N_SEEDS
from src import viz
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier

CACHE = os.path.join(ROOT, "outputs", "cache")
DATA = os.path.join(ROOT, "data", "TeamProject_train_변경")
os.makedirs(CACHE, exist_ok=True)

__all__ = ["ROOT", "CACHE", "np", "pd", "viz", "nested_cv", "load_train", "cached_table",
           "feature_table", "feature_table_ext", "feature_table_phys",
           "make_et", "ColModel", "RankAvg", "CollinearityFilter", "vif", "Experiment", "md_table", "error_md_table", "OUTLIER_NORMALS", "outlier_idx"]


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


KEY_OPEN, KEY_CLOSE = "<!-- key-figs: 자동 (save_fig key=True) -->", "<!-- /key-figs -->"


class Experiment:
    """실험 폴더 하나 = 코드(run.py) + figs/ + result.json + README.md

    README 맨 위 '노션 카드' = 노션 실험 DB 속성 (팀원·모델·F2 점수·변경 사항·메모).
    F2 점수는 record(..., main=True)로 지정한 대표 구성(없으면 처음 기록한 구성).
    README의 '## 해석'(결과 표 아래)은 손으로 쓰는 곳이라 다시 실행해도 보존한다. 카드의 메모 칸은 그 첫 문장. 수치 메모는 result.json에만.

    README 순서 (11-14 사용자 결정): 카드 → 결과 표 → 해석 첫 문단(결론 요약) → **핵심 그림**(save_fig(..., key=True))
    → 그 아래 손으로 쓴 설명(그림 읽는 법·상세) → '## 그림'(나머지 그림). 핵심 그림은 해석 첫 문단 바로 뒤에
    자동으로 끼워 넣고(<!-- key-figs --> 표시), 다시 읽을 때는 빼서 손으로 쓴 해석만 보존한다"""

    def __init__(self, run_file, exp_id, title, hypotheses=(), member="", model="", changes="",
                 focus=("abn_002", "abn_022", "abn_029"), key_changes=()):
        self.dir = os.path.dirname(os.path.abspath(run_file))
        self.figs = os.path.join(self.dir, "figs"); os.makedirs(self.figs, exist_ok=True)
        self.id, self.title, self.hyp, self.focus = exp_id, title, list(hypotheses), list(focus)
        self.json = os.path.join(self.dir, "result.json")
        self.res = json.load(open(self.json, encoding="utf-8")) if os.path.exists(self.json) else {}
        self.res.update(id=exp_id, title=title, hypotheses=self.hyp, member=member, model=model,
                        changes=changes, key_changes=list(key_changes), git=_git_hash(),
                        date=datetime.date.today().isoformat())
        self.res.setdefault("configs", {}); self.res.setdefault("figs", {})
        self._write()

    def save_fig(self, fig, name, caption="", key=False):
        """figs/{name}.png 저장. caption은 README에 '무엇을 보는 그림인지' 한 줄.
        key=True = 핵심 그림: README 해석 첫 문단 바로 아래에 놓는다 (결과 해석 + 그림을 먼저, 설명은 그 아래)"""
        import matplotlib.pyplot as plt
        fig.savefig(os.path.join(self.figs, f"{name}.png"), dpi=80)
        self.res["figs"][name] = caption
        keys = self.res.setdefault("key_figs", [])
        if key and name not in keys:
            keys.append(name)
        self._write()
        if "inline" not in plt.get_backend().lower():
            plt.close(fig)
        return fig

    def save_md(self, name, text):
        """실험 폴더에 {name}.md 저장 (오류 분석 표 · val 판정 등 표 문서). csv · 별도 json은 만들지 않는다 (11-22 사용자)"""
        p = os.path.join(self.dir, f"{name}.md"); open(p, "w", encoding="utf-8").write(text); return p

    def note(self, key, text):
        """result.json에 남길 짧은 수치 메모 (예: 피처 AUC 표). README에는 쓰지 않는다 — 필요한 수치는 해석에 옮겨 쓴다"""
        self.res.setdefault("notes", {})[key] = text
        self._write()

    def record(self, name, r, names, main=False, mid=False):
        """nested_cv 결과 한 구성을 기록하고 한 줄 출력. main=True면 노션 카드의 F2 점수로 쓴다"""
        ns_ = len(r["f2"]); k = ns_ // 2 + 1                                # 자주 = seed 과반 (5 → 3, 2 → 2)
        miss = [names[i] for i in np.flatnonzero(r["miss_counts"] >= k)]
        row = dict(f2=round(float(r["f2"].mean()), 3), f2_sd=round(float(r["f2"].std()), 3),
                   recall=round(float(r["recall"].mean()), 3), precision=round(float(r["precision"].mean()), 3),
                   mid=bool(mid), fpr=round(float(r["fpr"].mean()), 3), fpr_sd=round(float(r["fpr"].std()), 3),
                   f2_p10=round(_f2_at(r["recall"].mean(), r["fpr"].mean(), 0.1), 3),
                   f2_p40=round(_f2_at(r["recall"].mean(), r["fpr"].mean(), 0.4), 3),
                   missed=[m.replace("abn_", "") for m in miss], fp3=int((r["fp_counts"] >= k).sum()), n_seeds=ns_, mode=f"seed {ns_} · ET {ET_N}",
                   miss_counts={names[i]: int(c) for i, c in enumerate(r["miss_counts"]) if c},
                   fp_counts={names[i]: int(c) for i, c in enumerate(r["fp_counts"]) if c})
        self.res["configs"][name] = row
        if main or "main" not in self.res:
            self.res["main"] = name
        self._write()
        print(f"{name:40s} F2 {row['f2']:.3f} ± {row['f2_sd']:.3f} | R {row['recall']:.3f} FPR {row['fpr']:.3f} P {row['precision']:.3f}"
              f" | 자주 놓침 {' '.join(row['missed'])} | 오경보({k}회+) {row['fp3']} | {row['mode']}", flush=True)
        return row

    @staticmethod
    def _read_interp(txt):
        """README에서 손으로 쓴 해석 본문을 꺼낸다. 새 형식 = '## 해석' ~ '## 그림' 사이,
        옛 형식 = '## 메모' 뒤 전부 (안내문 '(결론·해석을 …)'은 버림)"""
        if "\n## 해석\n" in txt:
            body = txt.split("\n## 해석\n", 1)[1].split("\n## 그림", 1)[0]
            while KEY_OPEN in body and KEY_CLOSE in body:                     # 자동으로 넣은 핵심 그림은 뺀다
                a, rest = body.split(KEY_OPEN, 1)
                body = a.rstrip("\n") + "\n\n" + rest.split(KEY_CLOSE, 1)[1].lstrip("\n")
        elif "## 메모" in txt:
            body = txt.split("## 메모", 1)[1]
        else:
            return ""
        lines = [l for l in body.splitlines() if not l.startswith("(결론·해석을")]
        return "\n".join(lines).strip("\n")

    @staticmethod
    def _with_key_figs(interp, block):
        """해석 첫 문단(결론 요약) 바로 뒤에 핵심 그림 블록을 끼운다. 해석이 비어 있으면 블록만"""
        if not block:
            return interp
        if not interp:
            return block
        lines = interp.splitlines()
        i = next((k for k, l in enumerate(lines) if l.strip()), 0)          # 첫 문단 시작
        while i < len(lines) and lines[i].strip():                          # 첫 문단 끝 (빈 줄 전)
            i += 1
        return "\n".join(lines[:i] + ["", block, ""] + lines[i:]).replace("\n\n\n", "\n\n")

    def _write(self):
        """README = 노션 카드 → 결과 표 → 해석 첫 문단 → 핵심 그림(key=True) → 손으로 쓴 설명 → 그림(나머지, psd가 맨 끝).
        note()로 남긴 수치 메모는 result.json에만 두고 README에는 쓰지 않는다 (11-14 사용자 결정)"""
        json.dump(self.res, open(self.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        readme = os.path.join(self.dir, "README.md")
        interp = self._read_interp(open(readme, encoding="utf-8").read()) if os.path.exists(readme) else ""
        r = self.res
        notes = os.path.relpath(os.path.join(ROOT, "단계", "1_기준·기록", "물리가설.md"), self.dir).replace("\\", "/")
        first = next((l.strip() for l in interp.splitlines() if l.strip() and not l.startswith(("#", "|", "-", "!"))), "")
        m = r["configs"].get(r.get("main"), {})
        f2 = f"{m['f2']:.3f} ± {m['f2_sd']:.3f}" if m else ""
        hyp = f" ({', '.join(r['hypotheses'])})" if r["hypotheses"] else ""
        s = [f"# {r['id']} {r['title']}\n",
             "노션 카드 (실험 DB에 그대로 옮긴다)\n",
             "| 팀원 | 모델 | F2 점수 | 변경 사항 | 메모 |", "|---|---|---|---|---|",
             f"| {r.get('member', '')} | {r.get('model', '')} | {f2} | {r.get('changes', '')}{hyp} | {first} |\n"]
        if r.get("key_changes"):
            s += ["## 주요 변경점 (기준 실험 대비)\n"] + [f"- {c}" for c in r["key_changes"]] + [""]
        s += [f"- 가설: {', '.join(r['hypotheses']) or '—'} ([물리가설]({notes}))",
             f"- 실행: {r['date']} · git `{r['git']}` · 코드 [run.py](run.py)"]
        if r["configs"]:
            modes = sorted({v.get("mode", "seed 5 · ET 300") for v in r["configs"].values()})
            s += [f"- 평가: 중첩 CV (바깥 5 × 안쪽 4) · 모드 {' / '.join(modes)}. 자주 놓침 · 오경보 n회+ = seed 과반. quick 수치는 선별용, 기록 수치는 QE_FULL=1\n",
                  "## 결과\n", "| 구성 | F2 | 재현율 | 오경보율 | 정밀도 | F2 (고장 10%·40%) | 자주 놓침 | 오경보 3회+ |", "|---|---|---|---|---|---|---|---|"]
        def _row(k, v):
            return (f"| {k.replace('|', chr(92) + '|')} | {v['f2']:.3f} ± {v['f2_sd']:.3f} | {v['recall']:.3f} | {v.get('fpr', float('nan')):.3f} | {v['precision']:.3f} "
                    f"| {v.get('f2_p10', float('nan')):.3f} · {v.get('f2_p40', float('nan')):.3f} "
                    f"| {' '.join(v['missed']) or '—'} | {v['fp3']} |")
        mid = [(k, v) for k, v in r["configs"].items() if v.get("mid")]          # 중간 단계 결과(비교 · 절제 · 이전 값)는 README 맨 아래로 (11-22 사용자)
        for k, v in r["configs"].items():
            if not v.get("mid"):
                s.append(_row(k, v))
        keys = [k for k in r.get("key_figs", []) if k in r["figs"]]
        block = "\n".join([KEY_OPEN] + [f"![{k}](figs/{k}.png)\n\n*{k}* — {r['figs'][k]}\n" for k in keys] + [KEY_CLOSE]) if keys else ""
        body = self._with_key_figs(interp, block)
        if body:
            s.append("\n## 해석\n\n" + body + "\n")
        figs = sorted([kv for kv in r["figs"].items() if kv[0] not in keys], key=lambda kv: kv[0].startswith("psd"))   # psd 그림은 맨 끝
        s.append("\n## 그림\n")
        for k, cap in figs:
            s.append(f"### {k}\n{cap}\n\n![{k}](figs/{k}.png)\n")
        if mid:
            s += ["\n## 중간 단계 결과 (비교 · 절제 · 이전 실험 값)\n", "| 구성 | F2 | 재현율 | 오경보율 | 정밀도 | F2 (고장 10%·40%) | 자주 놓침 | 오경보 3회+ |", "|---|---|---|---|---|---|---|---|"]
            s += [_row(k, v) for k, v in mid]
        open(readme, "w", encoding="utf-8").write("\n".join(s))


def md_table(df):
    """DataFrame → 마크다운 표 문자열 (tabulate 없이)"""
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v).replace("|", chr(92) + "|") for v in r) + " |")
    return chr(10).join(rows)


def error_md_table(names, y, pos_counts, files, n_seeds=None, extra=None):
    """오류 분석 표 (11-22 사용자 형식, 가로 배치): 열 = 파일, 행 = 우리 판단(정답 라벨) · 모델 판단(seed n번 중 고장/정상 횟수, n = 모드) · 결과 · 모델 근거.
    pos_counts[i] = 파일 i를 고장으로 판정한 횟수 (nested_cv의 miss_counts · fp_counts에서: 고장 = n − miss, 정상 = fp)
    extra = {행 이름: {파일: 값}} 모델 근거(ET · AE · 규칙 켜짐 등). 문자열 표를 돌려준다"""
    n_seeds = N_SEEDS if n_seeds is None else n_seeds
    cols = {}
    for f in files:
        i = list(names).index(f); k = int(pos_counts[i]); fault = bool(y[i]); ok = (k > n_seeds / 2) == fault
        c = {"우리 판단 (정답 라벨)": "고장" if fault else "정상", f"모델 판단 (seed {n_seeds}번 중)": f"고장 {k} · 정상 {n_seeds - k}",
             "결과": ("맞음" if ok else "틀림") + f" ({k if fault else n_seeds - k}/{n_seeds} 정답)"}
        for row, d in (extra or {}).items():
            c[row] = d.get(f, "—")
        cols[f] = c
    rows = list(next(iter(cols.values())).keys())
    out = ["| | " + " | ".join(files) + " |", "|---|" + "---|" * len(files)]
    out += [f"| {r} | " + " | ".join(str(cols[f][r]) for f in files) + " |" for r in rows]
    return chr(10).join(out)


# 11-22 사용자 결정: 데이터로 구분할 수 없는 정상 이상치. 학습에서는 빼고, 평가 · 결과 계산에는 넣고, 그림에는 따로 표시한다
OUTLIER_NORMALS = ("nor_059", "nor_065", "nor_116")


def outlier_idx(names):
    """OUTLIER_NORMALS의 행 번호 (names 순서)"""
    return [list(names).index(n) for n in OUTLIER_NORMALS]
