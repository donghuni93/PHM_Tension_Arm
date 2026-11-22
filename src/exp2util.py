"""EXP-041 · EXP-042 공용: 데이터·행렬 준비, 추정기 팩토리, OOF 점수·전문가 켜짐 횟수, 그림(평면·t-SNE·hard_files), 오류 분석 표, val 채점"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedGroupKFold

from src.data import load_folder
from src.evaluate import best_f2_threshold
from src.exp import ROOT, cached_table, feature_table_ext, load_train
from src.features2 import PeakRef, H_FOLD, cached_feat2, robust_z
from src.models2 import Two

NOR, ABN, PUR = "#2a78d6", "#eb6834", "#7B1FA2"
EXP040 = os.path.join(ROOT, "단계", "5_모델비교", "base", "EXP-040_1121_2차설계_1단계")
short = lambda s: s.replace("abn_", "A").replace("nor_", "N")


class NS:
    """데이터 · 피처 행렬 묶음 (EXP-040 run.py와 같은 M·Mv)"""
    def __init__(self):
        self.X, self.y, names = load_train(); self.names = [str(n) for n in names]
        vd = os.path.join(ROOT, "data", "TeamProject_val", "val"); vd = vd if os.path.isdir(vd) else os.path.dirname(vd)   # PC = val/ 하위 폴더, 클라우드 = 바로 아래
        self.Xv, nv = load_folder(vd); self.nv = [str(n) for n in nv]
        FJ = json.load(open(os.path.join(ROOT, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))
        self.FIXED, self.PEAK = list(FJ["fixed"]), list(FJ["hfold"]); assert FJ["lock"] == ["L1_S"]
        self.T, self.L, fr = cached_feat2("train", self.X); self.Tv, self.Lv, _ = cached_feat2("val", self.Xv)
        self.df = float(fr[1] - fr[0]); self.nf = len(self.FIXED)
        self.M = np.hstack([self.T[self.FIXED].values, self.T[["L1_S"]].values, self.L])
        self.Mv = np.hstack([self.Tv[self.FIXED].values, self.Tv[["L1_S"]].values, self.Lv])
        self.n = len(self.y); self.g = np.arange(self.n); self.nor = self.y == 0
        self.cols = self.FIXED + self.PEAK + ["L1_S"]

    def mk(self, comb, experts=(), scale_thr=1.0):
        return lambda: Two(comb, experts, fixed=tuple(self.FIXED), nf=self.nf, df=self.df, peak=tuple(self.PEAK), scale_thr=scale_thr)


def oof_fires(ns, mk_, seed=0):
    """OOF 5-fold: ET logit · 박자 z · 결합 점수 · 블록별 켜짐 (한 seed)"""
    n = ns.n; out = dict(a=np.empty(n), z=np.empty(n), score=np.empty(n), main=np.zeros(n, bool), E1=np.zeros(n, bool), E2=np.zeros(n, bool))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(ns.g, ns.y, ns.g):
        f = mk_().fit(ns.M[tr], ns.y[tr]).fires(ns.M[te])
        for k in out:
            if k in f: out[k][te] = f[k]
    return out


def wrong_of(ns, row, k=2):
    return {ns.names.index(f): v for f, v in {**row["miss_counts"], **row["fp_counts"]}.items() if v >= k}


def expert_counts(ns, which, scales=(1.0,), seeds=range(5)):
    """전문가 E1/E2 단독 켜짐을 중첩 CV와 같은 분할(seed 5 × 바깥 5겹)에서: 학습 폴드 정상 99 % 문턱을 테스트 폴드에 적용.
    반환 {scale: 파일별 켜진 횟수(seed 5번 중)} — scale = 문턱 거리(분위수 − 정상 중앙값)의 배수 (C3: 0.8·1.2)"""
    h18 = ns.T["H18_hfl"].values; h1 = ns.T["H1_nimp"].values
    out = {s: np.zeros(ns.n, int) for s in scales}
    for seed in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(ns.g, ns.y, ns.g):
            trn = tr[ns.y[tr] == 0]
            if which == "E1":
                pr = PeakRef(ns.df).fit(ns.L[trn]); v_tr = pr.transform(ns.L[trn])[:, 0]; v_te = pr.transform(ns.L[te])[:, 0]
            else:
                v_tr, v_te = h18[trn], h18[te]
            q, med = np.percentile(v_tr, 99), np.median(v_tr)
            for s in scales:
                on = v_te > med + (q - med) * s
                if which == "E2": on &= h1[te] >= np.log(2) - 1e-9
                out[s][te] += on
    return out


def plane_A(ax, ns, A, row, title):
    """EXP-034 방식 AE z × ET z (규칙으로 잡힌 고장은 뺌)"""
    Z, rule = A.oof_plane(); y, nor, names = ns.y, ns.nor, ns.names
    Zc = np.clip(Z, -3, 4.5)
    xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 300), np.linspace(-3, 4.5, 300))
    D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8); ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5)
    ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    show = ~(rule & (y == 1))
    ax.scatter(Zc[nor & show, 0], Zc[nor & show, 1], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상")
    ax.scatter(Zc[~nor & show, 0], Zc[~nor & show, 1], color=ABN, s=26, lw=0, alpha=.9, label="고장 (규칙 밖)")
    if (rule & nor).any(): ax.scatter(Zc[rule & nor, 0], Zc[rule & nor, 1], s=110, marker="D", facecolor="none", edgecolor="#7a3db8", lw=1.3, label="규칙이 켜진 정상")
    w = wrong_of(ns, row); wi = list(w)
    ax.scatter(Zc[wi, 0], Zc[wi, 1], s=80, marker="x", color="k", lw=1.6, label="자주 틀림 (2/5+)")
    for i in wi: ax.annotate(f"{short(names[i])} {'미탐' if y[i] else '오경보'} {w[i]}/5", Zc[i], xytext=(4, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR, weight="bold")
    gone = [short(names[i]) for i in np.flatnonzero(rule & (y == 1))]
    ax.text(.98, .02, f"규칙이 잡아 뺀 고장 {len(gone)}개:\n" + " ".join(gone), transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (ext150 + 울림 2열)")
    ax.set_title(f"{title}\n중첩 F2 {row['f2']:.3f} ± {row['f2_sd']:.3f} · 재현율 {row['recall']:.3f} · 오경보율 {row['fpr']:.3f}", fontsize=9.5); ax.legend(fontsize=8, loc="upper left")


def plane_W(ax, ns, mk_, fo, row, title):
    """ET 점수(OOF logit) × 박자 z 평면, 결합 경계(0 실선·±1 점선; max는 두 문턱선), 전문가가 켠 파일 표시. fo = oof_fires 결과"""
    y, nor, names = ns.y, ns.nor, ns.names
    m = mk_().fit(ns.M, ns.y); a, z = fo["a"], fo["z"]
    xc, yc = np.clip(a, -7, 7), np.clip(z, -3, 20)
    xx, yy_ = np.meshgrid(np.linspace(-7, 7, 300), np.linspace(-3, 20, 300))
    D = m._comb_score(xx.ravel(), yy_.ravel()).reshape(xx.shape)
    if m.comb == "max":
        ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8); ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5)
    else:
        ax.contourf(xx, yy_, D, levels=[-1e9, m.tc_ if hasattr(m, "tc_") and m.experts else 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
        ax.contour(xx, yy_, D, levels=[m.tc_ if hasattr(m, "tc_") and m.experts else 0], colors="k", linewidths=1.5)
        ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    ax.scatter(xc[nor], yc[nor], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상"); ax.scatter(xc[~nor], yc[~nor], color=ABN, s=28, lw=0, alpha=.9, label="고장")
    for e, mk, col in (("E1", "D", "#2ca02c"), ("E2", "P", "#7a3db8")):
        if e in fo and fo[e].any() and e in m.experts:
            ax.scatter(xc[fo[e]], yc[fo[e]], s=120, marker=mk, facecolor="none", edgecolor=col, lw=1.4, label=f"{e}가 켠 파일 (OOF seed 0)")
    w = wrong_of(ns, row); wi = list(w)
    ax.scatter(xc[wi], yc[wi], s=80, marker="x", color="k", lw=1.6, label="자주 틀림 (2/5+)")
    for i in wi: ax.annotate(f"{short(names[i])} {'미탐' if y[i] else '오경보'} {w[i]}/5", (xc[i], yc[i]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR, weight="bold")
    for nm_ in ("abn_022", "abn_014"):
        i = names.index(nm_); ax.annotate(short(nm_), (xc[i], yc[i]), xytext=(-18, 6), textcoords="offset points", fontsize=8, color=ABN)
    ax.set_xlabel("H 점수 (ET logit, 폴드 밖 OOF)"); ax.set_ylabel("박자 z (학습 정상 S 강건 z, 20으로 자름)"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{title}\n중첩 F2 {row['f2']:.3f} ± {row['f2_sd']:.3f} · 재현율 {row['recall']:.3f} · 오경보율 {row['fpr']:.3f}", fontsize=9.5)


def hard_fig(ns, ex, labs, bad):
    HARD = bad + [h for h in ("abn_022", "abn_014", "abn_017", "abn_002", "abn_015", "nor_116") if h not in bad]
    Mh = np.array([[ex.res["configs"]["중첩: " + c]["miss_counts" if h.startswith("abn") else "fp_counts"].get(h, 0) for c in labs] for h in HARD])
    fig, ax = plt.subplots(figsize=(1.4 * len(labs) + 4, 0.42 * len(HARD) + 2))
    im = ax.imshow(Mh, cmap="Reds", vmin=0, vmax=5, aspect="auto")
    for i in range(Mh.shape[0]):
        for j in range(Mh.shape[1]): ax.text(j, i, Mh[i, j], ha="center", va="center", fontsize=9, color="w" if Mh[i, j] >= 3 else "k")
    ax.set_yticks(range(len(HARD)), [("고장 " if h[0] == "a" else "정상 ") + h for h in HARD], fontsize=8); ax.set_xticks(range(len(labs)), [c.split(" | ")[0] for c in labs])
    ax.set_title("파일별 틀린 횟수 (seed 5번 중). 고장 = 미탐, 정상 = 오경보", loc="left", fontsize=10); fig.colorbar(im, shrink=.6); fig.tight_layout()
    return fig


def tsne_fig(ns, score_all, mark, title_score):
    def stdz(Atr, Ava):
        Z, Zv = np.empty(Atr.shape), np.empty(Ava.shape)
        for j in range(Atr.shape[1]):
            f = robust_z(Atr[ns.nor, j]); Z[:, j] = f(Atr[:, j]); Zv[:, j] = f(Ava[:, j])
        return Z, Zv
    pk = PeakRef(ns.df).fit(ns.L[ns.nor]); Tn, Tnv = ns.T.copy(), ns.Tv.copy()
    p1, p2 = pk.transform(ns.L), pk.transform(ns.Lv)
    Tn["H16_peak_h"], Tn["H17_peak_w"] = p1[:, 0], p1[:, 1]; Tnv["H16_peak_h"], Tnv["H17_peak_w"] = p2[:, 0], p2[:, 1]
    E = cached_table("ext", feature_table_ext, ns.X); Ev = cached_table("ext_val", feature_table_ext, ns.Xv).reindex(columns=E.columns, fill_value=0.0)
    views = {f"새 피처 {len(ns.cols)}열 (H + L1 S)": stdz(Tn[ns.cols].values, Tnv[ns.cols].values), "ext150 (A의 ET 입력)": stdz(E.values, Ev.values)}
    fig, axs = plt.subplots(2, 2, figsize=(14, 12.4)); n = ns.n
    for j, (title, (Z, Zv)) in enumerate(views.items()):
        best = None
        for s in range(5):
            t = TSNE(2, perplexity=30, init="random", random_state=s, learning_rate="auto"); Em = t.fit_transform(np.vstack([Z, Zv]))
            if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, Em)
        Em = best[1]; Et, Ev_ = Em[:n], Em[n:]
        a_ = axs[0, j]; a_.scatter(*Et[ns.nor].T, s=20, color=NOR, alpha=.55, lw=0, label="train 정상"); a_.scatter(*Et[~ns.nor].T, s=36, color=ABN, lw=0, label="train 고장")
        a_.scatter(*Ev_.T, s=45, color=PUR, marker="^", edgecolor="white", lw=.4, label="val")
        for h in mark:
            i = ns.names.index(h); a_.annotate(short(h), Et[i], xytext=(3, 3), textcoords="offset points", fontsize=7, color=ABN if ns.y[i] else NOR)
        a_.set_title(f"{title} · 라벨 색 (KL 최소 {best[0]:.2f})", fontsize=10); a_.legend(fontsize=8)
        a_ = axs[1, j]; sc = a_.scatter(*Em.T, c=score_all, cmap="viridis", s=22, lw=0); fig.colorbar(sc, ax=a_, shrink=.8, label=title_score)
        a_.scatter(*Et[~ns.nor].T, s=55, facecolor="none", edgecolor=ABN, lw=1); a_.set_title(f"{title} · 색 = {title_score} (주황 테두리 = 고장)", fontsize=10)
    for a_ in axs.ravel(): a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("t-SNE (perplexity 30, 강건 표준화, train 150 + val 40 함께). 모양은 perplexity에 따라 바뀌므로 4-7의 5-NN 숫자를 함께 본다", x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96)); return fig


def error_table(ns, files, imp, topk=6):
    """파일별: 피처 학습 정상 백분위(train 정상 120 기준, H16·H17은 정상 전체 참고 ref), 95 %를 넘은 피처, ET 중요도 상위 겹침"""
    pk = PeakRef(ns.df).fit(ns.L[ns.nor]); p = pk.transform(ns.L)
    V = ns.T[ns.cols[:ns.nf]].copy(); V["H16_peak_h"], V["H17_peak_w"] = p[:, 0], p[:, 1]; V["L1_S"] = ns.T["L1_S"].values
    V = V[ns.cols]
    order = np.argsort(imp)[::-1]
    top = [ns.cols[i] for i in order[:topk]]
    rows = []
    for f in files:
        i = ns.names.index(f)
        pct = {c: float((V[c].values[ns.nor] <= V[c].values[i]).mean() * 100) for c in ns.cols}
        hi = [c for c in ns.cols if pct[c] > 95]; lo = [c for c in ns.cols if pct[c] < 5]
        rows.append(dict(file=f, label="고장" if ns.y[i] else "정상", over95=" ".join(c.split("_")[0] for c in hi), under5=" ".join(c.split("_")[0] for c in lo),
                         top_overlap=" ".join(c.split("_")[0] for c in top if c in hi + lo), **{c.split("_")[0]: round(pct[c]) for c in ns.cols}))
    return pd.DataFrame(rows), top


def val_table(ns, model, extra=None):
    f = model.fires(ns.Mv)
    D = pd.DataFrame(dict(file=ns.nv, ET_logit=f["a"].round(2), 박자z=f["z"].round(2), 결합점수=np.asarray(f["score"]).round(3), main=f["main"].astype(int)))
    for e in ("E1", "E2"):
        if e in f: D[e] = f[e].astype(int)
    D["pred"] = (model.decision_function(ns.Mv) >= (0.5 if (model.comb == "max" or model.experts) else getattr(model, "thr_", 0))).astype(int)
    return D


# ---------------------------------------------------------------------------
# EXP-043: 8.3 kHz 규칙(src/final.py와 같은 정의, 학습 없음)을 Two 위에 OR로 붙임
# ---------------------------------------------------------------------------
def rule83_flags(X):
    """신호 배열 → (스펙트럼 규칙, 공진 규칙) bool. 돌출·폭 계산은 final.signal_tables의 P83·W83 부분과 같은 식, 공진은 final.ring_table"""
    from scipy.ndimage import median_filter, uniform_filter1d
    from scipy.signal import welch
    from src.final import FS, P83_MIN, RES_FIT_MIN, W83_MIN, ring_table
    X = np.asarray(X, float); Xc = X - X.mean(1, keepdims=True)
    f, P8 = welch(Xc, fs=FS, nperseg=8192); L = np.log10(P8); df = f[1] - f[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)
    PR = Sm - median_filter(Sm, size=(1, int(round(1200 / df))), mode="nearest")
    band = (f >= 8200) & (f < 8450); ia = np.flatnonzero(band)
    P83, W83 = PR[:, band].max(1), np.zeros(len(PR))
    for i in range(len(PR)):
        k = ia[PR[i, band].argmax()]; h = PR[i, k] / 2; l = r = k
        while l > 0 and PR[i, l - 1] >= h: l -= 1
        while r < PR.shape[1] - 1 and PR[i, r + 1] >= h: r += 1
        W83[i] = f[r] - f[l]
    _, RES, _ = ring_table(X)
    return (P83 >= P83_MIN) & (W83 >= W83_MIN), RES >= RES_FIT_MIN


class Add83:
    """입력 행의 마지막 열 = 8.3k 규칙 켜짐(0/1). 나머지 열로 mk()의 모델(Two)을 학습, 판정 = 그 판정 OR 규칙 (이진). Two의 속성은 그대로 넘긴다"""
    def __init__(self, mk): self.mk = mk
    def get_params(self, deep=True): return dict(mk=self.mk)
    def set_params(self, **p): self.mk = p.get("mk", self.mk); return self
    def fit(self, F, y): self.m_ = self.mk().fit(np.asarray(F)[:, :-1], y); return self
    def __getattr__(self, k):
        if k in ("m_", "mk") or k.startswith("__"): raise AttributeError(k)
        return getattr(self.m_, k)
    def fires(self, F):
        F = np.asarray(F); out = self.m_.fires(F[:, :-1]); out["R83"] = F[:, -1] > 0.5; return out
    def decision_function(self, F):
        F = np.asarray(F); return np.maximum(self.m_.decision_function(F[:, :-1]), F[:, -1])
