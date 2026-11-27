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
from src.exp import ROOT, cached_table, feature_table_ext, load_train, find_exp
from src.features2 import PeakRef, H_FOLD, cached_feat2, robust_z, new_name, RENAMED
from src.models2 import Two

NOR, ABN, PUR = "#2a78d6", "#eb6834", "#7B1FA2"
EXP040 = find_exp("EXP-040")
short = lambda s: s.replace("abn_", "A").replace("nor_", "N")


class NS:
    """데이터 · 피처 행렬 묶음 (EXP-040 run.py와 같은 M·Mv)"""
    def __init__(self):
        self.X, self.y, names = load_train(); self.names = [str(n) for n in names]
        vd = os.path.join(ROOT, "2_data", "TeamProject_val", "val"); vd = vd if os.path.isdir(vd) else os.path.dirname(vd)   # PC = val/ 하위 폴더, 클라우드 = 바로 아래
        self.Xv, nv = load_folder(vd); self.nv = [str(n) for n in nv]
        FJ = json.load(open(os.path.join(ROOT, "3_단계", "4_피처", "4-2_H피처목록", "자료", "최종피처.json"), encoding="utf-8"))
        self.FIXED, self.PEAK = [new_name(c) for c in FJ["fixed"]], list(FJ["hfold"]); assert [new_name(c) for c in FJ["lock"]] == ["L1_lock"]   # json은 옛 이름 (H18_hfl · L1_S)
        self.T, self.L, fr = cached_feat2("train", self.X); self.Tv, self.Lv, _ = cached_feat2("val", self.Xv)
        self.df = float(fr[1] - fr[0]); self.nf = len(self.FIXED)
        self.M = np.hstack([self.T[self.FIXED].values, self.T[["L1_lock"]].values, self.L])
        self.Mv = np.hstack([self.Tv[self.FIXED].values, self.Tv[["L1_lock"]].values, self.Lv])
        self.n = len(self.y); self.g = np.arange(self.n); self.nor = self.y == 0
        self.cols = self.FIXED + self.PEAK + ["L1_lock"]
        for o, nw in RENAMED.items():                                      # 옛 이름 별칭 (11-25 이름 변경 전 스크립트가 ns.T["L1_S"] 등으로 읽음)
            self.T[o], self.Tv[o] = self.T[nw], self.Tv[nw]

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
    h18 = ns.T["L2_hfl"].values; h1 = ns.T["H1_nimp"].values
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
    V = ns.T[ns.cols[:ns.nf]].copy(); V["H16_peak_h"], V["H17_peak_w"] = p[:, 0], p[:, 1]; V["L1_lock"] = ns.T["L1_lock"].values
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
def rule83_flags(X, k_gate=False):
    """신호 배열 → (스펙트럼 규칙, 공진 규칙) bool. 돌출·폭 계산은 final.signal_tables의 P83·W83 부분과 같은 식, 공진은 final.ring_table
    k_gate=True면 스펙트럼 규칙에 K83 > K83_MIN(충격성, 11-23)을 AND. 기본 False = EXP-050까지 기록된 실험을 그대로 재현. 새 실험(EXP-054~)은 True"""
    from scipy.ndimage import median_filter, uniform_filter1d
    from scipy.signal import welch
    from src.final import FS, K83_MIN, P83_MIN, RES_FIT_MIN, W83_MIN, kurt83, ring_table
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
    spec = (P83 >= P83_MIN) & (W83 >= W83_MIN)
    if k_gate:
        spec = spec & (kurt83(Xc) > K83_MIN)
    return spec, RES >= RES_FIT_MIN


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


class ExcludeTrain:
    """학습할 때만 drop 번호의 행을 뺀다 (평가 · 판정에는 그대로 쓴다). 입력 행의 마지막 열 = 행 번호(파일 번호). 11-22 이상치 정상 3개(src.exp.OUTLIER_NORMALS) 처리용"""
    def __init__(self, mk, drop): self.mk, self.drop = mk, tuple(drop)
    def fit(self, F, y):
        F = np.asarray(F); keep = ~np.isin(F[:, -1].astype(int), self.drop)
        self.m_ = self.mk().fit(F[keep], np.asarray(y)[keep]); return self
    def __getattr__(self, k):
        if k in ("m_", "mk", "drop") or k.startswith("__"): raise AttributeError(k)
        return getattr(self.m_, k)
    def fires(self, F): return self.m_.fires(F)
    def decision_function(self, F): return self.m_.decision_function(F)


# ---------------------------------------------------------------------------
# 11-22 4-10: t-SNE · 5-NN용 공간 (라벨 없음). robust_z는 정상 값의 절반 이상이 같은 값이면(H9: 정상 60/120 = 1.0)
# MAD가 0 근처가 되어 z가 100 넘게 터진다(EXP-044 t-SNE가 한 줄로 늘어선 원인) → 척도 하한 = 정상 표준편차의 0.5배
# ---------------------------------------------------------------------------
def map_scale(A_ref, *As, floor=0.5):
    """열마다 (v − 정상 중앙) / max(1.4826·MAD, floor·정상 sd). A_ref = 기준 정상 행. 반환 = As 각각의 변환"""
    A_ref = np.asarray(A_ref, float)
    med = np.median(A_ref, 0)
    sc = np.maximum(1.4826 * np.median(np.abs(A_ref - med), 0), floor * A_ref.std(0)) + 1e-9
    return [(np.asarray(A, float) - med) / sc for A in As]


def ae_map_cols(ns, seed=0):
    """AE(정상만 학습, src.experts.ae_segerr) 복원 오차 → train OOF 5겹 열 (n, 5): [전체 평균, 1–2 · 2–4 · 4–8 · 8–12.5 kHz 칸 평균], log10.
    라벨은 '정상만 학습'에만 쓴다(AE의 정의). 칸 = segment_logpsd 1025 bin(12.2 Hz 간격)"""
    from sklearn.model_selection import StratifiedKFold
    from src.experts import ae_segerr
    fb = np.arange(1025) * 25000 / 2048
    bands = [(1000, 2000), (2000, 4000), (4000, 8000), (8000, 12500)]
    out = np.empty((ns.n, 5)); idx = np.arange(ns.n)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, ns.y):
        e = ae_segerr(idx[a], idx[b]).mean(1)                       # (te, 1025)
        out[b, 0] = np.log10(e.mean(1))
        for k, (lo, hi) in enumerate(bands):
            out[b, k + 1] = np.log10(e[:, (fb >= lo) & (fb < hi)].mean(1))
    return out


AE_MAP_COLS = ["AE_mean", "AE_1-2k", "AE_2-4k", "AE_4-8k", "AE_8-12.5k"]


def tsne_best(Z, seeds=3, perplexity=30):
    """seed 여러 개 중 KL 최소 배치 → (KL, 배치 (n, 2), seed별 배치 목록)"""
    from sklearn.manifold import TSNE
    runs = []
    for s_ in range(seeds):
        t_ = TSNE(2, perplexity=perplexity, init="random", random_state=s_, learning_rate="auto"); E = t_.fit_transform(Z)
        runs.append((float(t_.kl_divergence_), E))
    kl, E = min(runs, key=lambda r: r[0])
    return kl, E, [r[1] for r in runs]


def knn_report(Z, y, k=5, seeds=5):
    """공간의 정상-고장 분리 숫자: (5-NN 고장 이웃 중 정상 비율 평균, 정상 이웃 중 고장 비율 평균, 폴드 밖 5-NN ROC-AUC 평균, sd)"""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    Z, y = np.asarray(Z, float), np.asarray(y)
    _, I = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z); I = I[:, 1:]
    other = (y[I] != y[:, None]).mean(1)
    aucs = []
    for s in range(seeds):
        pr = np.empty(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(Z, y):
            pr[b] = KNeighborsClassifier(k).fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
        aucs.append(roc_auc_score(y, pr))
    return float(other[y == 1].mean()), float(other[y == 0].mean()), float(np.mean(aucs)), float(np.std(aucs))


# ---------------------------------------------------------------------------
# 11-22 기본 t-SNE (사용자 결정, 4-10 S3 방식): 모든 실험의 tsne_라벨 그림은 이 함수로 그리고 핵심 그림(key=True)으로 둔다
# 공간 = 모델 입력 H 열(+ 봉우리 H16 · H17, 박자 S) + AE 5열, map_scale(기준 = 지도에 넣는 정상). 지도 = 146개(TSNE_DROP 제외)
# 왼쪽 = 라벨 없는 배치(회색), 오른쪽 = 같은 배치에 라벨 색. 숫자 = 폴드 밖 5-NN AUC · 고장 이웃 중 정상 비율
# ---------------------------------------------------------------------------
TSNE_DROP = ("nor_059", "nor_065", "nor_116", "abn_022")
TSNE_LABELS = ("abn_002", "abn_029", "abn_009", "abn_018", "abn_013", "abn_014", "abn_017", "abn_023", "abn_028")


def tsne_space(ns, T=None, cols=None, drop=TSNE_DROP):
    """기본 t-SNE 공간 → (Z (지도 파일 × 열), keep (150 bool), 열 이름). tsne_default와 같은 계산(그림 없이). 두 실험 비교용"""
    T = ns.T if T is None else T; cols = list(ns.cols if cols is None else cols)
    names = np.array(ns.names); nor = ns.y == 0; keep = ~np.isin(names, list(drop))
    Tn = T.copy(); pk = PeakRef(ns.df).fit(ns.L[nor & keep]).transform(ns.L); Tn["H16_peak_h"], Tn["H17_peak_w"] = pk[:, 0], pk[:, 1]
    A = np.column_stack([Tn[cols].values, ae_map_cols(ns)])
    return map_scale(A[nor & keep], A)[0][keep], keep, cols + AE_MAP_COLS


def tsne_default(ns, T=None, cols=None, drop=TSNE_DROP, mark=(), mark_label="", labels=TSNE_LABELS, title=""):
    """기본 t-SNE 그림 → (fig, info). T · cols = H 표와 열(기본 ns.T · ns.cols; 열을 바꾼 실험은 자기 표 · 열을 넘긴다. H16 · H17은 여기서 계산).
    mark = 빨간 원으로 표시할 파일 번호(150개 기준, 예: 오경보). info = dict(kl, auc, auc_sd, abn_nb_nor, nor_nb_abn, n)"""
    import matplotlib.pyplot as plt
    cols = list(ns.cols if cols is None else cols)
    names = np.array(ns.names); y = ns.y
    Z, keep, colsA = tsne_space(ns, T, cols, drop); A = Z; yk, nk = y[keep], names[keep]
    kl, E, _ = tsne_best(Z); k5 = knn_report(Z, yk)
    mk = np.zeros(len(names), bool); mk[list(mark)] = True; mk = mk[keep]
    fig, axs = plt.subplots(1, 2, figsize=(17, 8))
    axs[0].scatter(*E.T, s=22, color="#666", alpha=.8, lw=0)
    axs[0].set_title(f"라벨 없음 — {keep.sum()}개 ({' · '.join(short(d) for d in drop)} 제외), {A.shape[1]}열 (H {len(cols)} + AE 5)", fontsize=9.5)
    ax = axs[1]
    ax.scatter(*E[yk == 0].T, s=22, color=NOR, alpha=.65, lw=0, label=f"정상 ({(yk == 0).sum()})")
    ax.scatter(*E[yk == 1].T, s=40, color=ABN, lw=0, label=f"고장 ({(yk == 1).sum()})")
    if mk.any(): ax.scatter(*E[mk].T, s=130, facecolor="none", edgecolor="#d62728", lw=1.6, label=mark_label or "표시")
    for f_ in labels:
        if f_ in nk:
            i = list(nk).index(f_); ax.annotate(short(f_), E[i], xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=ABN if f_.startswith("abn") else NOR)
    ax.legend(fontsize=8.5, loc="upper right"); ax.set_title("같은 배치, 라벨 색 (비교용)", fontsize=9.5)
    for a_ in axs: a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"t-SNE 기본(4-10){(' · ' + title) if title else ''} — perplexity 30, KL {kl:.2f} · 폴드 밖 5-NN AUC {k5[2]:.3f} · 고장 이웃 중 정상 {k5[0]:.2f}", x=.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    return fig, dict(kl=kl, auc=k5[2], auc_sd=k5[3], abn_nb_nor=k5[0], nor_nb_abn=k5[1], n=int(keep.sum()))


# ---------------------------------------------------------------------------
# 11-22 기본 t-SNE + val (사용자 요청): 기본 공간(tsne_space와 같음)에 val 40개를 얹는다. 라벨 없음
# 왼쪽 = train 지도(기본 t-SNE) 위에 val을 원래 공간의 가까운 train 5개 위치 평균으로 올림, 오른쪽 = train + val을 함께 다시 펼침
# val AE 5열 = train 정상 전체로 학습한 AE(ae_segerr와 같은 구조 · seed)의 복원 오차. train AE 열은 5겹 OOF
# ---------------------------------------------------------------------------
def ae_map_cols_val(ns):
    """val의 AE 5열 (n_val, 5): train 정상 전체로 학습한 AE. 열 = AE_MAP_COLS"""
    from src.experts import LP as LP_TR, segment_logpsd
    from src.final import _fit_ae
    mu, sd, mdl = _fit_ae(LP_TR[ns.y == 0]); _, LPv = segment_logpsd(ns.Xv)
    Z = (LPv - mu) / sd; R = mdl.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape); e = ((R - Z) ** 2).mean(1)
    fb = np.arange(1025) * 25000 / 2048
    return np.column_stack([np.log10(e.mean(1))] + [np.log10(e[:, (fb >= lo) & (fb < hi)].mean(1)) for lo, hi in ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 12500))])


def tsne_space_val(ns, T=None, Tv=None, cols=None, drop=TSNE_DROP):
    """기본 공간의 (train 지도 행 Z, val 행 Zv, keep, 열 이름). 눈금 기준 = 지도에 넣는 train 정상"""
    T = ns.T if T is None else T; Tv = ns.Tv if Tv is None else Tv; cols = list(ns.cols if cols is None else cols)
    names = np.array(ns.names); nor = ns.y == 0; keep = ~np.isin(names, list(drop))
    pr = PeakRef(ns.df).fit(ns.L[nor & keep]); Tn, Tvn = T.copy(), Tv.copy()
    for D_, L_ in ((Tn, ns.L), (Tvn, ns.Lv)):
        pk = pr.transform(L_); D_["H16_peak_h"], D_["H17_peak_w"] = pk[:, 0], pk[:, 1]
    A, Av = np.column_stack([Tn[cols].values, ae_map_cols(ns)]), np.column_stack([Tvn[cols].values, ae_map_cols_val(ns)])
    Zt, Zv = map_scale(A[nor & keep], A, Av)
    return Zt[keep], Zv, keep, cols + AE_MAP_COLS


def tsne_val_default(ns, val_fault, T=None, Tv=None, cols=None, title="", drop=TSNE_DROP):
    """기본 t-SNE + val 그림 → (fig, 표 DataFrame). val_fault = val 40개 고장 판정(bool, ns.nv 순서).
    표 = val마다 가까운 train 5개 중 고장 수 · 가장 가까운 train 거리 · 그 거리가 train끼리 거리의 몇 %인가 · |z|가 가장 큰 열"""
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    Zt, Zv, keep, colsA = tsne_space_val(ns, T, Tv, cols, drop)
    names = np.array(ns.names); yk, nk = ns.y[keep], names[keep]; vf = np.asarray(val_fault, bool); nv = [s[-3:] for s in ns.nv]
    nn = NearestNeighbors(n_neighbors=6).fit(Zt); dt, _ = nn.kneighbors(Zt); d_tr = dt[:, 1]          # train끼리 가장 가까운 거리 (자기 제외)
    dv, Iv = nn.kneighbors(Zv, n_neighbors=5)
    kl, Et, _ = tsne_best(Zt)
    Eplace = Et[Iv].mean(1)                                                                              # 왼쪽: 가까운 train 5개 위치 평균
    klj, Ej, _ = tsne_best(np.vstack([Zt, Zv])); Ejt, Ejv = Ej[:len(Zt)], Ej[len(Zt):]
    q95, qmax = np.percentile(d_tr, 95), d_tr.max()
    zi = np.abs(Zv).argmax(1)
    tab = pd.DataFrame(dict(파일=nv, 판정=np.where(vf, "고장", "정상"), 이웃고장=(yk[Iv] == 1).sum(1), 최근접거리=dv[:, 0].round(2),
                            거리백분위=[round(float(np.mean(d_tr <= d) * 100)) for d in dv[:, 0]], 최대z열=[colsA[j] for j in zi], 최대z=[round(float(Zv[i, j]), 1) for i, j in enumerate(zi)]))
    far = dv[:, 0] > q95
    fig, axs = plt.subplots(1, 2, figsize=(17, 8))
    for ax, (Etr, Ev, ttl) in zip(axs, ((Et, Eplace, f"기본 지도 ({keep.sum()}개) 위에 val (원래 공간의 가까운 train 5개 위치 평균)"),
                                       (Ejt, Ejv, f"{keep.sum()} + val {len(nv)}을 함께 다시 t-SNE (KL {klj:.2f})"))):
        ax.scatter(*Etr[yk == 0].T, s=20, color=NOR, alpha=.35, lw=0, label=f"train 정상 ({(yk == 0).sum()})")
        ax.scatter(*Etr[yk == 1].T, s=34, color=ABN, alpha=.55, lw=0, label=f"train 고장 ({(yk == 1).sum()})")
        ax.scatter(*Ev[~vf].T, marker="^", s=70, color="#c9b3ea", edgecolor=PUR, lw=1, label=f"val 정상 판정 ({(~vf).sum()})")
        ax.scatter(*Ev[vf].T, marker="^", s=80, color=PUR, edgecolor="#222", lw=1.2, label=f"val 고장 판정 ({vf.sum()})")
        ax.scatter(*Ev[far].T, s=260, facecolor="none", edgecolor="#d62728", lw=1.4, label=f"train에서 먼 val (최근접 거리 > train 95 %, {far.sum()})")
        for i in np.flatnonzero(vf | far): ax.annotate(nv[i], Ev[i], xytext=(5, 4), textcoords="offset points", fontsize=8, color=PUR)
        ax.set_title(ttl, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"기본 t-SNE + val{(' · ' + title) if title else ''} — {Zt.shape[1]}열 (H {len(colsA) - 5} + AE 5), 제외 {' · '.join(short(d) for d in drop)}", x=.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    tab.attrs.update(q95=float(q95), qmax=float(qmax))
    return fig, tab


# ---- J-축소 (EXP-050 기준 구성) 공용 조립 (11-22, EXP-052부터) ----
RING_COLS = ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]


def j_mats(ns, r83, fixed=None):
    """J 계열 입력 행렬. train M = [H 고정 열 | L1_lock | 봉우리 | 파일 번호 | 8.3k 켜짐], val Mv = [H 고정 열 | L1_lock | 봉우리] (EXP-050 mats와 같음)"""
    fixed = [new_name(c) for c in (ns.FIXED if fixed is None else fixed)]
    M = np.column_stack([ns.T[fixed].values, ns.T[["L1_lock"]].values, ns.L, np.arange(ns.n, dtype=float), np.asarray(r83, float)])
    Mv = np.column_stack([ns.Tv[fixed].values, ns.Tv[["L1_lock"]].values, ns.Lv])
    return M, Mv


def j_shrink(ns, fixed=None, alpha=3.0):
    """J-축소 모델 팩토리: RingShrink(Add83(TwoStack lin + E2 + AE)). EXP-050 build(F0, True)와 같다"""
    from src.models2 import TwoStack, RingShrink
    fixed = [new_name(c) for c in (ns.FIXED if fixed is None else fixed)]; nf = len(fixed)
    base = lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(fixed), nf=nf, df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=True))
    ri = [fixed.index(c) for c in RING_COLS if c in fixed]; i1 = fixed.index("H1_nimp")
    return lambda: RingShrink(base, ri, i1, alpha)


def ts_of(top):
    """학습된 J-축소 → (안쪽 TwoStack, 입력 변환). val은 ts.fires_raw(tf(Mv), raw)로"""
    return top.m_.m_, top._tf
