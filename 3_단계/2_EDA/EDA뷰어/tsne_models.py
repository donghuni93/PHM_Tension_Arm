# %% EDA 뷰어 "경계" 화면 아래 t-SNE용 (11-22, 사용자 요청): EXP-044 · EXP-047 기본 t-SNE(4-10 방식)에 val 40개를 함께 펼친 좌표 → figs/tsne_models.json
#   공간 = 그 모델의 H 열 + 봉우리 + 박자 S + AE 5열, map_scale(기준 = 지도에 넣는 train 정상). train 146개(059 · 065 · 116 · 022 제외) + val 40개를 한 번에 펼친다(라벨 안 씀)
#   val AE 열 = train 정상 전체로 학습한 AE(같은 구조 · seed)의 복원 오차. train AE 열은 5겹 OOF(src.exp2util.ae_map_cols)
#   두 모델 지도를 비교하기 쉽게 047은 044 배치에서 출발해 펼치고 회전 · 크기만 맞춘다. val마다 원래 공간(t-SNE 전)의 가까운 train 5개 중 고장 수도 저장
#   실행: conda run -n qe-phm python tsne_models.py
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else "."); HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from src.exp2util import NS, PeakRef, map_scale, ae_map_cols, tsne_best, TSNE_DROP, AE_MAP_COLS
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae
from src.features2 import cached_hi_split, HI_SPLIT
ns = NS(); y, names = ns.y, np.array(ns.names); nor = y == 0; keep = ~np.isin(names, list(TSNE_DROP))

# val AE 5열 (train 정상 전체로 학습)
mu, sd, mdl = _fit_ae(LP_TR[y == 0]); _, LPv = segment_logpsd(ns.Xv)
Zv_ = (LPv - mu) / sd; R = mdl.predict(Zv_.reshape(-1, Zv_.shape[-1])).reshape(Zv_.shape); e = ((R - Zv_) ** 2).mean(1)
fb = np.arange(1025) * 25000 / 2048
AEv = np.column_stack([np.log10(e.mean(1))] + [np.log10(e[:, (fb >= lo) & (fb < hi)].mean(1)) for lo, hi in ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 12500))])
AEt = ae_map_cols(ns)
SP, SPv = cached_hi_split("train", ns.X), cached_hi_split("val", ns.Xv)
Hs, Hsv = pd.concat([ns.T, SP], axis=1), pd.concat([ns.Tv, SPv], axis=1)
FX47 = [c for c in ns.FIXED if c not in ["H10_ring_8-12.5k", "H15_share_8-12.5k"]] + HI_SPLIT
CFG = {"EXP-044": (ns.T, ns.Tv, list(ns.cols)), "EXP-047": (Hs, Hsv, FX47 + list(ns.PEAK) + ["L1_S"])}


def space(T, Tv, cols):
    pr = PeakRef(ns.df).fit(ns.L[nor & keep])
    Tn, Tvn = T.copy(), Tv.copy()
    for D_, L_ in ((Tn, ns.L), (Tvn, ns.Lv)):
        pk = pr.transform(L_); D_["H16_peak_h"], D_["H17_peak_w"] = pk[:, 0], pk[:, 1]
    A, Av = np.column_stack([Tn[cols].values, AEt]), np.column_stack([Tvn[cols].values, AEv])
    Zt, Zv = map_scale(A[nor & keep], A, Av)
    return np.vstack([Zt[keep], Zv]), Zt[~keep]                  # 뒤 = 지도에서 뺀 4개 (펼치지 않고 가까운 5개 위치에 올림, 11-22 6-2와 같은 방식)


def tsne_from(Z, init):
    I = init - init.mean(0); I = I / I[:, 0].std() * 1e-4
    return TSNE(2, perplexity=30, init=I, random_state=0, learning_rate="auto").fit_transform(Z)


def align(E, ref):
    a, b = E - E.mean(0), ref - ref.mean(0); a = a / np.linalg.norm(a) * np.linalg.norm(b)
    Rm, _ = orthogonal_procrustes(a, b); return a @ Rm + ref.mean(0)


ntk, nv = int(keep.sum()), len(ns.nv); tr_idx = np.flatnonzero(keep)
OUT = {}; E0 = None
for mid, (T, Tv, cols) in CFG.items():
    Z, Zd = space(T, Tv, cols)
    if E0 is None:
        kl, E0, _ = tsne_best(Z); E = tsne_from(Z, E0)
    else:
        E = align(tsne_from(Z, E0), OUT["EXP-044"]["_E"])
    # val마다 원래 공간에서 가까운 train 5개 (지도에 넣은 train 146개 중)
    nn = NearestNeighbors(n_neighbors=5).fit(Z[:ntk]); _, I = nn.kneighbors(Z[ntk:])
    nb = [[str(names[tr_idx[j]]) for j in row] for row in I]
    xy = [None] * ns.n
    for k, i in enumerate(tr_idx): xy[i] = [round(float(E[k, 0]), 3), round(float(E[k, 1]), 3)]
    _, Id = nn.kneighbors(Zd); drop_idx = np.flatnonzero(~keep)       # 뺀 4개: 원래 공간에서 가까운 train 5개(146개 중)의 지도 위치 평균
    for k, i in enumerate(drop_idx): p = E[Id[k]].mean(0); xy[i] = [round(float(p[0]), 3), round(float(p[1]), 3)]
    OUT[mid] = dict(_E=E, tr=xy, out=[int(i) for i in drop_idx], val=[[round(float(v[0]), 3), round(float(v[1]), 3)] for v in E[ntk:]], val_names=list(ns.nv),
                    nb=nb, nb_abn=[int(sum(n_.startswith("abn") for n_ in row)) for row in nb], ncol=int(Z.shape[1]),
                    label=f"기본 t-SNE(4-10) · {mid} 공간 {Z.shape[1]}열 (H {len(cols)} + AE 5) · train {ntk}개 + val {nv}개 함께 펼침 · ◆ = 펼치기에서 뺀 {' · '.join(TSNE_DROP)} (가까운 train 5개 위치에 올림)")
    print(mid, "done", flush=True)
for v in OUT.values(): v.pop("_E")
json.dump(OUT, open(os.path.join(HERE, "figs", "tsne_models.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("saved")
