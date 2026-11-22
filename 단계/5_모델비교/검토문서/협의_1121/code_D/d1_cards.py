"""D 진단 카드: (1) 지표 위치 (2) 충격 목록 (3) 라벨 없는 닮은 train 5개 (4) 반사실 판정 분해. 라벨은 보고용."""
from common import *
from src.features2 import PeakRef, cached_feat2
from src.final import _fit_et, _et_score
P = log_to("d1_cards_log.txt")
X, y, names, Xv, nv = load_all(); FM = load_FM(); core = FM.core_; THR = FM.thr_
Ttr, Tva, cols = tables(FM, X, Xv)
NA, IMP = pickle.load(open(os.path.join(CACHE, "impacts.pkl"), "rb")); NAi = {n_: i for i, n_ in enumerate(NA)}
n = len(y); nor = y == 0
Dt = core.parts(Ttr); Dv = core.parts(Tva)
oofz = core.oof_z_
lr_oof = core.lr_.decision_function(oofz)
F2tr, Ltr, fr = cached_feat2("train", X); F2va, Lva, _ = cached_feat2("val", Xv)
pk = PeakRef(float(fr[1] - fr[0])).fit(Ltr[nor])
ptr, pva = pk.transform(Ltr), pk.transform(Lva)
MM = pd.read_csv(os.path.join(ROOT, "단계", "4_피처", "base", "4-8_1121_모드지도", "features_modemap.csv")).set_index("file")


def frame(T, F2, pp, nm):
    E = pd.DataFrame(T["E"], columns=cols); E.index = nm
    d = E.copy()
    for k, c in zip(("R24", "RBODY"), (0, 1)): d["RING_" + k] = T["RING"][:, c]
    for k in ("P83", "W83", "RES"): d["RULE_" + k] = T[k]
    for j, k in enumerate(("K1", "K2", "K3")): d["RULE_" + k] = T["K"][:, j]
    d["RULE_n20"] = T["n20"]; d["RULE_S"] = T["S"]; d["RULE_NIMP"] = T["NIMP"]
    f2 = F2.copy(); f2.index = nm
    f2["H16_peak_h"], f2["H17_peak_w"] = pp[:, 0], pp[:, 1]
    d = d.join(f2).join(MM.drop(columns=["split", "label"]))
    return d


U = pd.concat([frame(Ttr, F2tr, ptr, names), frame(Tva, F2va, pva, nv)])
U = U.loc[:, ~U.columns.duplicated()].replace([np.inf, -np.inf], np.nan)


def group_of(c):
    p = c.split("_")[0]
    if c in ("H18_hfl", "RULE_K2"): return "맥락"
    if c in ("RULE_S", "RULE_n20", "RULE_NIMP", "RULE_K1", "RULE_K3", "L1_S", "L1_r", "L1_T", "L1_hit", "H1_nimp", "H2_ratio21", "n_aligned", "n_imp") or p in ("P02", "S01", "E01") or c.startswith(("H3_", "H4_", "H5_", "H6_")): return "박자"
    if p in ("T01", "T02", "T03", "T04", "T06", "T12", "T13", "T16", "T17", "T19", "W01", "W02"): return "입력"
    if p in ("T07", "T08", "T09", "T10", "T11", "T15", "T18", "T20", "W03", "W04"): return "박자"
    return "구조"


G = {c: group_of(c) for c in U.columns}
VW = pd.read_csv(os.path.join(ROOT, "단계", "2_EDA", "2-4_EDA뷰어", "figs", "features_hp.csv"), encoding="utf-8-sig", index_col=0)
VCTX = ["load_hf", "lb_rlp", "lb_e0_20", "lb_e20_200", "lb_hit", "lb_S", "rms_hf", "fast_decay", "n_ring", "ring_o1", "ring_o2", "ring_o3", "ring_o4", "kurt_1k"]
V = VW[VCTX].add_prefix("VW_")
for c in V.columns:
    G[c] = {"VW_load_hf": "맥락", "VW_lb_rlp": "맥락", "VW_lb_e0_20": "맥락", "VW_lb_e20_200": "맥락", "VW_lb_hit": "박자", "VW_lb_S": "박자", "VW_rms_hf": "입력", "VW_kurt_1k": "박자"}.get(c, "구조")
Un = U.loc[names]
Rn = {c: np.sort(Un[c].values[nor][~np.isnan(Un[c].values[nor])]) for c in U.columns}


def pct(c, v):
    r = Rn[c]
    if np.isnan(v) or len(r) == 0: return np.nan
    lo, hi = np.searchsorted(r, v, "left"), np.searchsorted(r, v, "right")
    return 100 * (lo + hi) / 2 / len(r)


RV = {c: np.sort(V[c].values[nor]) for c in V.columns}
E_all = np.vstack([np.hstack([Ttr["E"], Ttr["RING"]]), np.hstack([Tva["E"], Tva["RING"]])])
En = E_all[:n][nor]; med = np.median(En, 0); sp = np.percentile(En, 95, 0) - np.percentile(En, 5, 0); sp = np.where(sp > 1e-9, sp, 1.0)
Zn = np.clip((E_all - med) / sp * 3.29, -6, 6)


def knn(i, k=5):
    d = np.linalg.norm(Zn[:n] - Zn[i], axis=1)
    if i < n: d[i] = np.inf
    o = np.argsort(d)[:k]; return [(names[j], int(y[j]), float(d[j])) for j in o]


Efit = np.hstack([Ttr["E"], Ttr["RING"]]); ecols = cols + ["RING_R24", "RING_RBODY"]; nmed = np.median(Efit[nor], 0)
et_full = core.experts_[1]; m1, s1 = core.med_[1], core.sc_[1]


def et_z(et, M): return (_et_score(et, M) - m1) / s1


def family(c):
    p = c.split("_")[0]
    return {"B04": "B04 스펙트럼 칸 비율", "B05": "B05", "P02": "P02 대역 충격(첨도·파고)", "S01": "S01 시간 변동", "E01": "E01 포락선"}.get(p, "단일 T/W/F/RING 열")


def counterfactual(i, is_val):
    if is_val: et = et_full; x = E_all[n + i]
    else:
        keep = np.ones(n, bool); keep[i] = False; et = _fit_et(Efit[keep], y[keep], seed=0); x = Efit[i]
    base = float(et_z(et, x[None])[0]); allnorm = float(et_z(et, nmed[None])[0])
    M = np.tile(x, (len(x), 1)); M[np.arange(len(x)), np.arange(len(x))] = nmed
    dz = base - et_z(et, M)
    fams = sorted(set(family(c) for c in ecols)); famcf = {}
    for k in fams:
        mask = np.array([family(c) == k for c in ecols]); Mx = x.copy(); Mx[mask] = nmed[mask]; famcf[k] = base - float(et_z(et, Mx[None])[0])
    top = np.argsort(-np.abs(dz))[:5]
    return base, allnorm, [(ecols[j], float(dz[j])) for j in top], famcf


MISS = ["abn_022", "abn_002", "abn_017", "abn_009"]
FP = ["nor_116", "nor_065", "nor_059", "nor_037", "nor_110", "nor_095", "nor_055", "nor_097", "nor_101", "nor_120"]
rule_on = (Dv.rule83 | Dv.knock).values
bnd = [nv[i] for i in range(40) if (not rule_on[i]) and abs(Dv.lr.values[i] - THR) < 1.0]
rule_val = [nv[i] for i in range(40) if rule_on[i]]
CARD = [(f, "miss" if f in MISS else "fp") for f in MISS + FP] + [(f, "val") for f in bnd + rule_val]
rows, txt = [], []
for f, kind in CARD:
    is_val = kind == "val"; i = nv.index(f) if is_val else names.index(f); ia = (n + i) if is_val else i
    u = U.loc[f]; out = {}
    for c in U.columns:
        p_ = pct(c, u[c])
        if not np.isnan(p_) and (p_ < 5 or p_ > 95): out.setdefault(G[c], []).append((c, p_))
    if not is_val:
        for c in V.columns:
            r = RV[c]; p_ = 100 * (np.searchsorted(r, V.loc[f, c], "left") + np.searchsorted(r, V.loc[f, c], "right")) / 2 / len(r)
            if p_ < 5 or p_ > 95: out.setdefault(G[c], []).append((c, p_))

    def top_names(g, k=4):
        L = sorted(out.get(g, []), key=lambda t: -abs(t[1] - 50)); return ", ".join(f"{c}({p_:.0f}%)" for c, p_ in L[:k])

    cnt = {g: len(out.get(g, [])) for g in ("입력", "구조", "맥락", "박자")}
    if is_val: aez, etz, lr = Dv.ae_z.values[i], Dv.et_z.values[i], Dv.lr.values[i]
    else: aez, etz = oofz[i]; lr = float(lr_oof[i])
    nn = knn(ia); nfault = sum(t[1] for t in nn)
    base, allnorm, topf, famcf = counterfactual(i, is_val)
    im = IMP[NAi[f]]
    imp_str = "; ".join(f"{t:.2f}s h{hx:.1f}x {fk:.2f}k z{z:.2f} fit{ft:.2f}" for t, hx, fk, z, ft in zip(im["t"], im["hx"], im["f"] / 1000, im["z"], im["fit"]))
    rule_flag = bool(rule_on[i]) if is_val else bool(Dt.rule83.values[i] | Dt.knock.values[i])
    rows.append(dict(file=f, kind=kind, label=("" if is_val else int(y[i])), ae_z=round(float(aez), 2), et_z=round(float(etz), 2), lr=round(float(lr), 2),
                     dist_thr=round(float(lr - THR), 2), rule=int(rule_flag), n_imp=len(im["t"]), n_out_in=cnt["입력"], n_out_struct=cnt["구조"], n_out_ctx=cnt["맥락"], n_out_beat=cnt["박자"],
                     nn5_fault=nfault, nn5=" ".join(f"{a[-7:]}:{b}" for a, b, c in nn), et_z_loo=round(base, 2), et_z_allnormal=round(allnorm, 2),
                     top_drivers="; ".join(f"{c}:{d_:+.2f}" for c, d_ in topf), fam_cf="; ".join(f"{k}:{v:+.2f}" for k, v in famcf.items())))
    lab = "" if is_val else (" 고장" if y[i] else " 정상")
    txt.append(f"### {f} ({kind}{lab})  AE z {aez:.2f} · ET z {etz:.2f} · lr {lr:.2f} (임계값 1.0 대비 {lr-THR:+.2f}) · 규칙 {'켜짐' if rule_flag else '꺼짐'} · 충격 {len(im['t'])}개")
    txt.append(f"- 지표 5-95% 밖 (입력 {cnt['입력']} · 구조 {cnt['구조']} · 맥락 {cnt['맥락']} · 박자 {cnt['박자']}):")
    for g in ("입력", "구조", "맥락", "박자"):
        if out.get(g): txt.append(f"  - {g}: {top_names(g)}")
    txt.append(f"- 충격: {imp_str if imp_str else '없음'}")
    txt.append("- 닮은 train 5 (라벨 없는 거리, ET 입력 공간): " + ", ".join(f"{a}[{'고장' if b else '정상'}] d{c:.1f}" for a, b, c in nn) + f" → 고장 {nfault}/5")
    txt.append(f"- ET 반사실({'최종 ET' if is_val else '자기 제외 재학습'}): z {base:.2f} · 전부 정상 중앙값이면 {allnorm:.2f} · 상위 피처 " + "; ".join(f"{c} {d_:+.2f}" for c, d_ in topf[:4]))
    txt.append("- 가족 전체를 정상 중앙값으로 바꿨을 때 ET z 감소: " + "; ".join(f"{k} {v:+.2f}" for k, v in famcf.items()))
    txt.append("")
pd.DataFrame(rows).to_csv(os.path.join(HERE, "cards_D.csv"), index=False, encoding="utf-8-sig")
open(os.path.join(HERE, "cards_D.md"), "w", encoding="utf-8").write("# D 진단 카드 (자동 생성, 라벨은 보고용)\n\n" + "\n".join(txt))
P("cards", len(CARD), "val 경계(규칙 꺼짐, |lr-1|<1):", bnd, "규칙 켜진 val:", rule_val)
