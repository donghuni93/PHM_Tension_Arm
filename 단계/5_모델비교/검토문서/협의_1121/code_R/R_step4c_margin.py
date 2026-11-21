"""R 4c: 4b 결과를 보고 (1) '정상 0개가 켜지는 가장 낮은 문턱'이 경계 정상 위에 앉아 있어 폴드마다 경계 정상(N019 N001 N048)이 켜지는 문제를
여유 m(정상 최댓값 u <= 1/(1+m))으로 푼 변형, (2) 고른 울림 대역의 내용(어느 파일이 왜 켜지나), (3) 정상 밖 비율(t-SNE 분리의 거리판)을 잰다.
정의는 4b와 같은 선택기에 '여유'만 더한다. 출력 out/R4c.txt
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "R_step4b_select.py"), encoding="utf-8").read().split("# ================= 1.")[0]
exec(compile(src, "R_step4b_head", "exec"))
import itertools

# ---------- (2) 4b가 고른 넓은 울림 대역 (7750-9000, zeta<=0.04) 안의 파일과 좁은 정의(8000-8700, zeta<=0.02, fit>=0.7)의 차이 ----------
p("[A] 울림 대역 내용: 충격별 (f, zeta, fit) 중 7750-9000 Hz")
for s in ["A011", "A019", "A013", "A002", "A003", "N019", "N092", "N059", "V022", "V035", "V011"]:
    i = sn.index(s); m = (rf[i] >= 7750) & (rf[i] < 9000)
    p(f"   {s}: " + (" ".join(f"({rf[i][j]:.0f},{rz[i][j]:.2f},{rt[i][j]:.2f})" for j in np.flatnonzero(m)) or "-"))
nar = RING[(8000, 8750, 0.02)] if (8000, 8750, 0.02) in RING else ring_v(8000, 8750, 0.02)
wide = RING[(7750, 9000, 0.04)]
p("   좁은 정의(8000-8750, zeta<=.02) fit>=0.7 켜짐: 고장 " + " ".join(sn[TRI[i]] for i in np.flatnonzero(~nor) if nar[TRI[i]] >= .7) + " | 정상 " + (" ".join(sn[TRI[i]] for i in np.flatnonzero(nor) if nar[TRI[i]] >= .7) or "-") + " | val " + (" ".join(sn[i] for i in VAL if nar[i] >= .7) or "-"))
p("   넓은 정의(7750-9000, zeta<=.04) fit>=0.65 켜짐: 고장 " + " ".join(sn[TRI[i]] for i in np.flatnonzero(~nor) if wide[TRI[i]] >= .65) + " | 정상 " + (" ".join(sn[TRI[i]] for i in np.flatnonzero(nor) if wide[TRI[i]] >= .65) or "-") + " | val " + (" ".join(sn[i] for i in VAL if wide[i] >= .65) or "-"))
p("   정상 상위값: 좁은 " + " ".join(f"{sn[TRI[i]]}={nar[TRI[i]]:.2f}" for i in np.argsort(-nar[TRI] * nor)[:4]) + " | 넓은 " + " ".join(f"{sn[TRI[i]]}={wide[TRI[i]]:.2f}" for i in np.argsort(-wide[TRI] * nor)[:5]))

# ---------- (1) 여유 m 변형: 선택기를 '정상 u 최댓값 <= 1/(1+m)'로 ----------
def pick_ring_m(tr, m, narrow=False):
    best = None
    for key, v in RING.items():
        if narrow and key[2] > 0.02: continue
        mx = v[TRI][tr][nor[tr]].max()
        t = max(0.5, mx * (1 + m)) if mx > 0 else 0.5 * (1 + m)
        on = v[TRI][tr] >= t
        sc = (int(on[~nor[tr]].sum()), -(key[1] - key[0]), -key[2])
        if best is None or sc > best[0]: best = (sc, key, t)
    return dict(key=best[1], t=float(best[2]))


def pick_spec_m(tr, m):
    best = None
    for key, (P, W) in SPEC.items():
        for pt in np.arange(0.4, 1.21, 0.05):
            for wt in (100, 125, 150, 175, 200, 225, 250):
                u = np.minimum(P[TRI] / pt, W[TRI] / wt)
                if u[tr][nor[tr]].max() <= 1 / (1 + m):
                    sc = (int((u[tr][~nor[tr]] >= 1).sum()), pt * wt)
                    if best is None or sc > best[0]: best = (sc, key, pt, wt)
    _, key, pt, wt = best
    return dict(key=key, pt=float(pt), wt=float(wt))


def pick_J_m(tr, m):
    best = None
    for a in np.arange(1.0, 2.61, 0.1):
        for b in np.arange(0.4, 1.21, 0.05):
            u = np.minimum(K1n[TRI] / a, K3n[TRI] / b)
            if u[tr][nor[tr]].max() <= 1 / (1 + m):
                sc = (int((u[tr][~nor[tr]] >= 1).sum()), a * b)
                if best is None or sc > best[0]: best = (sc, a, b)
    _, a, b = best
    return dict(a=float(a), b=float(b))


def pick_L_m(tr, m):
    mx = H18g[TRI][tr][nor[tr]].max(); med = np.median(H18g[TRI][tr][nor[tr]])
    return dict(t=float(med + (mx - med) * (1 + m)))      # 정상 중앙~최댓값 거리의 (1+m)배


p("\n[B] 여유 m: 문턱을 학습 정상 최댓값에서 m만큼 더 멀리 (m=0 = 4b). 5겹 x 20 seed 시험 폴드 결과")
SEEDS = range(20)
out = {}
for m in (0.0, 0.1, 0.2, 0.3):
    for narrow in (False, True):
        hitc = np.zeros((n, 4)); cntc = np.zeros(n); un = np.zeros(n); chs = {k: [] for k in EXPK}
        for seed in SEEDS:
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y):
                sel = {"ring": pick_ring_m(tr, m, narrow), "spec": pick_spec_m(tr, m), "J": pick_J_m(tr, m), "L": pick_L_m(tr, m)}
                for k in EXPK: chs[k].append(sel[k])
                U = u_all(sel, tr); on = np.zeros(len(te), bool)
                for j, k in enumerate(EXPK): hitc[te, j] += U[k][TRI][te] >= 1; on |= U[k][TRI][te] >= 1
                un[te] += on; cntc[te] += 1
        hrc = hitc / cntc[:, None]; ur = un / cntc
        out[(m, narrow)] = (hrc, ur, chs)
        p(f"  m={m:.1f} ring{'(zeta<=.02만)' if narrow else '(전체 격자)':>14}: 재현(고장 평균) ring {hrc[~nor,0].mean():.3f} spec {hrc[~nor,1].mean():.3f} J {hrc[~nor,2].mean():.3f} L {hrc[~nor,3].mean():.3f} | 오경보(정상 평균) " +
          " ".join(f"{hrc[nor,j].mean():.4f}" for j in range(4)) + f" | 합집합 재현 {ur[~nor].mean():.3f} 오경보 {ur[nor].mean():.4f} (정상 켜진 파일 {int((ur[nor]>0).sum())})")
p("\n[C] 여유 m별 켜짐 비율 (파일별, ring spec J L 합집합) — 핵심 고장과 경계 정상")
files_ = ["A002", "A003", "A006", "A010", "A029", "A030", "A013", "A005", "A008", "A023", "A004", "A011", "A017", "A019", "A024", "N019", "N092", "N001", "N023", "N048", "N016", "N103", "N119"]
for (m, narrow), (hrc, ur, chs) in out.items():
    if not narrow and m in (0.0, 0.2): pass
    elif narrow and m == 0.2: pass
    else: continue
    p(f"  m={m:.1f} {'narrow' if narrow else 'full'}:")
    p("    " + "  ".join(f"{s}:" + "/".join(f"{hrc[list(names[:0]) or [names.index(('abn_' if s[0]=='A' else 'nor_') + s[1:])], j][0] if False else hrc[names.index(('abn_' if s[0]=='A' else 'nor_') + s[1:]), j]:.1f}" for j in range(4)) for s in files_))
    c = Counter((d["key"], round(d["t"], 2)) for d in chs["ring"]); p("    ring 대역·문턱 최빈: " + "; ".join(f"{k}:{v}%" for k, v in c.most_common(3)))
    c = Counter((d["key"], round(d["pt"], 2), d["wt"]) for d in chs["spec"]); p("    spec 최빈: " + "; ".join(f"{k}:{v}%" for k, v in c.most_common(3)))
    c = Counter((round(d["a"], 1), round(d["b"], 2)) for d in chs["J"]); p("    J 최빈: " + "; ".join(f"{k}:{v}%" for k, v in c.most_common(4)))
    tt = np.array([d["t"] for d in chs["L"]]); p(f"    L 문턱 분위 {np.round(np.percentile(tt,[0,50,100]),3)}")

# 전체 train, m=0.2 변형 정의 (좁은 울림) — 그림·표용 고정 정의 후보 (여유 20%)
M = 0.2; trall = np.arange(n)
SEL2 = {"ring": pick_ring_m(trall, M, True), "spec": pick_spec_m(trall, M), "J": pick_J_m(trall, M), "L": pick_L_m(trall, M)}
U2 = u_all(SEL2, trall)
p("\n[D] 전체 train, 여유 20% 정의: " + json.dumps({k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in d.items()} for k, d in SEL2.items()}, ensure_ascii=False))
for k in EXPK:
    on = U2[k] >= 1
    p(f"   {k:>5}: 고장 {on[TRI][~nor].sum():2d} ({' '.join(sn[TRI[i]] for i in np.flatnonzero(~nor) if on[TRI][i])}) 정상 {on[TRI][nor].sum()} val {on[VAL].sum()} ({' '.join(sn[i] for i in VAL if on[i])})  val 최대 u {U2[k][VAL].max():.2f}({sn[VAL[U2[k][VAL].argmax()]]}) 정상 최대 u {U2[k][TRI][nor].max():.2f}")
A3b = np.max(np.column_stack([np.clip(U2[k], 0, None) for k in EXPK]), 1)
on3 = A3b >= 1
p(f"   A3: 고장 {on3[TRI][~nor].sum()} ({' '.join(sn[TRI[i]] for i in np.flatnonzero(~nor) if on3[TRI][i])}) 정상 {on3[TRI][nor].sum()} val {on3[VAL].sum()} ({' '.join(sn[i] for i in VAL if on3[i])})")
pd.DataFrame({"file": names, "split": split, **{f"u_{k}": U2[k] for k in EXPK}, "A3": A3b}).to_csv(os.path.join(here, "out", "R4c_u_m20.csv"), index=False, encoding="utf-8-sig")

# ---------- (3) 정상 밖 비율 (거리판) ----------
Xtr, ytr, ntr = load_train(); Tt, Lt, fr = cached_feat2("train", Xtr)
FJ = json.load(open(os.path.join(root, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))
pk = PeakRef(float(fr[1] - fr[0])).fit(Lt[nor]).transform(Lt)
Tt = Tt.copy(); Tt["H16_peak_h"], Tt["H17_peak_w"] = pk[:, 0], pk[:, 1]
BASE = Tt[list(FJ["final"])].values
EXT = cached_table("ext", feature_table_ext, Xtr).values


def zs(A, ref):
    Z = np.empty_like(A, float)
    for j in range(A.shape[1]): Z[:, j] = robust_z(A[ref, j])(A[:, j])
    return Z


def outside(Z, ref, test_f):
    """고장 점의 '가장 가까운 학습 정상까지 거리' > 학습 정상끼리 LOO 최근접 거리의 95%  -> 정상 무리 밖. 반환 파일별 bool"""
    Zr = Z[ref]
    D = np.sqrt(((Zr[:, None] - Zr[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    q = np.percentile(D.min(1), 95)
    dd = np.sqrt(((Z[test_f][:, None] - Zr[None]) ** 2).sum(-1)).min(1)
    return dd > q


def purity(Z, test_faults):
    idx = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z[test_faults], return_distance=False)[:, 1:]
    return y[idx].mean(1)


grpA = ["A002", "A003", "A006", "A010", "A029", "A030", "A013"]          # 8.3k 군
grpB = ["A014", "A017", "A004", "A008", "A011", "A019", "A005", "A023"]    # 유격·하중 군
iA = [names.index(s.replace("A", "abn_")) for s in grpA]; iB = [names.index(s.replace("A", "abn_")) for s in grpB]
rest = [i for i in np.flatnonzero(~nor) if i not in iA and i not in iB]
p("\n[E] 정상 무리 밖 비율 (고장 점의 최근접 정상 거리 > 학습 정상 LOO 최근접 거리 95%) / 5-NN 순도  — 폴드 안 재선택(여유 20%·ring 좁음), 100 폴드, 시험 고장만")
folds = [(seed, tr, te, None) for seed in SEEDS for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y)]
acc = {}
for seed, tr, te, _ in folds:
    sel = {"ring": pick_ring_m(tr, M, True), "spec": pick_spec_m(tr, M), "J": pick_J_m(tr, M), "L": pick_L_m(tr, M)}
    U = u_all(sel, tr); Uc = np.column_stack([np.clip(U[k][TRI], 0, 3) for k in EXPK]); ref = tr[nor[tr]]
    tf = te[~nor[te]]
    spaces = {"ext150": EXT, "base16": BASE, "base16+u(ring,spec)": np.hstack([BASE, Uc[:, :2]]), "base16+u 4개": np.hstack([BASE, Uc]), "base16+A3": np.hstack([BASE, Uc.max(1)[:, None]]),
              "u 4개만": Uc}
    for nm_, Mx in spaces.items():
        Z = zs(Mx, ref) if nm_ != "u 4개만" else Mx
        o = outside(Z, ref, tf); pu = purity(Z, tf)
        for i, oo, pp in zip(tf, o, pu): acc.setdefault(nm_, {}).setdefault(i, []).append((oo, pp))
for nm_, d in acc.items():
    def agg(ids, which):
        v = [np.mean([t[which] for t in d[i]]) for i in ids if i in d]; return np.mean(v) if v else float("nan")
    p(f"   {nm_:>20}: 밖 비율 전체 {agg(np.flatnonzero(~nor),0):.2f} | 8.3k군 {agg(iA,0):.2f} · 유격군 {agg(iB,0):.2f} · 나머지 {agg(rest,0):.2f} || 순도 전체 {agg(np.flatnonzero(~nor),1):.2f} | 8.3k군 {agg(iA,1):.2f} · 유격군 {agg(iB,1):.2f} · 나머지 {agg(rest,1):.2f}")
p("   (밖 비율 기대: 정상 자신이면 5%)  8.3k군 = " + " ".join(grpA) + " / 유격·하중 군 = " + " ".join(grpB))
p("   파일별 밖 비율 base16 → base16+u(ring,spec) (8.3k군·유격군):")
for s, i in zip(grpA + grpB, iA + iB):
    p(f"     {s}: {np.mean([t[0] for t in acc['base16'][i]]):.2f} -> {np.mean([t[0] for t in acc['base16+u(ring,spec)'][i]]):.2f} -> (4개) {np.mean([t[0] for t in acc['base16+u 4개'][i]]):.2f}")
open(os.path.join(here, "out", "R4c.txt"), "w", encoding="utf-8").write(buf.getvalue())
print("done")
