"""정상 증거 피처 (4-9, 11-22): 정상 파일에 흔하고 고장에는 (거의) 없는 성질. 파일마다 독립, 라벨을 쓰지 않는다.

모든 정의는 src.impacts의 새 충격 정의(1 kHz 고역 · |Hilbert| 포락선 · 100 ms 중앙값 바탕 · 4.5배) 위에서 한다.
사건 = 50 ms 안에 이어진 충격 묶음 (검토문서/협의_1121/code_D/d5_repeat.py 와 같은 정의).

원값 열 (정상 쪽이 낮은 것은 방향이 −)
  N1_reb      사건 뒤 30–300 ms 창의 후속 충격 높이(g) 합 ÷ 그 사건 최대 충격 높이, 사건 중 최댓값 (충격 0 → 0)      방향 −
  N2_ret_ms   가장 큰 사건(최대 height_g)의 포락선이 바탕의 2배 아래로 내려오기까지 ms (상한 1000, 충격 0 → 0)     방향 −
  N3_ifrac    충격 창([−2 ms, +10 ms]) 안 고역 에너지 ÷ 전체 고역 에너지                                            방향 −
  N3_crest    고역 신호 max|x| ÷ RMS                                                                              방향 −
  N5_edge     충격 높이² 중 첫·끝 100 ms 안 몫 (충격 0 → 0)                                                       방향 +
  N6_calm     100 ms 창 20개의 고역 RMS 중 파일 자기 중앙값 × 1.5 아래인 창의 비율                                  방향 +
  (N4 = 15열 중 정상 5–95 % 밖 개수 는 run.py에서: 학습 정상에 의존)
"""
import numpy as np
from scipy.signal import hilbert, sosfiltfilt

from src.impacts import FS, _baseline, _sos, detect_impacts, highpass, impact_rings_v2

GAP_EV = 0.05          # 사건 묶음 간격 (s)
REB_LO, REB_HI = 0.03, 0.30
EDGE_S = 0.10


def file_work(x):
    """신호 1개 → (스칼라 dict, 배열 dict)"""
    x = np.asarray(x, float)
    xh = highpass(x)
    imp = detect_impacts(x, _hp=xh)
    rings = impact_rings_v2(x, imp=imp, _hp=xh)
    e = sosfiltfilt(_sos("lowpass", 1000, FS), np.abs(hilbert(xh)))
    b = _baseline(e)
    t, hg, hx, idx = imp["t"], imp["height_g"], imp["height_x"], imp["idx"]
    k = len(t)
    d = dict(n_imp=k, hp_rms=float(np.sqrt(np.mean(xh ** 2))), hp_peak=float(np.abs(xh).max()),
             raw_peak=float(np.abs(x - x.mean()).max()), env_med=float(np.median(e)),
             log_hp_rms=float(np.log10(np.sqrt(np.mean(xh ** 2)))),
             log_raw200=float(np.log10(np.std(highpass(x, f=200)))))
    r2 = np.sqrt(np.mean(xh.reshape(20, -1) ** 2, 1))
    d["N6_calm"] = float(np.mean(r2 < 1.5 * np.median(r2)))
    d["N3_crest"] = float(np.abs(xh).max() / np.sqrt(np.mean(xh ** 2)))
    mask = np.zeros(len(xh), bool)
    for p in idx:
        mask[max(0, p - int(0.002 * FS)):p + int(0.010 * FS)] = True
    d["N3_ifrac"] = float(np.sum(xh[mask] ** 2) / np.sum(xh ** 2))
    arr = dict(t=t, hg=hg, hx=hx, f=rings[:, 0] if len(rings) == k and k else np.zeros(0))
    if k == 0:
        d.update(n_ev=0, dom=1.0, N1_reb=0.0, N1_nreb=0, N1_mean=0.0, N2_ret_ms=0.0, N2_half_ms=0.0, N2_f=np.nan,
                 N5_edge=0.0, first_t=np.nan, last_t=np.nan, max_hg=0.0, n_ring811=0, big_hg=0.0)
        return d, arr
    grp = np.r_[0, np.cumsum(np.diff(t) > GAP_EV)]
    ne = int(grp.max()) + 1
    eh = np.array([hx[grp == g].max() for g in range(ne)])
    reb, nreb, pk_hg = [], [], []
    for g in range(ne):
        m = np.flatnonzero(grp == g)
        te = t[m[-1]]
        w = (t > te + REB_LO) & (t <= te + REB_HI)
        reb.append(hg[w].sum() / hg[m].max()); nreb.append(int(w.sum())); pk_hg.append(hg[m].max())
    gbig = int(np.argmax(pk_hg))
    m = np.flatnonzero(grp == gbig)
    jb = m[np.argmax(hg[m])]
    p = idx[jb]
    below = (e[p:] / b[p:]) < 2.0
    k2 = int(np.argmax(below)) if below.any() else len(below)
    half = (e[p:] < 0.5 * e[p])
    k3 = int(np.argmax(half)) if half.any() else len(half)
    E = hg ** 2
    edge = (t < EDGE_S) | (t > 2.0 - EDGE_S)
    d.update(n_ev=ne, dom=float(eh.max() / eh.sum()), N1_reb=float(max(reb)), N1_nreb=int(max(nreb)), N1_mean=float(np.mean(reb)),
             N2_ret_ms=float(min(k2 / FS * 1000, 1000.0)), N2_half_ms=float(min(k3 / FS * 1000, 1000.0)),
             N2_f=float(arr["f"][jb]) if len(arr["f"]) else np.nan,
             N5_edge=float(E[edge].sum() / E.sum()), first_t=float(t[0]), last_t=float(t[-1]),
             max_hg=float(hg.max()), n_ring811=int(np.sum((arr["f"] >= 8000) & (arr["f"] <= 11000))), big_hg=float(pk_hg[gbig]))
    return d, arr
