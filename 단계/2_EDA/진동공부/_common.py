"""진동공부 EDA 조각 공용 준비: 루트 찾기 · 데이터 · 색 · 저장 · 로그(UTF-8)"""
import os, sys

def find_root(start):
    p = os.path.abspath(start)
    while not os.path.isdir(os.path.join(p, "src")):
        q = os.path.dirname(p)
        if q == p:
            raise RuntimeError("src/ 폴더를 찾지 못함")
        p = q
    return p

ROOT = find_root(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.exp import load_train

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"   # 로그 눈금(10^-2)의 빼기 기호 깨짐 방지
plt.rcParams["mathtext.default"] = "rm"
FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"


def setup(here):
    fig_dir = os.path.join(here, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    X, y, names = load_train()
    names = [str(n) for n in names]
    y = np.asarray(y)
    log_lines = []

    def log(*a):
        log_lines.append(" ".join(str(v) for v in a))

    def save(fig, name):
        fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=90, bbox_inches="tight")
        plt.close(fig)

    def flush():
        with open(os.path.join(here, "result.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

    return X, y, names, log, save, flush


def pct_in_normal(v, y, i):
    """파일 i의 값이 정상 분포에서 몇 % 위치인가"""
    return float((v[y == 0] < v[i]).mean() * 100)


def plain_log(ax, axis="y"):
    """로그 눈금 숫자를 10^-2 대신 0.01처럼 쓴다 (한글 글꼴에서 빼기 기호 깨짐 방지)"""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())
