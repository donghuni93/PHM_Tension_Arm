# %% 발표 덱 (발표_목차.md) → team_project_9조_초안.pptx + 미리보기/slide-NN.png
# 슬라이드마다 요소 목록(명세) 하나를 만들고, 같은 명세로 PPTX와 미리보기 PNG를 함께 그린다.
# 이 PC에는 PowerPoint · LibreOffice가 없어서 미리보기는 PIL로 근사한다(맑은 고딕 실측 줄바꿈). 넘침은 경고로 출력.
# 수치는 파이프라인_설명.md · EXP-050 result.json과 같은 값. val 점수(11/26) · test 결과(11/27)는 【 】 칸으로 비워 둠
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

FIG = os.path.join(HERE, "figs")
OUT = os.path.join(HERE, "..", "team_project_9조_초안.pptx")   # 11-25: 덱 파일은 발표/ 바로 아래 (스크립트 · 그림은 자료/)
PRE = os.path.join(HERE, "미리보기")
os.makedirs(PRE, exist_ok=True)
W, H = 13.333, 7.5
FONT = "맑은 고딕"
NAVY, ACC, BLUE, INK, MUTED = "0B2A4F", "EB6834", "2A78D6", "1F2937", "6B7280"
TINT, TINT_A, WHITE, LINE = "EEF2F7", "FDF0EA", "FFFFFF", "D1D5DB"
TEAM = "권소린 · 박나영 · 신효섭 · 이동훈 · 한동훈"


# ---------- 명세 만들기 ----------
def T(x, y, w, h, paras, size=16, color=INK, bold=False, align="l", anchor="t", fill=None, line=None, radius=False, name=""):
    """paras = 문자열 또는 [문자열 | (문자열, {size, color, bold})] 목록. 한 항목 = 한 문단"""
    if isinstance(paras, str):
        paras = [paras]
    return dict(k="text", x=x, y=y, w=w, h=h, paras=paras, size=size, color=color, bold=bold, align=align,
                anchor=anchor, fill=fill, line=line, radius=radius, name=name)


def R(x, y, w, h, fill, line=None, radius=True, name="", dash=False):
    return dict(k="rect", x=x, y=y, w=w, h=h, fill=fill, line=line, radius=radius, name=name, dash=dash)


def I(path, x, y, w, h, name=""):
    """상자 (x, y, w, h) 안에 비율을 지켜 넣는다 (가운데 위쪽 정렬)"""
    p = os.path.join(FIG, path)
    iw, ih = Image.open(p).size
    s = min(w / iw, h / ih)
    ww, hh = iw * s, ih * s
    return dict(k="img", path=p, x=x + (w - ww) / 2, y=y, w=ww, h=hh, name=name or path)


def title(text):
    return T(0.6, 0.35, W - 1.2, 0.9, text, size=28, color=NAVY, bold=True, anchor="m", name="제목")


def footer(n):
    return [T(0.6, H - 0.45, 6, 0.3, "9조 · 텐션암 진동 이상감지", size=10, color=MUTED, name="꼬리말"),
            T(W - 1.6, H - 0.45, 1.0, 0.3, str(n), size=10, color=MUTED, align="r", name="쪽 번호")]


def card(x, y, w, h, head, body, fill=TINT, head_color=NAVY, size=15):
    return [R(x, y, w, h, fill, name=f"카드 {head}"),
            T(x + 0.25, y + 0.2, w - 0.5, 0.45, head, size=17, color=head_color, bold=True, name=f"카드 제목 {head}"),
            T(x + 0.25, y + 0.7, w - 0.5, h - 0.85, body, size=size, color=INK, name=f"카드 본문 {head}")]


S = []   # (요소 목록, 배경, 발표자 노트)

# 0 표지
S.append(([T(0.9, 1.7, 11.5, 0.5, "품질공학 팀 프로젝트 · 9조", size=18, color="AFC3DA"),
           T(0.9, 2.3, 11.5, 1.2, "텐션암 진동 이상감지", size=44, color=WHITE, bold=True),
           T(0.9, 3.5, 11.5, 0.8, "진동의 '크기'가 아니라 '형태'를 읽는 모델 — J-축소", size=24, color=WHITE),
           T(0.9, 5.4, 11.5, 0.5, TEAM, size=16, color="AFC3DA")],
          NAVY, "9조 발표를 시작하겠습니다. 저희는 텐션암 진동 2초로 정상과 고장을 가르는 모델을 만들었고, 핵심은 진동의 크기가 아니라 형태를 읽는 것입니다."))

# 1 결론
kpi = [("중첩 CV F2", "0.977", "± 0.010 (5회 반복)"), ("재현율", "1.000", "고장 30개 모두 검출"),
       ("오경보율", "0.030", "정상 120개 중 약 3.6개"), ("val 고장 판정", "9 / 40", "1차 제출 (11/26)")]
els = [title("결론: 고장 30개를 모두 잡고 오경보는 3 % — 중첩 CV F2 0.977")]
for i, (h_, v, s_) in enumerate(kpi):
    x = 0.6 + i * 3.08
    els += [R(x, 1.75, 2.85, 2.6, TINT, name=f"KPI {h_}"),
            T(x + 0.3, 1.95, 2.4, 0.45, h_, size=16, color=MUTED),
            T(x + 0.3, 2.45, 2.4, 1.0, v, size=44, color=ACC if i == 0 else NAVY, bold=True, anchor="m"),
            T(x + 0.3, 3.6, 2.4, 0.5, s_, size=14, color=MUTED)]
els += [T(0.6, 4.85, 12.1, 0.5, "어떻게 이 점수가 나왔나 — 부품을 하나씩 빼 보며 거꾸로 설명합니다", size=20, color=NAVY, bold=True),
        T(0.6, 5.45, 12.1, 1.2, ["평가: 학습 150개(정상 120 · 고장 30)의 중첩 교차검증 (바깥 5등분 × 안쪽 4등분, 난수 5회). 점수용 파일은 학습 · 문턱 선택에 쓰지 않음",
                                 "F2 = 미탐 1건을 오경보 4건만큼 무겁게 보는 지표"], size=15, color=INK)]
S.append((els + footer(1), WHITE, "결론부터 말씀드리면, 최종 모델은 학습 데이터 중첩 교차검증에서 F2 0.977, 고장 30개를 모두 잡고 오경보율은 3%입니다. val에서는 40개 중 9개를 고장으로 판정해 제출했습니다. 지금부터 이 모델이 왜 이런 모양인지 부품을 하나씩 빼 보며 거꾸로 설명하겠습니다."))

# 2 문제
S.append(([title("진동 크기로는 고장 28 / 30을 놓친다 (RMS 관리도 F2 0.080)"),
           I("2_rms_chart.png", 0.6, 1.3, 12.1, 3.95),
           *card(0.6, 5.35, 3.9, 1.4, "데이터", "2초 × 25 kHz · 정상 120 · 고장 30", size=14),
           *card(4.72, 5.35, 3.9, 1.4, "평가 F2", "미탐 1건 = 오경보 4건", size=14),
           *card(8.85, 5.35, 3.9, 1.4, "결론", "크기가 아니라 형태에서 다르다", fill=TINT_A, head_color=ACC, size=14),
           *footer(2)], WHITE,
          "가장 단순한 방법은 진동 크기, RMS로 관리도를 그리는 것입니다. 정상 120개로 관리한계를 잡으면 고장 30개 중 28개가 정상 범위 안에 들어가 F2는 0.08입니다. 고장은 크기가 아니라 형태에서 다르다는 것이 출발점입니다."))

# 3 구조
S.append(([title("최종 모델 J-축소: 전문가 3명 + 빈틈을 막는 물리 규칙 2개"),
           I("3_structure.png", 1.2, 1.4, 10.9, 5.1),
           T(0.6, 6.55, 12.1, 0.45, "전문가 점수는 정상 기준 z로 맞춘 뒤 로지스틱 결합(모수 4개) · 규칙은 고장 쪽으로만 올림 → 규칙 오류의 비용은 오경보로 제한", size=14, color=MUTED),
           *footer(3)], WHITE,
          "최종 모델의 구조입니다. 데이터에서 학습한 전문가 세 명, ET·박자·AE가 점수를 내고 로지스틱으로 합칩니다. 여기에 물리 가설에서 나온 규칙 두 개를 OR로 붙였습니다. 규칙은 고장 쪽으로만 올리기 때문에 틀려도 비용은 오경보입니다."))

# 4 사다리
S.append(([title("부품을 하나씩 빼 보면, 부품마다 맡은 고장이 있다"),
           I("4_ladder.png", 0.6, 1.4, 12.1, 5.0),
           T(0.6, 6.5, 12.1, 0.45, "막대는 아래에서 위로 부품을 더한 순서 · 모든 점수는 같은 중첩 교차검증 · 출처 EXP-040 · 043 · 044 · 050", size=14, color=MUTED),
           *footer(4)], WHITE,
          "부품을 하나씩 빼 보면 각자 맡은 일이 보입니다. 맨 위 제출 모델에서 AE를 빼면 0.959, 8.3k 규칙을 빼면 0.933, E2 규칙을 빼면 0.907까지 내려갑니다. 이제 아래 부품부터 하나씩 보겠습니다."))

# 5 ET · H 피처
S.append(([title("ET · 충격 피처: 고장은 큰 타격이 반복되고, 다른 주파수로 운다"),
           T(0.6, 1.4, 6.0, 0.4, "① 큰 타격의 반복 (2위 ÷ 1위 충격)", size=16, color=NAVY, bold=True),
           I("5a_큰타격반복.png", 0.6, 1.85, 6.0, 2.7),
           T(6.75, 1.4, 6.0, 0.4, "② 충격이 울리는 주파수", size=16, color=NAVY, bold=True),
           I("5b_울림주파수.png", 6.75, 1.85, 6.0, 2.7),
           *card(0.6, 4.85, 12.15, 1.85, "H 피처 15열 → ExtraTrees (나무 300그루)",
                 ["1 kHz 고역통과로 하중 흔들림을 지우고 충격만 남김 → 충격 수 · 2위/1위 · 첨도 · 울림 비율 · 파워 몫 · 봉우리",
                  "대역은 라벨을 보기 전에 정한 옥타브 4칸만 사용 · 진동 크기는 넣지 않음"], size=14),
           *footer(5)], WHITE,
          "첫 번째 전문가 ET는 충격의 형태를 봅니다. 정상은 큰 충격이 한 번 있고 나머지는 작습니다. 고장은 비슷한 크기의 충격이 여러 번 반복됩니다. 또 충격 뒤 울림을 보면 정상은 4에서 8 kHz, 본체 모드로 울리고 고장은 2에서 4 kHz와 8 kHz 위로 웁니다. 어느 부품이 울렸는지가 다르다는 뜻입니다."))

# 6 박자
S.append(([title("박자 z: 고장 022는 하중이 흔들리는 주기에 맞춰 덜컥거린다"),
           I("6_박자_022.png", 0.6, 1.4, 8.7, 4.9),
           *card(9.6, 1.4, 3.15, 2.3, "왜 필요했나", "022는 에너지 · 스펙트럼 · 충격 크기가 모두 정상 범위", size=14),
           *card(9.6, 3.9, 3.15, 2.4, "어떻게 재나", "느린 흔들림의 주기 T를 찾고, T 떨어진 곳에 짝 충격이 있는 비율 → 정상 기준 z", size=14),
           *footer(6)], WHITE,
          "두 번째 전문가는 박자입니다. 고장 022는 다른 모든 지표가 정상 범위인데, 충격이 하중 흔들림 주기에 맞춰 반복됩니다. 이 주기 잠금 정도를 정상 분포 기준 z로 바꿔 결합에 넣었습니다."))

# 7 규칙
S.append(([title("물리 규칙 2개가 통계 모델이 못 보는 017 · 002를 잡는다"),
           T(0.6, 1.4, 6.0, 0.4, "E2 · 하중 대비 고주파 (유격)", size=16, color=NAVY, bold=True),
           I("7a_E2_하중대비고주파.png", 0.6, 1.85, 6.0, 2.4),
           T(6.75, 1.4, 6.0, 0.4, "8.3k · 8.3 kHz 모드가 크고 넓게 솟음", size=16, color=NAVY, bold=True),
           I("7b_8.3k모드.png", 6.75, 1.85, 6.0, 2.6),
           *card(0.6, 4.75, 6.0, 1.95, "E2 → 004 · 011 · 017 · 019",
                 "마모로 생긴 틈을 하중이 걸릴 때마다 건너 충돌\n문턱 = 정상의 99 % 분위 (라벨을 쓰지 않음) · 정상 0개", size=14),
           *card(6.75, 4.75, 6.0, 1.95, "8.3k → 002 · 003 · 006 · 010 · 029 · 030",
                 "충격과 에너지는 정상 범위인 002 · 029를 잡음\n충격성 K > 3 조건: 지속음(val 001)은 제외\n주의: 대역 · 폭 · 돌출 문턱은 라벨을 본 뒤 정한 사후값", fill=TINT_A, head_color=ACC, size=14),
           *footer(7)], WHITE,
          "통계 모델 점수가 정상 범위에 머무는 고장은 규칙으로 막았습니다. E2는 하중에 비해 고주파 덜컥거림이 큰 파일로, 정상 분포의 99% 분위를 넘으면 고장입니다. 8.3k 규칙은 8.3 kHz 모드가 크고 넓게 솟는 파일입니다. 여기에 대역 에너지가 충격으로 몰렸는지를 보는 충격성 조건을 붙여, 2초 내내 이어지는 지속음은 제외합니다. 다만 8.3k의 문턱은 라벨을 본 뒤 정한 값이라 점수가 낙관적이라는 점을 뒤에서 따로 보여 드리겠습니다. 두 규칙을 빼면 F2는 0.916으로, 002와 017을 놓칩니다."))

# 8 AE
S.append(([title("AE는 오경보를 줄인다: 오경보율 0.047 → 0.030"),
           I("8_AE_평면.png", 0.6, 1.35, 7.9, 5.3),
           *card(8.8, 1.35, 3.95, 2.35, "AE = 정상 스펙트럼의 관리도", "정상 파일의 스펙트럼만 학습해 복원 오차를 점수로 (PCA 잔차 관리도의 비선형판)", size=14),
           *card(8.8, 3.9, 3.95, 2.75, "그림 읽는 법", "점선 오른쪽 = ET만 쓰면 고장\n파란 배경 = AE를 더한 결합에서 정상\n→ 065 · 066 · 110이 '스펙트럼은 정상'이라 내려감", size=14),
           *footer(8)], WHITE,
          "세 번째 전문가 AE는 정상 스펙트럼만 학습한 다변량 관리도입니다. 가로축은 ET 점수, 세로축은 AE 점수입니다. 점선 오른쪽은 ET만 쓰면 고장인데, 배경이 파란 곳은 AE를 더하면 정상으로 돌아옵니다. 충격은 있지만 스펙트럼은 정상인 파일을 내려 주는 역할이고, 오경보율이 0.047에서 0.030으로 줄었습니다."))

# 9 축소
S.append(([title("EXP-050 울림 비율 축소: 점수는 그대로, val 판정 흔들림은 줄었다"),
           I("9_울림비율축소.png", 0.6, 1.35, 12.1, 3.85),
           *card(0.6, 5.35, 12.15, 1.45, "베이즈 축소 (α = 3, 실행 전 고정)",
                 "충격이 2–3개뿐인 파일은 울림 비율이 한두 개로 0 ↔ 1을 오간다 → 정상 쪽으로 당김 (표본이 적은 로트의 불량률을 공정 평균 쪽으로 당기는 것과 같음)", size=14),
           *footer(9)], WHITE,
          "마지막 손질이 EXP-050입니다. 충격이 두세 개뿐인 파일은 울림 비율이 충격 한두 개로 크게 흔들립니다. 그래서 충격이 적을수록 비율을 정상 쪽으로 당겼습니다. F2는 0.977로 같지만, 결합기를 다시 뽑았을 때 val 고장 판정 수의 범위가 7에서 20개에서 7에서 13개로 좁아졌습니다."))

# 10 정직한 점수
S.append(([title("점수는 정직하게: 낙관 요인을 따로 보여 준다"),
           I("10_정직한점수.png", 0.6, 1.35, 12.1, 3.1),
           *card(0.6, 4.75, 3.9, 1.95, "중첩 교차검증", "점수용 파일은 학습 · 정상 기준 · 문턱 선택에 쓰지 않음", size=14),
           *card(4.72, 4.75, 3.9, 1.95, "사후 문턱 표시", "8.3k 규칙을 빼면 0.950 → 새 데이터 기대 0.950–0.977", fill=TINT_A, head_color=ACC, size=14),
           *card(8.85, 4.75, 3.9, 1.95, "val은 확인용", "val 결과로 문턱을 바꾸지 않음 · 학습 그대로 제출", size=14),
           *footer(10)], WHITE,
          "점수를 어떻게 쟀는지도 중요합니다. 같은 폴드로 문턱까지 고르는 25년 방식을 우리 데이터에 재현하면 0.538이었습니다. 저희는 중첩 교차검증으로 쟀고, 라벨을 보고 정한 8.3k 문턱을 빼면 0.950입니다. 그래서 새 데이터에서는 0.95에서 0.977 사이를 기대합니다. 학습 40개를 val로 가정한 리허설 3회도 0.93에서 0.976이었습니다."))

# 11 val
S.append(([title("val 40개 중 32개는 모델 25개가 만장일치로 같은 판정"),
           I("11a_val확실도.png", 0.6, 1.35, 8.6, 3.1),
           *card(9.45, 1.35, 3.3, 3.1, "흔들리는 파일", "025: 여유 -0.09, 경계\n018: AE 배경 차이\n001: 미끄러지는 지속음 → 8.3k 규칙 제외(정상)", size=14),
           *card(0.6, 4.7, 5.9, 2.0, "val 측정 조건 차이", "val은 배경 3–5 kHz가 학습보다 약 +10.9 dB 높음 · 재보정해도 판정이 거의 같아(10 → 9, 충격성 조건 도입 전 값) 그대로 제출", size=14),
           R(6.75, 4.7, 6.0, 2.0, WHITE, line=ACC, name="val 점수 칸", dash=True),
           T(7.0, 4.9, 5.5, 0.45, "1차 val 점수", size=17, color=ACC, bold=True),
           T(7.0, 5.45, 5.5, 1.1, "【11/26 공개 후 기입】 F2 · 맞힌 고장 수 · 오경보 수", size=15, color=MUTED),
           *footer(11)], WHITE,
          "val 40개에 대해, 학습 분할만 바꾼 모델 25개로 판정해 보면 32개는 모두 같은 판정입니다. 흔들리는 파일은 025, 018 정도이고 이유도 파일별로 확인했습니다. 001은 8.3 kHz 대역을 미끄러지며 지나가는 지속음이라 8.3k 규칙의 충격성 조건에서 제외되어 정상으로 판정했습니다. 1차 val 점수는 공개되면 여기에 채우겠습니다."))

# 12 한계
S.append(([title("남은 오경보: 2초 내내 이어지는 고주파 지속음 정상 3개는 정보 한계"),
           I("12_고주파지속음.png", 0.6, 1.35, 7.9, 5.4),
           *card(8.8, 1.35, 3.95, 2.6, "남는 오류", "미탐 없음\n오경보 116 (5회 중 5회) · 025 · 059 · 065 (각 3회)", size=14),
           *card(8.8, 4.15, 3.95, 2.6, "확인 못 한 것", "지속음 출처: 구조 공진 / 일정 속도 부품 / 계측 간섭\n노면 · 속도 · 교체 부품 정보 없음", fill=TINT_A, head_color=ACC, size=14),
           *footer(12)], WHITE,
          "남은 오경보는 대부분 2초 내내 좁은 고주파 지속음이 있는 정상 파일입니다. 어떤 피처나 모델로도 내려가지 않아 정보 한계로 두었습니다. 지속음의 출처는 데이터만으로는 확인하지 못했습니다."))

# 13 test · 개선
S.append(([title("test 결과와 개선 방향"),
           R(0.6, 1.5, 3.9, 3.2, WHITE, line=ACC, name="test 결과 칸", dash=True),
           T(0.85, 1.7, 3.4, 0.45, "test 결과", size=17, color=ACC, bold=True),
           T(0.85, 2.3, 3.4, 2.2, "【11/27 test 공개 후 기입】\n학습은 다시 하지 않고 저장한 모델로 한 번만 판정", size=15, color=MUTED),
           *card(4.72, 1.5, 3.9, 3.2, "① 정직한 J", "8.3k 문턱을 폴드 안에서 다시 골라 사후 몫을 직접 측정", size=15),
           *card(8.85, 1.5, 3.9, 3.2, "② 원인 정보", "노면 · 속도 · 교체 부품 기록이 있으면 지속음 · 유격 가설을 확인할 수 있음", size=15),
           *footer(13)], WHITE,
          "test 결과는 공개 후 채우겠습니다. 개선 방향은 두 가지입니다. 사후 문턱인 8.3k를 폴드 안에서 다시 골라 낙관 몫을 직접 재는 것, 그리고 노면이나 교체 부품 같은 원인 정보로 물리 가설을 확인하는 것입니다."))

# 14 마무리
S.append(([T(0.9, 2.4, 11.5, 1.2, "감사합니다", size=44, color=WHITE, bold=True),
           T(0.9, 3.6, 11.5, 0.8, "질문을 받겠습니다", size=24, color=WHITE),
           T(0.9, 5.4, 11.5, 0.5, "9조 · " + TEAM, size=16, color="AFC3DA")], NAVY, "감사합니다. 질문 받겠습니다."))

# 부록 1 닫힌 질문
rows = [("질문", "결과", "근거 실험"),
        ("모델을 바꾸면 나아지나", "효과 없음 (RF · 부스팅 · SVM 등)", "EXP-019 · 037 · PHO-005"),
        ("결합기를 비선형으로", "52칸 중 기준을 sd 넘게 넘은 칸 0 (최선 SVM 0.969)", "EXP-045 · 052"),
        ("피처 선택", "개선 없음", "EXP-027"),
        ("이상치 정상을 학습에서 빼면", "0.961로 하락", "EXP-046"),
        ("배깅", "+0.005 (sd 안), val 고장 증가", "EXP-051"),
        ("AE 대신 정상 PCA", "0.970으로 낮음", "EXP-049")]
S.append(([title("부록 · 시도했지만 채택하지 않은 것"), dict(k="table", x=0.6, y=1.5, w=12.1, rows=rows, colw=[3.6, 5.6, 2.9]), *footer("A1")], WHITE,
          "질의 대비: 모델 교체, 비선형 결합, 피처 선택, 학습 정상 배제, 배깅은 모두 같은 중첩 교차검증에서 기준을 표준편차 넘게 넘지 못했습니다."))

# 부록 2 val 평면
S.append(([title("부록 · val 40개를 학습과 같은 결합 경계 위에 놓으면"), I("11b_val평면.png", 2.4, 1.35, 8.5, 5.4), *footer("A2")], WHITE,
          "질의 대비: val을 학습과 같은 경계 위에 놓은 그림입니다. 025는 경계 바로 왼쪽, 018은 ET 경계 근처, 001은 결합 점수가 낮고, 지속음이라 8.3k 규칙의 충격성 조건에서도 제외되어 정상입니다."))


# ---------- PPTX ----------
def rgb(h):
    return RGBColor.from_string(h)


def set_font(run, size, color, bold):
    run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = rgb(color); run.font.name = FONT
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        e = rPr.find(qn(tag))
        if e is None:
            e = rPr.makeelement(qn(tag), {}); rPr.append(e)
        e.set("typeface", FONT)


def add_text(sl, e):
    shp = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if e["radius"] else MSO_SHAPE.RECTANGLE,
                              Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"])) if e["fill"] else \
        sl.shapes.add_textbox(Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
    if e["fill"]:
        shp.fill.solid(); shp.fill.fore_color.rgb = rgb(e["fill"]); shp.line.fill.background()
    shp.name = e["name"] or "텍스트"
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE}[e["anchor"]]
    first = True
    for p_ in e["paras"]:
        txt, o = (p_, {}) if isinstance(p_, str) else p_
        for line in txt.split("\n"):
            para = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
            para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[e["align"]]
            para.space_after = Pt(4)
            r = para.add_run(); r.text = line
            set_font(r, o.get("size", e["size"]), o.get("color", e["color"]), o.get("bold", e["bold"]))


def add_rect(sl, e):
    shp = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if e["radius"] else MSO_SHAPE.RECTANGLE,
                              Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
    shp.name = e["name"] or "도형"
    if e["radius"]:
        shp.adjustments[0] = 0.06
    shp.fill.solid(); shp.fill.fore_color.rgb = rgb(e["fill"])
    if e["line"]:
        shp.line.color.rgb = rgb(e["line"]); shp.line.width = Pt(1.5)
        if e["dash"]:
            from pptx.enum.dml import MSO_LINE
            shp.line.dash_style = MSO_LINE.DASH
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False


def add_table(sl, e):
    rows = e["rows"]
    tb = sl.shapes.add_table(len(rows), len(rows[0]), Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(0.55 * len(rows))).table
    for j, cw in enumerate(e["colw"]):
        tb.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = tb.cell(i, j); cell.text = ""
            cell.fill.solid(); cell.fill.fore_color.rgb = rgb(NAVY if i == 0 else (TINT if i % 2 else WHITE))
            r = cell.text_frame.paragraphs[0].add_run(); r.text = v
            set_font(r, 14, WHITE if i == 0 else INK, i == 0)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
blank = prs.slide_layouts[6]
for els, bg, note in S:
    sl = prs.slides.add_slide(blank)
    sl.background.fill.solid(); sl.background.fill.fore_color.rgb = rgb(bg)
    for e in els:
        if e["k"] == "text":
            add_text(sl, e)
        elif e["k"] == "rect":
            add_rect(sl, e)
        elif e["k"] == "img":
            p = sl.shapes.add_picture(e["path"], Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            p.name = e["name"]
        elif e["k"] == "table":
            add_table(sl, e)
    sl.notes_slide.notes_text_frame.text = note
prs.core_properties.title = "텐션암 진동 이상감지 — 9조"
prs.save(OUT)

# ---------- 미리보기 (PIL 근사) + 넘침 검사 ----------
DPI = 96
FR = {False: "C:/Windows/Fonts/malgun.ttf", True: "C:/Windows/Fonts/malgunbd.ttf"}
_fc = {}


def font(size, bold):
    k = (size, bold)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FR[bold], int(round(size * DPI / 72)))
    return _fc[k]


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def wrap(text, f, wpx):
    out = []
    for line in text.split("\n"):
        cur = ""
        for ch in line:
            if f.getlength(cur + ch) <= wpx:
                cur += ch
            else:
                cut = cur.rfind(" ")
                if cut > 0 and len(cur) - cut < 12:
                    out.append(cur[:cut]); cur = cur[cut + 1:] + ch
                else:
                    out.append(cur); cur = ch
        out.append(cur)
    return out


warn = []
for n, (els, bg, _) in enumerate(S):
    im = Image.new("RGB", (int(W * DPI), int(H * DPI)), hexrgb(bg))
    d = ImageDraw.Draw(im)
    px = lambda v: int(round(v * DPI))
    for e in els:
        box = (px(e["x"]), px(e["y"]), px(e["x"] + e["w"]), px(e["y"] + e.get("h", 0)))
        if e["k"] == "rect":
            d.rounded_rectangle(box, radius=px(0.12) if e["radius"] else 0, fill=hexrgb(e["fill"]),
                                outline=hexrgb(e["line"]) if e["line"] else None, width=2)
        elif e["k"] == "img":
            im.paste(Image.open(e["path"]).convert("RGB").resize((box[2] - box[0], box[3] - box[1])), box[:2])
        elif e["k"] == "table":
            y0 = px(e["y"])
            for i, row in enumerate(e["rows"]):
                x0 = px(e["x"])
                for j, v in enumerate(row):
                    cw = px(e["colw"][j])
                    d.rectangle((x0, y0, x0 + cw, y0 + px(0.55)), fill=hexrgb(NAVY if i == 0 else (TINT if i % 2 else WHITE)))
                    f = font(14, i == 0)
                    if f.getlength(v) > cw - 10:
                        warn.append(f"{n}: 표 칸 넘침 '{v}'")
                    d.text((x0 + 6, y0 + 8), v, font=f, fill=hexrgb(WHITE if i == 0 else INK))
                    x0 += cw
                y0 += px(0.55)
        elif e["k"] == "text":
            if e["fill"]:
                d.rounded_rectangle(box, radius=px(0.12) if e["radius"] else 0, fill=hexrgb(e["fill"]))
            lines = []
            for p_ in e["paras"]:
                txt, o = (p_, {}) if isinstance(p_, str) else p_
                f = font(o.get("size", e["size"]), o.get("bold", e["bold"]))
                for ln in wrap(txt, f, box[2] - box[0]):
                    lines.append((ln, f, o.get("color", e["color"])))
            lh = [int(f.size * 1.2) for _, f, _ in lines]
            total = sum(lh) + int(len(e["paras"]) * 4 * DPI / 72)
            y = box[1] if e["anchor"] == "t" else box[1] + (box[3] - box[1] - total) // 2
            for (ln, f, c), h_ in zip(lines, lh):
                xw = f.getlength(ln)
                x = box[0] if e["align"] == "l" else (box[2] - xw if e["align"] == "r" else box[0] + (box[2] - box[0] - xw) / 2)
                d.text((x, y), ln, font=f, fill=hexrgb(c)); y += h_
            if total > box[3] - box[1] + 2:
                warn.append(f"{n}: 글 넘침 [{e['name'] or e['paras'][0][:15]}] 필요 {total / DPI:.2f}in > 상자 {e['h']:.2f}in")
        if e["k"] in ("text", "rect", "img") and (box[2] > im.width + 1 or box[3] > im.height + 1):
            warn.append(f"{n}: 슬라이드 밖 [{e.get('name')}]")
    im.save(os.path.join(PRE, f"slide-{n:02d}.png"))

open(os.path.join(PRE, "_점검.txt"), "w", encoding="utf-8").write(f"슬라이드 {len(S)}장 → {os.path.basename(OUT)}\n" + ("\n".join(warn) if warn else "넘침 없음") + "\n")
