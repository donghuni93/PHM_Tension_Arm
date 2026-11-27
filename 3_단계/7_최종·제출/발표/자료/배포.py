"""팀원 배포용 (11-25): 흐름 설명 HTML(그림을 안에 넣은 파일 하나) + zip
실행: conda run -n qe-phm python "3_단계/7_최종·제출/발표/자료/배포.py"
출력: 4_outputs/배포/9조_텐션암_흐름.html · 4_outputs/배포/9조_텐션암_배포.zip (git 제외)
  zip = 흐름 HTML + 그림/(같은 그림 PNG, PPT에 바로 쓰기) + 발표 덱 초안 + 읽어보기.txt
페이지 글은 같은 폴더 배포_흐름.html({{img:키}} 자리)에서 고친다"""
import os, base64, datetime, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
F = os.path.join(HERE, "figs")
E62 = os.path.join(ROOT, "3_단계", "5_모델실험", "6_최종모델", "EXP-062_11kHz상한·그룹CV", "자료", "figs")
# 키 → (원본, zip 안 이름). 번호 = 페이지 순서
IMGS = {"rms": (os.path.join(F, "2_rms_chart.png"), "01_크기로는_안된다_RMS관리도.png"),
        "hits": (os.path.join(F, "5a_큰타격반복.png"), "02_큰타격반복.png"),
        "ring": (os.path.join(F, "5b_울림주파수.png"), "03_울림주파수.png"),
        "lock": (os.path.join(F, "6_박자_022.png"), "04_박자_022.png"),
        "e2": (os.path.join(F, "7a_E2_하중대비고주파.png"), "05_규칙E2_하중대비고주파.png"),
        "r83": (os.path.join(F, "7b_8.3k모드.png"), "06_규칙8.3k.png"),
        "structure": (os.path.join(F, "3_structure.png"), "07_모델구조.png"),
        "ladder": (os.path.join(F, "4_ladder.png"), "08_점수사다리.png"),
        "shrink": (os.path.join(F, "9_울림비율축소.png"), "09_울림비율축소.png"),
        "plane": (os.path.join(E62, "logistic_plane.png"), "10_최종모델_결합평면.png"),
        "honest": (os.path.join(F, "10_정직한점수.png"), "11_정직한점수.png"),
        "val": (os.path.join(E62, "val_판정.png"), "12_val판정.png"),
        "tone": (os.path.join(F, "12_고주파지속음.png"), "13_한계_고주파지속음.png")}
PPTX = os.path.join(HERE, "..", "team_project_9조_초안.pptx")
OUT = os.path.join(ROOT, "4_outputs", "배포"); os.makedirs(OUT, exist_ok=True)

html = open(os.path.join(HERE, "배포_흐름.html"), encoding="utf-8").read()
for k, (p, _) in IMGS.items():
    html = html.replace("{{img:%s}}" % k, "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode())
html = html.replace("{{date}}", datetime.date.today().isoformat())
assert "{{" not in html, "채우지 못한 자리: " + html[html.find("{{"):html.find("{{") + 30]
H = os.path.join(OUT, "9조_텐션암_흐름.html")
open(H, "w", encoding="utf-8").write(html)

README = """9조 텐션암 이상감지 — 배포 자료 ({date})

1. 9조_텐션암_흐름.html  ← 먼저 여세요 (더블클릭, 인터넷 없이 열림)
   문제 → 고장의 특징 → 모델 구조 → 발전 과정 → 최종 모델 → val → 한계 → 결정 모음 → 용어 순서.
   그림을 누르면 크게 보입니다.
2. 그림/  페이지에 나온 그림 원본(PNG). 번호 = 페이지 순서. PPT에 그대로 넣어도 됩니다.
3. 발표_덱_초안.pptx  발표 덱 초안. 아직 옛 모델(J-축소, F2 0.977 · val 9/40) 숫자라서
   최종 모델 숫자(F2 0.971 ± 0.022 · 재현율 0.987 · 오경보율 0.023 · val 10/40)로 바꿔야 합니다.

최종 모델 = EXP-062 구성 3. 코드 · 전체 기록은 저장소 3_단계/README.md에서 시작합니다.
""".format(date=datetime.date.today().isoformat())
Z = os.path.join(OUT, "9조_텐션암_배포.zip")
with zipfile.ZipFile(Z, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(H, "9조_텐션암_흐름.html")
    for k, (p, name) in IMGS.items():
        z.write(p, "그림/" + name)
    if os.path.exists(PPTX):
        z.write(PPTX, "발표_덱_초안.pptx")
    z.writestr("읽어보기.txt", README)
open(os.path.join(OUT, "_log.txt"), "w", encoding="utf-8").write(
    f"html {os.path.getsize(H) / 1e6:.1f} MB · zip {os.path.getsize(Z) / 1e6:.1f} MB · 그림 {len(IMGS)}개 · pptx {'있음' if os.path.exists(PPTX) else '없음'}\n")
