# 공진(고유진동수) 이동·강성/감쇠 변화·스펙트럼 피크 피처 기반 손상 감지 — 텐션암 적용 관점

> 조사일 2026-11-14. 팀 기존 목록(Balaji&Sugumaran 2023, Karthikeyan 2024, Antoni 2006/2007, Jardine 2006, Zhang 2020, Andreaus 2007, Shah 2024, Sai 2023, Hoang&Kang 2019, Lee 2017 부싱 FEM, KTH 부싱 학위논문)은 제외했다.
> 검증 상태: **verified** = 서지(제목·저자·저널·DOI)를 색인/출판사/기관 저장소에서 확인 / **partially** = 서지 일부 또는 2차 출처만 확인 / **unverified** = 존재는 보이나 서지·내용 미확인. 본문 전문은 대부분 열람하지 못했다(초록·색인·2차 인용 기반). 수치는 출처가 준 것만 적었다.
> OA = 무료 전문 접근 가능.

## 1. 고유진동수 이동·모달 파라미터 기반 손상 감지 고전 리뷰 — 민감도와 한계

### Takeaway
고유진동수 변화는 측정이 쉽고 정확하지만, (a) 손상에 둔감하고(작은 이동), (b) 위치·심각도를 유일하게 정하지 못하며, (c) 환경·운전 변화가 같은 크기 이상의 이동을 만든다는 것이 고전 리뷰들의 공통 결론이다. 결국 "모달 변화 → 통계적 패턴인식(이상치 판정)" 문제로 귀결된다.

### Cited Findings
- **[S1] Salawu, O.S. (1997). Detection of structural damage through changes in frequency: a review. *Engineering Structures*, 19(9), 718–723. DOI 10.1016/S0141-0296(96)00149-6** — verified (서지), 전문 미열람.
  - 고유진동수 기반 손상 감지 방법 약 65편을 리뷰하고, 진동 모니터링 적용을 제한하는 요인을 논의. 진동수 변화만으로는 손상 위치·심각도의 유일한 식별이 불충분할 수 있다고 결론 — [Scilit](https://www.scilit.net/publications/3d6369086527aa93ee797f5a6f9f582f); 후속 리뷰의 요약 [DYNA 연대기 리뷰](https://revistas.unal.edu.co/index.php/dyna/article/view/91693)
  - 적용: 우리 과제는 "위치" 아닌 "정상/고장 이진 판정"이므로 Salawu의 핵심 한계(위치 비유일성)는 덜 치명적. 단, **진동수 하나(8.25 vs 8.6 kHz)만으로 판정하지 말 것**의 근거.
- **[S2] Doebling, S.W., Farrar, C.R., Prime, M.B. (1998). A summary review of vibration-based damage identification methods. *The Shock and Vibration Digest*, 30(2), 91–105. DOI 10.1177/058310249803000201** — verified. OA(LANL 사본) — [LANL PDF](https://public.lanl.gov/prime/doebling_svd.pdf), [Scilit](https://scilit.com/publications/91a5f9a5c3504eddc687b62f555c8c47)
  - 방법을 손상 감지 수준(존재/위치/크기), 모델 기반/비모델, 선형/비선형으로 분류. 모달 특성은 "데이터 압축"이라 상태 정보 일부를 잃으며, 결국 패턴인식 문제로 귀결된다고 지적 — 같은 출처
  - 1996년 LANL 장편 리뷰(LA-13070-MS로 알려짐)의 요약판 — 보고서 번호는 기억 기반, **unverified**.
  - 적용: PSD 전체(또는 대역 비율)를 쓰는 접근이 "피크 하나로 압축"보다 정보 손실이 적다는 논거. 피크 피처는 해석용, 분류는 패턴 전체로.
- **[S3] Fan, W., Qiao, P. (2011). Vibration-based damage identification methods: a review and comparative study. *Structural Health Monitoring*, 10(1), 83–111. DOI 10.1177/1475921710365419** — verified (서지) — [WSU 저장소](https://rex.libraries.wsu.edu/esploro/4_outputs/journalArticle/Vibration-based-Damage-Identification-Methods-A-Review/99900547889601842), [Structurae](https://structurae.net/fr/litterature/article-de-revue/vibration-based-damage-identification-methods-a-review-and-comparative-study)
  - 방법을 고유진동수 기반 / 모드형상 기반 / 곡률 모드형상 기반 / 결합형으로 분류, 보·판 구조에서 5개 알고리즘 비교. 모드형상 계열은 대부분 위치 추정에 초점 — 같은 출처
  - 적용: 우리는 가속도계 1개 → 모드형상 계열은 **원천적으로 불가**. 남는 것은 진동수·감쇠·진폭(스펙트럼) 계열뿐임을 발표에서 명시할 근거.
- **[S4] Peeters, B., De Roeck, G. (2001). One-year monitoring of the Z24-Bridge: environmental effects versus damage events. *Earthquake Engng Struct. Dyn.*, 30(2), 149–171. DOI 10.1002/1096-9845(200102)30:2<149::AID-EQE1>3.0.CO;2-Z** — verified (서지) — [Structurae](https://structurae.net/en/literature/journal-article/one-year-monitoring-of-the-z24-bridge-environmental-effects-versus-damage-events), [Scilit](https://scilit.net/publications/3062b8d0cebd475ec9ff49c1ef3a777c)
  - 정상 변동의 지배 요인은 온도(습도·바람도 기여). 건전 상태 데이터로 "고유진동수–온도" 블랙박스 모델을 학습하고, 신뢰구간을 벗어나면 손상으로 판정. 관련 장(Strauss/De Roeck)에서는 환경 영향 제거가 필수이며 고유진동수 감소가 약 1%를 넘으면 강성 저하 손상을 검출할 수 있었다고 보고 — [CTResources 장](https://www.ctresources.info/csets/chapter.html?id=304) (2차 기록, **partially**)
  - 적용: 8.6→8.25 kHz는 약 4% 이동(f∝√k 이면 강성 약 8% 변화에 해당). 교량 기준 1% 검출 한계보다 크지만, **고무 부싱은 온도 민감도가 강철보다 훨씬 커서** 같은 판단을 그대로 옮길 수 없다(아래 3절).

### Inferences
- 우리 데이터의 "두 혹이 동시에 있는 파일"은 단일 모드의 단순 좌측 이동과 맞지 않는다. 리뷰들이 언급하는 대안 기전—모드 분리(대칭 깨짐으로 중근 모드가 갈라짐), 모드 교차/비어링, 손상으로 인한 새 국부 모드—이 더 부합할 수 있다. (리뷰 본문에서 이 항목을 직접 확인하지 못했으므로 일반 진동 지식 기반 추론이다.)
- 따라서 H "같은 모드의 좌측 이동"보다 "8.2–8.3 kHz 대역에 **새 에너지가 나타남**"으로 가설을 재정식화하고, 피처는 피크 위치보다 **대역 에너지 비(8.15–8.35 kHz / 8.5–8.8 kHz)** 및 각 대역의 돌출도로 잡는 편이 덜 취약하다.
- 단, 이 대역은 라벨을 본 뒤 고른 것이므로 CLAUDE.md 원칙 3에 따라 **중첩 CV 내부에서 대역을 자동 선택**하거나, 사전 고정된 넓은 대역 격자를 쓰는 형태로만 평가해야 한다.

### Gaps
- Salawu·Doebling 본문의 정량 수치(예: 손상 크기 대비 진동수 변화율 표)는 전문 미열람으로 확인 못함.
- Doebling et al. 1996 LANL 원보고서 번호·URL 미확인.

## 2. 환경·운전 변동(EOV)에 의한 공진 이동과 정규화 방법

### Takeaway
온도·하중·경계조건이 고유진동수를 손상만큼(또는 그 이상) 움직이므로, "정상 변동 모델을 학습해 그 잔차로 판정"하는 데이터 정규화가 표준이다. 변동 원인을 측정하지 못하면(우리 경우) PCA/잠재변수로 정상 변동 방향을 제거하고 잔차의 이상도를 쓴다.

### Cited Findings
- **[S5] Sohn, H. (2007). Effects of environmental and operational variability on structural health monitoring. *Phil. Trans. R. Soc. A*, 365(1851), 539–560. DOI 10.1098/rsta.2006.1935** — verified (서지) — [Scilit](https://www.scilit.net/publications/3e47b85592daebf23cc75df3a81dbd66), [KAIST 저장소](https://koasas.kaist.ac.kr/handle/10203/18816?mode=full), OA 가능성 있는 LANL 사본 [OSTI](https://www.osti.gov/servlets/purl/976103)
  - 실구조물에서 보고된 환경·운전 영향 리뷰 + 데이터 정규화(손상 민감 피처에서 환경·운전 유도 추세 제거) 연구 동향 정리 — 같은 출처; 정규화 정의는 [Cross et al. 후속 논문 PDF](https://eprints.whiterose.ac.uk/76226/1/Full_Cross.pdf)가 Sohn 2007을 인용하며 기술
- **[S6] Yan, A.-M., Kerschen, G., De Boe, P., Golinval, J.-C. (2005). Structural damage diagnosis under varying environmental conditions — Part I: A linear analysis / Part II: Local PCA for non-linear cases. *Mechanical Systems and Signal Processing* 19 (2005).** — **partially** (Part II의 MSSP 2005년 7월 게재 확인, DOI·쪽수 미확인 → 기재하지 않음) — [ORBi Part II](https://orbi.uliege.be/handle/2268/18986), [Part I 색인](https://data.zhenzhi.com/details/1/891101014)
  - 환경 변수를 측정하지 않고도 PCA로 정상 변동 부분공간을 학습, 그 밖의 잔차(재구성 오차)로 novelty 판정. 건전 상태에서는 잔차가 작고 손상 시 유의하게 증가. 비선형 환경 영향에는 local PCA(군집별 PCA) — 같은 출처
  - 적용: 우리는 온도·예압·가진 정보가 **없다** → Yan 방식이 가장 직접적. 정상 120개 PSD(로그, >200 Hz 정규화)에 PCA → 재구성 오차 / Hotelling T²·Q 통계량을 이상점수로 쓰는 one-class 기준선을 만들 수 있다. 정상 군 내에 8.6 kHz 혹 유무 같은 "정상 하위 군집"이 있으면 local PCA(군집별)가 맞다.
- **[S7] Cross, E.J., Worden, K., Chen, Q. (2011). Cointegration: a novel approach for the removal of environmental trends in structural health monitoring data. *Proc. R. Soc. A*. DOI 10.1098/rspa.2011.0023** — verified (제목·저자·DOI), 권·쪽 미확인. OA 사본 — [White Rose PDF](https://eprints.whiterose.ac.uk/76226/1/Full_Cross.pdf)
  - 비정상(non-stationary) 피처 시계열들을 선형결합해 정상(stationary) 잔차를 만들고, 그 잔차를 환경·운전 조건에 무관한 손상 민감 피처로 사용 — 같은 출처
  - 적용: **공적분 자체는 시계열(시간 순서) 전제**라 우리처럼 순서 없는 독립 파일에는 직접 적용 불가. 다만 "여러 피크 진동수가 공통 요인(온도/예압)으로 함께 움직인다면 그 선형결합(예: f₂/f₁ 비율, 회귀 잔차)은 공통 요인에 둔감"이라는 아이디어는 그대로 쓸 수 있다.
- **[S4] Peeters & De Roeck 2001** (1절) — 온도가 지배적 정상 변동, 회귀모델 신뢰구간 밖 = 손상.

### Inferences
- 고무 부싱 강성은 온도·예압·진폭에 강하게 의존한다(3절). 우리 정상 파일 간 8.6–8.7 kHz 혹의 유무·위치 차이가 이런 미지 EOV 때문일 가능성이 있고, 그렇다면 **절대 피크 위치는 EOV에 오염된 피처**다.
- 권장 피처 형태: (1) 피크 진동수 **비율**(f_a/f_b) — 공통 강성 스케일 변화 상쇄, (2) 대역 파워 / >200 Hz 총 파워 — 센서 감도·가진 세기(이득) 상쇄, (3) PCA 잔차 기반 novelty 점수.
- 정상 변동 방향 학습은 정상 데이터만으로 하므로 라벨 누출 위험이 적다(단, CV 폴드 안에서 학습).

### Gaps
- 자동차 부품 단품 시험에서 온도·지그 체결력에 따른 kHz 대역 공진 이동량을 정량화한 문헌은 찾지 못했다.

## 3. 고무/엘라스토머 부품: 노화·균열·박리가 동적 강성·손실계수·공진에 미치는 영향

### Takeaway
열산화 노화는 대체로 **강성 증가(→ 공진 상승)·손실계수 감소(→ 피크가 날카로워짐)** 방향이고, 균열·박리는 강성 감소(→ 공진 하강)·마찰 감쇠 증가 방향이라 서로 반대다. kHz 대역에서는 고무 내부 파동 공진(standing wave)으로 동적 강성에 봉우리·골이 생기며, 노화가 그 위치를 옮긴다.

### Cited Findings
- **[S8] Kari, L. (2017). Dynamic stiffness of chemically and physically ageing rubber vibration isolators in the audible frequency range. Part 1: constitutive equations. *Continuum Mech. Thermodyn.*, 29(5), 1027–1046. DOI 10.1007/s00161-017-0569-7 / Part 2: waveguide solution, 1047–1059. DOI 10.1007/s00161-017-0573-y** — verified (서지). OA 여부 확인 못함(Springer PDF 링크 존재) — [Part 1 PDF](https://link.springer.com/content/pdf/10.1007/s00161-017-0569-7.pdf), [Part 2 PDF](https://link.springer.com/content/pdf/10.1007/s00161-017-0573-y.pdf), [색인](https://ouci.dntb.gov.ua/works/4Ob2Qdm7)
  - 20–20,000 Hz 전 가청대역에서 노화 고무 방진구의 동적 강성 모델링. 짧은 물리적 노화 시간에서는 강한 주파수 의존성(봉우리·골), 물리적 노화가 길어지면 강성 크기 증가·봉우리 완화, **화학적 노화는 봉우리·골의 주파수를 이동**시킨다 — 같은 출처(검색 요약 기반, 본문 수치 미확인)
  - 적용: **kHz 대역(8 kHz)까지 고무 부품 동적 강성에 파동 공진 구조가 있다는 거의 유일한 근거.** 8.25/8.6 kHz 혹이 금속 암의 굽힘 모드가 아니라 부싱 고무의 파동 공진일 가능성도 가설(H)로 등록할 가치가 있다. 노화(정상 범주의 개체 차)도 이 위치를 옮긴다 → 정상 내 산포 원인 후보.
- **[S9] Gil-Negrete, N., Nieto, ?, Apezetxea, ?, Murua, ? — Quasi-static and dynamic properties of aged natural rubber compounds for transport applications.** (Taylor & Francis 책 장, *Constitutive Models for Rubber XI* 계열 추정, DOI 10.1201/9780429324710-99) — **partially** (URL의 DOI 문자열만 확인, 저자 이름 일부·책 제목 미확인) — [T&F](https://www.taylorfrancis.com/chapters/edit/10.1201/9780429324710-99/quasi-static-dynamic-properties-aged-natural-rubber-compounds-transport-applications-gil-negrete-nieto-apezetxea-murua)
  - 산화 노화가 천연고무 부싱의 준정적·동적 강성을 모두 증가시키는 것이 주효과 — 검색 요약 기반
- **[S10] (2023) 고무 방진 패드의 열산화 노화 연구, *Advances in Mechanical Engineering*(SAGE), DOI 10.1177/16878132231186734** — **partially** (대학 프록시 미러로만 확인, 저자·제목 미확인) — [미러](https://https-sage-cnpereading-com-443.webvpn1.xju.edu.cn/doi/10.1177/16878132231186734)
  - 노화 후 정적 강성 약 +20.7%, 동적 강성 약 +9.3%, 손실계수 약 −35%; 아레니우스 모델로 노화 기술 — 같은 출처(검색 요약 수치, 원문 대조 필요)
  - 적용: 강성 +9%면 f∝√k 로 공진 약 +4.5% 상승, 손실계수 −35%면 반치폭(Q⁻¹)도 비슷하게 감소 → "**정상 노화 = 오른쪽·좁은 피크**". 우리 관찰(정상 8.6–8.7 kHz 좁은 혹, 고장 8.25 kHz 넓은 혹)과 방향은 일관되지만, 이는 우연일 수 있으며 검증되지 않았다.
- 주파수 의존 모델링: 대부분의 부싱 모델은 저·중주파 데이터 기반이며 고주파 연구가 적다(GA-BP로 61–100 Hz 예측) — [Sci. Rep. 2022, DOI 10.1038/s41598-022-04839-2](https://link.springer.com/10.1038/s41598-022-04839-2) (**partially**: 서지 일부만 확인, OA)
- 상충: 물리적으로 노화된 NBR은 저장·손실 탄성률과 손실계수가 모두 감소했다는 보고도 있다(검색 요약) — 열산화 노화 "강성 증가"와 방향이 다름. 재료·노화 방식에 따라 다르다.

### Inferences
- 균열·박리 고장 → 국부 강성 저하(공진 하강) + 균열면 마찰/접촉에 의한 감쇠 증가(피크 넓어짐) 가 일반적 기대. "8.25 kHz, ~200 Hz 폭(넓음)" 고장 혹은 이 기대와 정성적으로 일치하나, 고무 균열의 kHz 대역 효과를 정량화한 문헌은 이번 조사에서 찾지 못했다.
- 피처 제안: 피크별 **반치폭/Q**와 **돌출도(prominence, 이동중앙값 기준선 대비 배수)**를 함께 써서 "왼쪽+넓은" 조합을 잡는다. 위치 단독보다 (위치, 폭) 2차원이 노화(오른쪽·좁음)와 손상(왼쪽·넓음)을 구분할 여지가 있다.

### Gaps
- 부싱 **균열·박리**가 kHz 대역 동적 강성·손실계수를 얼마나 바꾸는지에 대한 측정 문헌 미발견.
- 자동차 부싱 고무의 온도 1 °C당 동적 강성 변화율(kHz 대역) 수치 미확보.

## 4. 스펙트럼 피크 피처 추출: 돌출도·반치폭/Q·PSD 기반 감쇠 추정·비율 피처·비선형 지표

### Takeaway
반치폭(half-power bandwidth)으로 감쇠를 추정할 때 주파수 분해능·누설(leakage)·창함수가 편향을 만들며, 미지 가진(출력만 측정) 상황에서는 PSD 모양에 가진 스펙트럼이 섞인다. 균열의 "열림-닫힘(breathing)"은 고조파·부조파 같은 비선형 징후를 만들며, 보·축에서는 민감하지만 판 구조에서는 제한적이다.

### Cited Findings
- **[S11] Kijewski, T., Kareem, A. (2002). On the reliability of a class of system identification techniques: insights from bootstrap theory. (*Structural Safety* 24 계열로 알려짐)** — **partially** (Notre Dame NatHaz PDF로 존재 확인, 저널 서지 미확인). OA — [NatHaz PDF](https://nathaz.nd.edu/journals/(2002)On_the_reliability_of_a_class_of_system_identification_techniques.pdf)
  - 반치폭 관계(대역폭 ≈ 2ζf_n)는 피크 근처 백색잡음 가정 하의 근사 — 검색 요약 기반
- **[S12] Brincker, R., Ventura, C.E., Andersen, P. (2001). Damping estimation by frequency domain decomposition. IMAC XIX 발표** — **partially** (Aalborg 저장소 문서 확인, 학회 서지 미확인). OA — [AAU](https://vbn.aau.dk/ws/files/12773740/Damping_Estimation_by_Frequency_Domain_Decomposition)
  - 단일자유도 PSD 봉우리를 역FFT로 시간영역 상관함수로 돌려 감쇠·진동수 추정(EFDD 계열). 반치폭 직접 측정보다 누설 편향에 강함 — 같은 출처
- 반치폭 감쇠 추정이 주파수 분해능에 의존하며 그 원인이 PSD 누설이라는 보고(보도교 사례) — [CAS 카탈로그: Damping Characteristics of a Footbridge](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsair,edsair.doi...........711e6f390ab815a8db72637fad19961e) (**unverified**: 저자·서지 미확인). Welch PSD 자체의 편향 분석 — [Brincker et al., Welch's Method for PSD Estimation Revisited (AAU)](https://vbn.aau.dk/ws/files/9172924/Welch_s_Method_for_PSD_Estimation_-_Revisited) (**partially**)
- **[S13] Bovsunovsky, A., Surace, C. (2015). Non-linearities in the vibrations of elastic structures with a closing crack: a state of the art review. *Mechanical Systems and Signal Processing*, 62–63, 129–148. DOI 10.1016/j.ymssp.2015.01.021** — verified (서지), 비공개 — [PoliTo IRIS](https://iris.polito.it/handle/11583/2625668), [Peeref](https://peeref.com/works/1048014)
  - 닫히는 균열의 비선형 효과(초·부조화 공진, 고조파) 기반 진단은 보형 구조·회전축에서 가장 민감, 판형 구조에서는 적용 잠재력이 제한적 — 같은 출처. 같은 저자들의 2006년 FE 연구는 초·부조화 공진에서의 비선형 왜곡이 균열 존재의 고민감 지표라고 보고 — [NAS Ukraine](https://harvester.nas.gov.ua/Record/irk-123456789-47867)
  - 적용: 텐션암 본체는 보형에 가까워 비선형 지표가 이론상 유효. 그러나 **가진 기본 주파수를 모르는 상태**에서 고조파 비율을 정의하기 어렵다. 대안: 스펙트럼 피크 집합에서 정수배 관계(f, 2f, 3f) 존재 여부·고조파 파워비 자동 탐지, 또는 바이스펙트럼/첨도 같은 가진 비의존 비선형 지표.

### Inferences (우리 수치에 대입)
- 고장 혹: 중심 8.25 kHz, 폭 ~200 Hz → Q ≈ 41, ζ ≈ 1/(2Q) ≈ 1.2%. Welch nperseg=4096(25 kHz)이면 빈 간격 ≈ 6.1 Hz로 200 Hz 폭에 약 33빈 → 분해능 편향은 작다. 반면 정상의 "좁은 혹"이 수십 Hz 폭이라면 빈 수가 적어 반치폭 추정이 불안정 → **nperseg를 바꿔 반치폭이 안정되는지 확인**하는 점검이 필요.
- 2초 신호는 Welch 평균 횟수와 분해능이 상충. 피크 폭 피처에는 multitaper나 피크 근방 로렌츠(SDOF) 곡선 맞춤이 더 안정적일 수 있다(S12의 아이디어).
- 출력만 측정 + 가진 미상 → PSD 피크 폭은 구조 감쇠 + 가진 스펙트럼 형태의 합성이다. 따라서 "Q = 감쇠"라고 발표에서 단정하지 말고 "**겉보기(apparent) Q**"로 표현할 것.
- 이득 강건 피처: 피크 높이 비, 대역 파워 / >200 Hz 파워, 이동중앙값 기준선 대비 돌출 배수 — 모두 센서 감도·가진 세기 배율 상쇄. 이동중앙값 창(1.2 kHz)이 혹 폭(200 Hz)의 6배라 기준선이 혹에 거의 끌려가지 않는 설정은 적절해 보인다.

### Gaps
- PSD 피크 돌출도·반치폭을 분류 피처로 써서 부품 양불 판정 성능을 비교한 논문은 이번 조사에서 직접 찾지 못했다(공진 검사 문헌은 주로 피크 진동수만 사용).

## 5. 자동차 부품의 공진 기반 양산·출하 검사(Acoustic Resonance Testing, PCRT)

### Takeaway
공진 검사는 "**정상 부품 집단의 공진 패턴 분포를 학습하고, 벗어나면 불량**"이라는 one-class/패턴 비교 방식이며, 컨트롤암·로커암 같은 서스펜션 부품이 표준의 명시적 적용 예다. 표준은 질량·치수 편차 >5%, 학습셋 오분류, 온도 불안정을 대표적 실패 요인으로 꼽는다 — 우리 과제의 함정과 거의 같다.

### Cited Findings
- **[S14] ASTM E2534-20(2025). Standard Practice for Process Compensated Resonance Testing Via Swept Sine Input for Metallic and Non-Metallic Parts.** — verified (ASTM 상점 페이지), 유료 — [ASTM](https://store.astm.org/e2534-20r25.html)
  - 시험 부품의 공진 패턴을 "합격 부품 교육셋(teaching set)" 및 "표적 결함 부품 교육셋"과 비교, 통계·패턴인식 도구로 판정. 균열·공극·기공·개재물·잔류응력·열처리 편차·치수 편차를 대상으로 함 — 같은 출처
  - 자동차 적용 예: 강·주철·알루미늄 **로커암·컨트롤암**, 실린더 헤드, 소결 기어·클러치 플레이트, 단조 조향·변속 부품 등 — 같은 출처
  - 한계: 질량/치수 편차 >5%면 사용 불가할 수 있음; 과도한 공정 편차는 민감도 저하; **교육셋 내 오분류 부품·미지 이상은 정확도와 민감도를 크게 떨어뜨림**; 전신(whole-body) 측정이라 결함 종류 구분은 일반적으로 불가; 온도가 빠르게 변하는 부품엔 쓰지 말고 온도 안정화 후 측정; 높은 Q의 강성 물체에만 유효; 진동 흡수재·두꺼운 오일층은 시험을 제한 — 같은 출처
  - 적용: (1) "**고무 부싱이 붙은 텐션암**"은 E2534가 전제하는 '높은 Q의 강성 물체'에서 벗어날 수 있다 → 고무가 지배하는 저주파 대신 금속 암 모드가 남는 kHz 대역에 집중한 우리 방향과 일치. (2) 결함 종류 구분 불가 = 우리에게 고장 유형 라벨이 없는 상황과 동일, 이진 판정에 집중하는 것이 표준적. (3) 정상 120개 중 라벨 오류 가능성은 성능에 치명적 → 오류 분석 단계에서 "정상인데 고장처럼 보이는 파일" 점검 근거.
- PCRT의 판정 컷오프는 이상적으로는 구조 시험에서 설계 수준 미만으로 파손된 부품들의 패턴 점수로 정한다(업계지) — [Quality Magazine, The power of PCRT](https://qualitymag.com/articles/89183-the-power-of-pcrt) (**partially**, 업계 기사)
- **[S15] Biedermann, E., Livings, R., Mazdiyasni, S., Heffernan, J., Jauriqui, L., Mayes, A. (2019). Validation of process compensated resonance testing (PCRT) sorting modules trained with modeled data. *AIP Conference Proceedings* (QNDE 2018). DOI 10.1063/1.5099724** — **partially** (저자·DOI는 도서관 색인으로 확인, 권·쪽 미확인, 본문 미열람) — [색인](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_scitation_primary_10_1063_1_5099724), [Vibrant 소개](https://www.vibrantndt.com/portfolio-posts/validation-of-process-compensated-resonance-testing-pcrt-sorting-modules-trained-with-modeled-data/)
  - 실제 부품 수가 부족할 때 PCRT 순방향(FE) 모델로 생성한 "가상 스펙트럼"으로 교육셋을 보강; 결정방위·크리프 변형 분류 모듈을 모델로 학습해 통계적으로 유의한 실물 쿠폰 집단으로 검증 — 같은 출처
  - 적용: 고장 30개뿐인 우리 상황과 같은 "결함 샘플 부족" 문제. FE 모델이 없으므로 직접 적용은 어렵지만, **"정상 집단만으로 패턴 기준을 세우고 결함은 검증에 쓴다"**는 설계 원칙은 그대로 쓸 수 있다.
- 공진 검사(ART) 원리: 균열은 강성을 낮춰 고유진동수를 낮추고, 주조품 기공은 질량을 줄여 고유진동수를 높인다; 경계(허용 범위) 밖 피크가 있는 부품은 불량으로 배제 — [MDPI Appl. Sci. 12(15) 7648 (2022)](https://mdpi.com/2076-3417/12/15/7648) (**unverified**: 403으로 본문·저자 미확인; 합성 데이터로 학습하는 ART 연구로 검색 요약됨)
- ML 기반 ART 사례: 소결 기어(Fraunhofer, 시간-주파수 피처+ML) — [Fraunhofer Publica](https://publica.fraunhofer.de/handle/publica/425380); 커넥팅로드 고유진동 신호 피처 추출·결함 분류 — [EUSIPCO 2017 PDF](https://new.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347241.pdf); 비파괴평가 저널 논문 — [J. Nondestruct. Eval., DOI 10.1007/s10921-023-01035-8](https://link.springer.com/10.1007/s10921-023-01035-8) — 모두 **unverified**(제목·저자·결과 미확인, 검색 요약만). OA: EUSIPCO PDF는 공개.
- 균열 크기 효과 백서(업체) — [The Modal Shop white paper](https://www.modalshop.com/docs/themodalshoplibraries/white-papers/effect-of-crack-size-ndt-white-paper-(md-0280).pdf) (**unverified**, 업체 자료)

### Inferences
- 공진 검사 산업 관행 = "정상 집단 분포 + 경계 밖이면 불량". 이는 우리 과제에서 **정상 120개로 학습하는 one-class 점수(마할라노비스 거리, PCA 잔차)**를 지도 분류기와 함께 비교 후보로 둘 근거다. F2(재현율 중시) 평가와도 성격이 맞다.
- 다만 PCRT는 동일 지그·온도 안정화·스윕 사인 가진으로 측정 조건을 통제한다. 우리 데이터는 조건 정보가 없으므로 공진 이동 피처의 정상 산포가 PCRT보다 클 것이고, 절대 진동수 경계(PCRT식)보다 비율·정규화 피처가 필요하다.

### Gaps
- PCRT의 구체적 패턴인식 알고리즘(예: 마할라노비스-다구치 시스템 사용 여부)과 필요한 결함 샘플 수는 표준 요약·색인에서 확인하지 못했다(Vibrant NDT PDF는 텍스트 추출 실패).
- 고무 부싱 일체형 부품에 공진 검사를 적용한 사례는 찾지 못했다.

---

### 우선순위 요약(보고서 작성자용)
| # | 문헌 | 검증 | 우리 프로젝트 연결 |
|---|---|---|---|
| S5 | Sohn 2007 Phil Trans A | verified | EOV 정규화 필요성(조건 정보 없음) |
| S6 | Yan et al. 2005 MSSP I/II | partially | 정상만으로 PCA 잔차 novelty — 직접 적용 |
| S14 | ASTM E2534 (PCRT) | verified | 컨트롤암 명시, 함정 목록(>5% 편차, 교육셋 오분류, 온도, 고Q 전제) |
| S8 | Kari 2017 CMT I/II | verified | kHz 대역 고무 파동 공진·노화에 따른 봉우리 이동 |
| S10 | 2023 AME 고무 노화 | partially | 노화=강성↑·손실계수↓ (오른쪽·좁은 피크) |
| S1–S3 | Salawu / Doebling / Fan&Qiao | verified | 진동수 단독 한계, 1센서→모드형상 불가 |
| S4 | Peeters&De Roeck 2001 | verified | 온도 지배, 회귀 신뢰구간 판정, ~1% 검출 |
| S7 | Cross et al. 2011 | verified | 공통요인 상쇄 결합(비율·잔차) 아이디어 |
| S13 | Bovsunovsky&Surace 2015 | verified | breathing crack 고조파 — 보형 구조에서 유효 |
| S11–S12 | Kijewski&Kareem / Brincker | partially | 반치폭·Q 추정 편향, nperseg 민감도 점검 |
| S15 | Biedermann et al. 2019 | partially | 결함 샘플 부족 시 교육셋 설계 |
