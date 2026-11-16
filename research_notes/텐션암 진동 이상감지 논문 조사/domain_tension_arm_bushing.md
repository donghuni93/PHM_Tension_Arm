# 텐션암·고무부싱·볼조인트: 고장 메커니즘과 진동 징후 (도메인 문헌 조사)

조사일 2026-11-14. 검증 표기: **verified** = Crossref/DBpia 메타데이터로 서지 확인, **partially** = 검색 스니펫·저장소 목록으로 일부만 확인(저자 또는 DOI 미확인), **unverified** = 서지 확인 못 함.
팀 기존 목록(Balaji 2023, Karthikeyan 2024, Antoni 2006/2007, Jardine 2006, Zhang 2020, Andreaus 2007, Shah 2024, Sai 2023, Hoang & Kang 2019, Lee 외 2017 부싱 FEM, KTH 부싱 진폭의존 학위논문)은 다시 보고하지 않았다.

**한계를 먼저 적어 둔다.** 이번 조사에서 텐션암(텐션 스트럿/로드) 자체를 다룬 진동 진단 논문, 볼조인트 유격을 가속도계로 진단한 논문, 부싱 고장을 EOL에서 진동으로 판별한 논문은 **찾지 못했다**. 아래 내용은 (a) 고무부싱 노화·동특성 연구, (b) 현가 암 피로파손 연구, (c) 일반 기구학의 조인트 유격·충격 진동 연구를 모아 "고장 → 진동" 연결을 간접적으로 세운 것이다. ScienceDirect·PMC 본문은 접근이 막혀(403/reCAPTCHA) 대부분 초록·메타데이터 수준에서만 확인했다.

---

## Q1. 텐션암이 받는 하중과 대표 고장 모드

### Takeaway
텐션암/로어암은 차체(서브프레임)와 너클을 잇는 링크로, 양 끝 고무부싱 또는 볼조인트를 거쳐 전후·횡 하중과 노면 충격을 받는다. 문헌상 주 고장은 (1) 부싱 고무의 노화·경화, 피로균열, 강성 저하, (2) 볼조인트·연결부 유격, (3) 암 본체(특히 부싱 장착부·용접부)의 피로균열이다. 텐션암만 따로 다룬 학술 논문은 확인하지 못했다.

### Cited Findings
- **[논문 1] Engineering Failure Analysis 11(4):485–499 (2004), "Fatigue failure of suspension arm: Experimental analysis and multiaxial criterion"** — partially(저널·권호·쪽은 ResearchGate 목록으로 확인, 저자·DOI 미확인). 현가 암 피로시험과 파면 분석으로 균열 시작점을 찾았고, 다축 피로기준이 피로한도와 균열 방향을 잘 예측했다. — [ResearchGate](https://www.researchgate.net/publication/222935742_Fatigue_failure_of_suspension_arm_Experimental_analysis_and_multiaxial_criterion)
  - 적용: "암 균열" 가설의 물리적 근거. 균열은 대개 부싱 장착부나 용접부 같은 응력집중부에서 생긴다. 열림·닫힘 크랙(팀이 이미 인용한 Andreaus 2007)과 묶어 "하중 반전 때 생기는 비선형 충격과 고조파" 지표의 근거로 쓴다.
- **[논문 2] 자동차 로우어 암의 내구성 및 응력 해석 (KSAE 계열 국내 논문, 2010, ScienceON JAKO201028542315980)** — partially(서지 상세·DOI 미확인). 로우어 컨트롤암의 **리어 부싱부에서 피로수명이 크게 줄었다**고 보고했다. — [ScienceON](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201028542315980&dbt=NART)
  - 적용: 고장 위치 사전정보. 암에서 응력이 가장 집중되는 곳이 부싱 장착부라서 "부싱+장착부" 계통 고장이 많다는 가설(H-부싱)을 뒷받침한다.
- **[논문 3] Failure Analysis of a Lower Wishbone, Acta Physica Polonica A 128(2B) (2015 추정)** — partially(저널 PDF 경로로 확인, 저자·연도·DOI 미확인). 경상용차 위시본의 피로파손이 **용접 비드의 고응력부에서 시작했고 기공이 균열 발생과 성장을 촉진**했다. — [PDF](http://przyrbwn.icm.edu.pl/APP/PDF/128/a128z2bp020.pdf), [초록](https://przyrbwn.icm.edu.pl/APP/ABSTR/128/a128-2b-20.html)
- **[논문 4] Liu et al., "Predicting Stress Intensity Factor for Aluminum 6062 T6 Material in L-Shaped Lower Control Arm (LCA) Design Using XFEM", Materials (2024, PMC10780118)** — partially(PMC 목록으로 확인, 저자명·DOI 미확인). 균열은 **차체 장착부의 응력집중부**에서 시작하고 진전하며, 깊이비 a/t≈0.6에서 불안정해진다. 오픈액세스. — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10780118/)
- **[논문 5] Putra & Ikbal 2021, "Automotive suspension component behaviors driven on flat and rough road surfaces", Heliyon, DOI 10.1016/j.heliyon.2021.e07528** — partially(DOI는 검색 결과에서 확인, Crossref 미조회). 평탄로와 험로에서 현가 부품의 하중·응답을 비교했다. 오픈액세스. — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8287242/)
- 실무 자료(학술 논문 아님): 볼조인트 마모 → 내부 간극 증가 → 저속 선회·출발·범프에서 clunk. 같은 clunk는 스태빌라이저 링크, 스트럿 마운트, 컨트롤암 부싱에서도 나므로 **소리만으로 부위를 특정할 수 없다**. — [autodecode](https://autodecode.in/blogs/suspension-parts-guides/front-suspension-noise-diagnosis-knock-clunk-squeak-and-rattle), [B&M](https://blog.bandmautocare.com/diagnose-suspension-noise-guide/)

### Inferences
- 라벨이 "수리 전/후(또는 교체)"라서 고장군 30개 안에는 부싱 노화, 부싱 파손, 유격, 암 균열이 **섞여 있을 가능성이 높다**. 그래서 한 가지 징후(예: 한 대역의 에너지)로 모든 고장을 잡기 어렵다. "조용한 고장" 2개가 다른 메커니즘(강성 변화형)이고, 나머지는 충격형일 수 있다.

### Gaps
- 텐션암(텐션 스트럿/로드) 전용 고장 통계와 고장 모드 분포는 찾지 못했다.
- 볼조인트 마모의 정량 연구(유격 mm와 진동의 관계)도 찾지 못했다. 특허 US10119894는 볼 위치를 자기센서로 재는 방식이라 진동 진단이 아니다. — [USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10119894)

---

## Q2. 고장 모드별 진동/NVH 징후 (충격, 강성·공진 변화, 감쇠, 비선형)

### Takeaway
**유격(볼조인트 마모, 부싱 찢김·이탈, 체결 풀림)** 은 하중이 반전될 때 접촉이 끊겼다가 다시 붙으면서 **짧은 충격 → 가속도 스파이크 → 고주파 광대역 성분**을 만든다. 간극이 클수록, 속도가 빠를수록, 방향이 바뀔 때 커진다. 반대로 **부싱 경화**는 충격 없이 강성↑·감쇠↓를 일으켜 공진을 위로 옮기고 피크를 날카롭게 만든다. 이것이 "조용한 고장" 유형과 맞을 수 있다.

### Cited Findings
- **[논문 6] Experimental investigation of clearance effects in a revolute joint (ResearchGate 228755866)** — unverified(저자·저널·DOI 확인 못 함). **100 kHz 가속도계** 측정에서 간극 때문에 접촉이 끊겼다가 다시 붙을 때 충격이 생기고, 이것이 가속도 신호의 스파이크로 나타났다. — [ResearchGate](https://www.researchgate.net/publication/228755866_Experimental_investigation_of_clearance_effects_in_a_revolute_joint)
  - 적용: 우리 데이터에서 관측된 "고장의 짧은 간헐 충격(1.25–10 kHz)"과 가장 직접 맞는 메커니즘이다. **충격성 지표**(첨도, crest factor, 임계 초과 횟수, 포락선 피크 수, 스펙트럴 첨도)를 고역통과 신호에서 계산하는 근거가 된다.
- 기구 유격 연구들의 공통 결론: 전체 운동은 저주파인데 **결함 조인트의 충격이 고주파 성분을 더한다**. 간극이 클수록 진동 응답과 조인트 충격하중이 커진다. 고속에서, 특히 **방향 전환 때** 진동이 커진다. 링크나 슬리브가 유연하면 이 진동이 줄어든다. — [Clearance-induced vibration responses (RG 322766527)](https://www.researchgate.net/publication/322766527_Clearance-induced_vibration_responses_of_mechanical_systems_computational_and_experimental_investigations), [Li et al. 2021, Adv. Mech. Eng., DOI 10.1177/16878140211012541](https://doi.org/10.1177/16878140211012541) (partially: DOI는 검색 결과로 확인), [MDPI Machines 14(9):1073 clearance test bench](https://www.mdpi.com/2075-1702/14/9/1073) (partially)
  - 적용: (1) 우리 신호의 **20 Hz 이하 흔들림이 하중 반전 타이밍**이다. 충격이 저주파 흔들림의 특정 위상(영점 교차나 극값)에 몰리는지 보는 "위상 동기 충격" 지표를 만들 수 있다. 저주파 성분을 제거하지 말고 **위상 기준 신호로 남겨 두는** 전처리를 고려한다. (2) "부싱이 유연하면 충격이 완화된다"는 결과는 거꾸로 **부싱이 경화되면 같은 유격에서도 충격이 더 날카롭게 전달된다**는 뜻으로 읽을 수 있다(추론).
- **[논문 7] Zhang L., Bai G., Zhang T., Zhang Y., Fang Y., Tao J., "Experimental Study on the Effects of Wear and Clearance in a 3D Revolute Joint", 2021 Global Reliability and PHM (PHM-Nanjing), pp. 1–7, DOI 10.1109/phm-nanjing52125.2021.9612857** — verified(Crossref). 3D 회전조인트에서 마모와 간극이 동적 응답에 주는 영향을 실험한 PHM 연구다(초록 미확보). — [DOI](https://doi.org/10.1109/phm-nanjing52125.2021.9612857)
  - 적용: 볼조인트 유격 가설에 가장 가까운 "마모 → 간극 → 진동" 실험 문헌. 본문에서 사용 센서와 대역을 확인할 가치가 있다.
- **[논문 8] Liu Z., Zhang H., Sun Y., Zhang H., Li P., "Experimental investigation and numerical contact analysis of an offset slider-crank mechanism with irregular revolute clearance joint", Nonlinear Dynamics 114(16), art. 1045 (2026), DOI 10.1007/s11071-026-12945-3** — verified(Crossref). 간극과 속도가 클수록 충격 응답이 커지고, **비원형(편마모) 간극은 충격 크기와 타이밍을 바꾼다**. — [DOI](https://doi.org/10.1007/s11071-026-12945-3)
  - 적용: 충격 간격·크기가 불규칙하다는 점은 **주기성 지표(포락선 스펙트럼 피크)보다 통계 지표(충격 개수, 크기 분포)가 더 맞다**는 근거가 된다.
- 부싱 경화의 징후(강성↑·loss factor↓)는 Q3에 정리했다.

### Inferences
- 충격형 고장(유격·찢김·풀림·열림 크랙)은 **시간 국소적 + 고주파 광대역**이므로 고역통과 후 첨도·SK가 맞다. 강성형 고장(경화)은 **충격 없이 공진 위치와 날카로움(Q)이 바뀌므로** 스펙트럼 형상 지표(피크 주파수, 대역 중심, 대역폭)가 맞다. 두 지표군을 함께 쓰는 설계가 물리적으로 자연스럽다.

### Gaps
- 차량 현가 부품의 clunk/rattle을 가속도계로 측정해 주파수 대역을 보고한 학술 논문은 찾지 못했다(정비 블로그와 특허만 있음).

---

## Q3. 고무부싱 노화·온도·예하중·주파수(특히 kHz)에 따른 동강성·loss factor 변화와 공진 이동 방향

### Takeaway
열산화 노화는 가교밀도를 높여 **정적·동적 강성↑, loss factor↓**(경화·감쇠 감소)를 일으킨다. 균열·이탈·찢김은 문헌에서 **강성 저하**로 정의된다(내구시험에서 20–40% 강성 저하를 수명 종료로 봄). 따라서 공진 이동 방향은 **경화 → 위로, 균열·이탈·풀림 → 아래로**가 물리적으로 기대된다. 부싱의 동특성은 주파수가 오르면 강성·loss factor가 커지고, 진폭이 커지면 둘 다 작아진다(Payne 효과). **kHz 대역에서는 부싱 내부 공진(고무 매스의 파동 공진)이 생긴다**는 측정 보고가 있다.

### Cited Findings
- **[논문 9] Chen J., Li X., Chen C., Yang C., Gao X., "Influence Mechanism of Rubber Thermal Oxygen Aging on Dynamic Stiffness and Loss Factor of Rubber Isolation Pad", International Journal of Polymer Science 2022, Article ID 8673245, DOI 10.1155/2022/8673245** — verified(Crossref), **오픈액세스(CC BY 4.0)**. 노화로 경도가 5 HA 오를 때 **정강성 최대 +20.7%, 동강성 최대 +4.5%, loss factor 최대 −10.6%**. 여러 진폭에서 모두 동강성 증가·loss factor 감소 경향이었고, 주원인은 가교밀도 변화였다. 사용 온도는 일반 최대 70 °C, 극한 108 °C로 설정했다. — [DOI](https://doi.org/10.1155/2022/8673245)
  - 적용: "경화형 고장 = 공진 상향 + 감쇠 감소 → 더 날카로운 피크" 가설의 정량 근거. 단, 동강성 변화(+4.5%)는 크지 않다. 공진주파수는 √k에 비례하므로 약 +2% 이동에 그친다. 8.25 kHz와 8.6 kHz의 차이는 약 4%로 이 크기와 비슷한 수준이지만, 방향으로 보면 고장(8.25)이 정상(8.6)보다 **낮다**. 그래서 단순 경화보다 **강성 저하형(균열·이탈·풀림)이나 조립·장착 차이**일 가능성을 함께 검토해야 한다(추론, 검증 필요).
- **[논문 10] Gil-Negrete N., Nieto J., Apezetxea I.S., Murua J., "Quasi-static and dynamic properties of aged natural rubber compounds for transport applications", in Constitutive Models for Rubber XI, CRC Press, 2019, pp. 560–565, DOI 10.1201/9780429324710-99** — verified(Crossref). 노화는 주로 부싱의 **준정적·동적 강성을 높이며**, 이것이 차량 수명 동안 동적 거동을 바꾼다. — [DOI](https://doi.org/10.1201/9780429324710-99)
  - 적용: 천연고무 운송용 부싱에서도 "노화 → 강성↑"이 확인된다. 논문 9의 방향을 차량 부싱 쪽으로 보강한다.
- **[논문 11] Gejguš T., Schröder J., Loos K., Lion A., Johlitz M., "Advanced Characterisation of Soft Polymers under Cyclic Loading in Context of Engine Mounts", Polymers 14(3):429 (2022), DOI 10.3390/polym14030429** — verified(Crossref), **오픈액세스(CC BY 4.0)**. 전기차용으로 **3000 Hz까지 고주파 동강성**을 측정했고, **부싱 내부 공진**이 측정 강성과 음향 거동을 크게 바꿨다. 반복 대변형에서 생기는 에너지 소산이 **자기발열과 노화**를 일으킨다고 설명했다. — [DOI](https://doi.org/10.3390/polym14030429)
  - 적용: 부싱은 kHz 대역에서 단순 스프링이 아니라 내부 파동 공진을 가진 요소다. 따라서 수 kHz 대역 스펙트럼 형상이 부싱 상태를 반영할 수 있다. 다만 8–10 kHz는 3 kHz보다 훨씬 높아서, 그 대역의 hump는 **암·브래킷·센서 장착부의 구조 공진**일 가능성도 크다(추론).
- **[논문 12] Zucchini A., Hülsmann A., Naets F., "A methodology for the experimental characterization of rubber isolators with frequency and preload dependency", Mechanical Systems and Signal Processing 225:112225 (2025), DOI 10.1016/j.ymssp.2024.112225** — verified(Crossref, 본문은 403으로 미확인). 검색 초록상 기존 방법은 저주파·무예하중에 묶여 있었다. 이 방법으로 **kHz 대역까지, 최대 3 kN 예하중에서** 유효한 동강성 크기를 얻었다. 프리프린트는 SSRN DOI 10.2139/ssrn.4720152. — [DOI](https://doi.org/10.1016/j.ymssp.2024.112225)
  - 적용: **예하중(조립 체결력·차량 하중)에 따라 kHz 동강성이 바뀐다** → 체결 풀림이나 장착 차이가 고주파 공진을 옮길 수 있다는 가설의 근거. 정상 일부의 8.6 kHz hump처럼 **정상군 안의 변동**을 설명하는 후보가 된다.
- **[논문 13] Li G., Wu L., Zhang S., Liu F., "Frequency dependence prediction and parameter identification of rubber bushing", Scientific Reports 12:863 (2022), DOI 10.1038/s41598-022-04839-2** — verified(Crossref), **오픈액세스(CC BY 4.0)**. 1–100 Hz, 진폭 3단계 시험에서 주파수가 오르면 동강성·loss factor가 커지고 진폭이 커지면 둘 다 작아지는 경향을 정리했다. 5-파라미터 모델로 강성 오차 2% 미만, loss factor 오차 3% 미만으로 맞췄다. — [DOI](https://doi.org/10.1038/s41598-022-04839-2)
  - 적용: 진폭 의존성(Payne 효과) 때문에 **진폭이 큰 구간(강한 저주파 흔들림 순간)과 작은 구간에서 부싱의 전달 특성이 다르다**. 저주파 흔들림 크기로 구간을 나눠 고주파 지표를 비교하는 방식(조건부 지표)의 근거가 된다.
- **[논문 14] 현민수, 윤재민, 이광우, 박순철, 강대오, 허승진, "주파수와 진폭에 따른 비선형 동특성 예측을 위한 현가장치 부시 모델에 대한 연구", 한국자동차공학회논문집 28(9):621–628 (2020), DOI 10.7467/KSAE.2020.28.9.621** — verified(DBpia). 일반화 Maxwell 모델에 **마찰 Maxwell 요소(진폭 의존 셀)** 를 더해 현가 부시의 주파수·진폭 의존 비선형 동특성을 표현했다. 맥퍼슨 현가 반차량 내구 시뮬레이션에서 Kelvin-Voigt 모델과 비교했다. 한국어 논문이다. — [DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09418641)
  - 적용: 부싱 히스테리시스에 **마찰(쿨롱) 성분**이 있다는 점은 하중 반전 때 stick-slip성 비선형이 생길 수 있음을 뜻한다. 발표에서 "부싱은 선형 스프링이 아니다"를 설명할 때 국내 근거로 쓴다.
- 내구시험의 수명 종료 기준: 현가 부품 고무에서 **압축강성 40% 저하**를 고장으로 정의했다(Marquette 석사논문, partially). 엔진마운트에서는 **정강성 20% 저하**를 기준으로 썼다(Chalmers 학위논문, partially). 고무 피로 리뷰는 크랙 길이를 관찰하기 어려워 **최대하중 20% 저하**와 유효강성(최대하중/최대변위)을 기준으로 쓴다고 정리했다(PMC9654122, partially, 오픈액세스). — [Marquette](https://epublications.marquette.edu/cgi/viewcontent.cgi?article=1509&context=theses_open), [Chalmers](https://publications.lib.chalmers.se/records/fulltext/255700/255700.pdf), [PMC 리뷰](https://pmc.ncbi.nlm.nih.gov/articles/PMC9654122/)
  - 적용: **피로 균열과 손상은 강성 저하로 나타난다**는 정량 근거. 손상형 고장이라면 공진이 **아래로** 이동할 것으로 기대할 수 있다. 고장 8.25 kHz와 정상 8.6 kHz의 방향이 이 기대와 맞는다(단, 해당 hump가 부싱 관련 모드인지는 미확인).
- Frequency and amplitude dependence of the axial and radial stiffness of carbon-black filled rubber bushings — unverified(저자·저널 미확인. KTH Kari 계열로 보이며, 팀 목록의 KTH 학위논문과 겹칠 수 있음). 0–2000 Hz 동적 시험을 쓰고 예하중·진폭이 강성과 감쇠를 바꾼다는 내용이다. 모델 검증은 150 Hz까지만 했다. — [ResearchGate](https://www.researchgate.net/publication/223418937_Frequency_and_amplitude_dependence_of_the_axial_and_radial_stiffness_of_carbon-black_filled_rubber_bushings)

### Inferences
- 공진 이동 방향 정리(문헌 근거 + 추론):
  - 노화·경화 → k↑, η↓ → 공진 ↑, 피크 더 날카로움, 고주파 전달↑ (논문 9·10)
  - 피로 균열·찢김·고무-금속 이탈 → k↓ (내구 기준 20–40%) → 공진 ↓, 비선형·충격 동반 가능
  - 체결 풀림·예하중 감소 → kHz 동강성 변화 (논문 12) → 공진 이동(방향은 미확인), 유격이 생기면 충격
- 우리 관측 "고장 8.25 kHz hump vs 일부 정상 8.6 kHz hump"는 **강성 저하형 손상 또는 장착 예하중 차이**와 방향이 맞는다. 하지만 약 4% 차이는 측정 조건(센서 부착, 체결 토크) 차이로도 충분히 생길 수 있다. 라벨을 본 뒤 대역을 고르지 않는다는 팀 원칙에 따라, 특정 주파수 대신 **"주요 hump의 피크 주파수·Q값을 데이터 기반으로 추출"** 하는 일반 지표로 다루는 편이 안전하다.

### Gaps
- 차량 부싱의 **노화 시간·온도 대 kHz 동강성** 곡선을 직접 측정한 논문은 찾지 못했다(논문 9는 방진패드, 1–수십 Hz 수준으로 추정).
- 고무-금속 **이탈(debonding)** 이 동특성에 주는 영향을 정량화한 논문도 찾지 못했다.
- 논문 12의 본문 수치(kHz 강성 곡선, 예하중별 차이)는 접근이 막혀 확인하지 못했다.

---

## Q4. 진동 기반 현가 부품 진단 사례 (EOL·시험대·실차, 가속도계·고주파)

### Takeaway
팀이 이미 가진 Balaji·Sai·Karthikeyan 계열(시험대 + 가속도계 + ML) 외에 **부싱·텐션암 전용 진동 진단 학술 논문은 찾지 못했다**. 관련 근거는 철도 현가 댐퍼 상태감시, 부싱 진동 시험대(상용), 진동 절연체 감시 특허 수준이다.

### Cited Findings
- 철도 차량 현가 댐퍼의 상태를 **대차 진동 응답 분석**으로 감시한 연구(Sensors 계열 추정, PMC9100782) — partially(저자·DOI 미확인), 오픈액세스. — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9100782/)
- 진동 절연체(부싱) 감시 특허: **육안 검사로는 부싱 내부 손상을 놓칠 수 있다**고 지적하고, 장착 상태에서 감시하는 방법을 제안한다(특허, 학술 논문 아님). — [US9238393](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9238393)
- 상용 현가 부품 NVH 시험대는 예하중을 건 상태에서 노면 모사 가진으로 **스웨이바·부싱 소음**을 분석한다(제조사 자료). — [MB Dynamics](https://mbdynamics.de/en/project/suspension-component-test-rig/)
- e-트랜스액슬 EOL 연구(arXiv, 2026): 일반 EOL 진동검사는 기본 진동 수준만 보므로 **미세하거나 과도적인(transient) 결함을 놓칠 수 있다**. 이를 웨이블릿·STFT 특징으로 보완했다 — unverified(arXiv 번호만 확인). — [arXiv](https://arxiv.org/pdf/2609.22172)
  - 적용: 우리 고장이 "간헐적 짧은 충격"이라는 관측과 맞는다. RMS 같은 전역 지표보다 **시간-주파수 국소 지표**(STFT 프레임별 최대, 웨이블릿 패킷 에너지 분포)가 필요하다는 근거가 된다.

### Inferences
- 우리 데이터는 2초 기록 + 강한 저주파 흔들림이 있으므로 **시험대 가진(또는 실차 노면 가진) 환경**으로 보인다. 그렇다면 하중 반전 횟수가 파일마다 다를 수 있다. 충격 지표는 "초당 개수"나 "흔들림 사이클당 개수"로 정규화하는 것을 고려한다.

### Gaps
- 텐션암/로어암 부싱의 EOL 진동 판정, 고주파(>5 kHz) 측정 사례를 다룬 학술 논문은 찾지 못했다.

---

## Q5. 한국어 논문 (텐션암 / 로어암 / 부싱 / 볼조인트 / 현가 진동)

### Takeaway
"텐션암" 자체를 다룬 국내 논문은 찾지 못했다. 부싱 동특성 모델링과 로어암·후륜 현가 피로 해석 논문은 있으나 **진단(이상감지) 논문은 없었다**.

### Cited Findings
- 현민수 외 2020, 한국자동차공학회논문집 28(9):621–628, DOI 10.7467/KSAE.2020.28.9.621 — verified (위 논문 14).
- 「자동차 로우어 암의 내구성 및 응력 해석」(2010) — partially (위 논문 2). — [ScienceON](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201028542315980&dbt=NART)
- 「동역학 해석을 위한 고무부싱의 실험적 모델링에 대한 연구」(2004, ScienceON JAKO200410912827136) — partially(저자·DOI 미확인). 1축 내구시험기로 부싱을 축·반경 방향으로 사인/랜덤 가진해 특성을 측정하고 인공신경망으로 모델링했다. — [ScienceON](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO200410912827136&dbt=NART)
- 「Fatigue Analysis of Rear Suspension Part Applying Multi-body Dynamics」, 대한기계학회논문집 A (2015, JAKO201530261998780) — partially. 정규모드 기반 다물체 동역학으로 **공진 내구해석**을 제안하고 시험과 비교했다. — [KoreaScience](https://koreascience.kr/article/JAKO201530261998780.page)
  - 적용: 현가 부품의 피로가 **공진 모드 응답**에 지배된다는 관점. 고장 부품에서 특정 구조 모드가 달라지는 것과 연결된다.
- 한국소음진동공학회논문집 21(7):674–681 (2011), DOI 10.5050/KSNVE.2011.21.7.674 — partially(DOI·권호는 PDF 머리말에서 확인, 제목·저자는 본문 파싱 실패로 미확인). 검색 요약상 로드노이즈 저감을 위해 **후륜 트레일링암 부싱 동강성**을 포함한 현가 인자를 DFSS로 강건 최적화한 연구다. — [KoreaScience PDF](https://koreascience.kr/article/JAKO201122852778453.pdf)
  - 적용: 부싱 동강성이 노면 진동이 차체로 전달되는 경로를 결정한다는 국내 근거. 부싱 상태가 바뀌면 센서까지 전달되는 함수가 바뀐다는 논리로 쓴다.

### Gaps
- 볼조인트 유격·마모의 국내 실험 논문, 텐션 스트럿 고장 사례는 찾지 못했다. KCI에서 "텐션 로드", "트레일링 암 부싱 NVH", "볼 조인트 마모"로 직접 검색하는 것을 권한다.

---

## 요약: 파이프라인·지표에 연결 (모두 추론, 검증 필요)

| 고장 메커니즘 | 기대 진동 징후 | 근거 | 후보 지표 / 단계 |
|---|---|---|---|
| 유격(볼조인트 마모, 부싱 찢김·이탈, 체결 풀림) | 하중 반전 때 짧은 충격 → 고주파 광대역 스파이크, 불규칙 간격 | 논문 6·7·8, clearance 리뷰 | 고역통과 후 첨도·crest·임계 초과 수, SK. 저주파 흔들림 위상 대비 충격 위치 |
| 부싱 노화·경화 | k↑·η↓ → 공진 ↑, 피크 날카로움, 충격 없음("조용한" 고장 후보) | 논문 9·10 | 주요 hump의 피크 주파수·Q·대역 중심(데이터 기반 추출) |
| 고무 피로균열·손상 | 강성 20–40%↓ → 공진 ↓, 비선형 | 내구 기준 문헌 | 같은 스펙트럼 형상 지표, 고조파·비선형 지표 |
| 예하중·장착 차이 | kHz 동강성 변화 → 정상군 내 공진 변동 | 논문 12 | 정상 분포의 폭으로 다룸(오경보 원인 분석) |
| 암 피로균열 | 열림/닫힘 비선형, 충격, 모드 변화 | 논문 1·3·4 | 충격 지표 + 모드 주파수 |
| 진폭 의존(Payne) | 흔들림이 큰 구간과 작은 구간에서 전달 특성이 다름 | 논문 13·14 | 저주파 진폭으로 구간을 나눈 조건부 고주파 지표 |

**오픈액세스**: 논문 9(IJPS 2022), 11(Polymers 2022), 13(Sci Rep 2022), 4(Materials 2024, PMC), 5(Heliyon 2021), 고무 피로 리뷰(PMC9654122), 철도 댐퍼(PMC9100782). 논문 12(MSSP)는 SSRN 프리프린트가 있다.
