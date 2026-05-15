# 実務位置づけシート

## 技術名

A10-RPT Feasibility-Diagnostic Rocket-Propulsion Surrogate  
または  
A10-RPT: Mission-Variable-Preserving Structured Priors for Rocket-Propulsion Surrogates

## 対象産業

- ロケット推進研究
- propulsion control research
- reduced-order simulation
- safety-oriented control evaluation
- feasibility diagnostics
- resource-frontier analysis

## 現場課題

ロケット推進では、圧力、燃焼不安定性、熱負荷、供給余裕、制御入力制限、インパルス達成を同時に満たす必要がある。  
名目性能を上げるだけでは、供給余裕の枯渇、熱制限、振動成長、combined stressで破綻する可能性がある。

## A10の役割

A10は、単一の万能制御器ではなく、危険領域に近づいたときに制御を構造化し、ボトルネックを分離してbarrierとして扱うstructured-prior frameworkとして機能する。

## 期待効果

- どのstress regimeがcontroller-feasibleかを分類できる。
- どのregimeがresource expansionを必要とするかを可視化できる。
- supply barrier / thermal barrierのように、支配的制約の移動を診断できる。
- 「制御で解ける問題」と「資源または構造変更が必要な問題」を分けられる。

## 検証済み範囲

論文内では以下が報告されている。

- v0.4 thermal-polished controllerがtested in-distribution ensembleでfailure rate 0、unsafe duration 0、CVaR0.9(V)=0を達成。
- combined_moderateはsupply scale S≈1.50–1.55でfeasible frontierが開く。
- heat_150はcooling scale C≈2.50でnear-feasibleになるが、rare residual thermal failuresが残る。
- combined_harshはC=2.50, S=1.75, Is=1.10でもfailure rate 0.075で、severe thermal-limited near-frontierとして残る。

## 未検証範囲

- 実エンジン検証
- 実推進剤・燃焼器・ターボポンプ・冷却チャネルへの対応
- ハードウェア実装
- flight controller
- hazardous experiment
- formal barrier certificate
- aerospace safety certification

## 実装への次ステップ

1. paper companion archiveとして公開する。
2. large raw CSVはGitHub本体ではなくZenodoまたはrelease asset候補にする。
3. 代表scriptとsummary CSVだけを公開repoに置く。
4. 専門家レビューでは「推進装置」ではなく「reduced surrogate control theory」として評価を受ける。
5. 実工学へ進む場合は、専門の航空宇宙工学者・安全規制・高忠実度モデルが必須。

## 想定読者

- 制御工学研究者
- reduced-order propulsion modelingに関心のある研究者
- safety / viability / barrier-method研究者
- resource-frontier diagnosticsに関心のある応用研究者
- A10 structured prior frameworkの評価者

## 誇張しない一文の結論

A10-RPTは、実ロケットエンジン設計ではなく、低次元・無次元推進サロゲートにおいて制御可能領域、resource-frontier領域、near-frontier領域を分類するfeasibility-diagnostic control frameworkである。
