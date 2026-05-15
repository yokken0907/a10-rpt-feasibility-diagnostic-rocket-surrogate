# A10-RPT Feasibility-Diagnostic Rocket-Propulsion Surrogate

このリポジトリは、以下のAI支援独立研究論文に対応するGitHub配置用フォルダである。

**A10-RPT: Mission-Variable-Preserving Structured Priors for Rocket-Propulsion Surrogates under Thermal, Supply, and Combined Stress**

著者: 吉村圭司（Independent Researcher）  
状態: GitHub-ready paper companion archive v0.1.1-public-gate; Zenodo-safe checkfix package

## 位置づけ

A10-RPTは、ロケット推進そのものの新方式ではなく、nondimensional reduced surrogate における feasibility-first structured prior control の研究である。

対象は、thermal stress、supply-margin stress、growth-rate stress、noise/disturbance stress、combined stress の下で、mission variableを守れるかどうかを調べる閉じた中間サロゲートである。

## 中心的解釈

A10-RPTは **feasibility-diagnostic control theory** として扱うのが安全である。  
つまり、制御だけで実現可能な領域、resource frontierを広げれば可能になる領域、near-frontier、architecture-limitedまたは未解決領域を分類するための理論である。

実エンジン性能を予測するものではない。

## 含まれるもの

- 論文PDF
- README日本語版・英語版
- claim boundary
- limitations
- AI支援開示
- 実務位置づけ表
- アップロードされたA10-RPT project archiveから抽出したselected scripts / figures
- 論文表から再構成したsummary CSV
- GitHub本体から除外したlarge raw CSVのinventory

## 主張しないこと

本リポジトリは、以下を主張しない。

- エンジン設計
- 推進性能予測
- 飛行ハードウェア実装
- 危険実験の手順
- 推進剤取扱い
- 実エンジン運用
- 形式的barrier certificate保証
- 航空宇宙安全認証

## 現在の状態

これはpaper companion archiveであり、航空宇宙工学用の実機モデル、認証済み制御器、または全raw sweepをone-commandで再現する完全パッケージではない。

## PUBLIC-GATE-0 status

判定: `PASS-WITH-MINOR-PUBLICATION-FIXES-A10-RPT-PUBLIC-GATE-0`  
公開版: `v0.1.1-public-gate`  
分類: ロケット推進サロゲート・feasibility診断制御理論

このリポジトリは、A10 Evidence-Lock Protocol型の公開前監査により、主張境界・非主張事項・manifest整合性・GitHub/Zenodo/Jxiv方針を固定した public-gate 版である。

