# 再現性メモ

## 現在の状態

このリポジトリはpaper companion archiveである。

## あるもの

- 論文PDF
- 論文表に対応するsummary CSV
- selected scripts
- selected figures
- small summary CSV
- large raw CSV inventory
- claim boundary / limitations

## あえて入れていないもの

- virtual environment
- compiled dependencies
- site-packages
- all_code
- large raw/per-seed CSV

## 再現性の意味

このサロゲートはnondimensionalであるため、再現性とは実エンジンデータとの一致ではなく、指定script、seed、stress profile、resource multiplierの一致を意味する。

## 現在の限界

全large raw sweepをGitHub本体には入れていないため、完全raw-output archiveではない。必要ならZenodo側でlarge raw CSVを別アーカイブ化する。
