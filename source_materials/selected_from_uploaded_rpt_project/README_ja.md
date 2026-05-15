# selected_from_uploaded_rpt_project

このフォルダは、アップロードされた `a10_rpt_phase1_selected.zip` から、GitHub公開に適した素材だけを抽出したsource snapshotである。

## 含めたもの

- A10-RPT関連 Python scripts
- shell / TeX / Markdown
- 論文図表に対応する可能性のあるPNG figures
- 小型summary CSV

## 除外したもの

- `.venv/`
- `venv_rocket/`
- `site-packages/`
- `__pycache__/`
- `*.pyc`
- compiled binaries
- `all_code*.txt`
- 1MB超のlarge raw/per-seed CSV

## 注意

これはpaper companion archiveであり、全raw sweepをGitHub本体に含める完全再現パッケージではない。  
大容量CSVは `large_assets_not_in_git/` にinventoryだけを置き、必要ならZenodoまたはGitHub release assetで扱う。
