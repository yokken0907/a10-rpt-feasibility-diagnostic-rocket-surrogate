# large_assets_not_in_git

このフォルダは、GitHub本体から除外した大容量raw CSVのinventoryを置く。

## 除外理由

- 1ファイル50MB級のper-seed frontier CSVが含まれる。
- GitHub本体に置くと重く、公開repoとして扱いにくい。
- raw sweepを公開する場合は、ZenodoまたはGitHub release assetへ分離する方がよい。

## 現在の扱い

- summary CSVとfiguresはrepo本体に含める。
- large raw CSVはinventoryのみ記録する。
