# openfx-pixels

OMT / NDI OpenFX 送出プラグイン向けの共有クレートです。

- `openfx` — OpenFX ABI ヘルパー（bindgen）
- `openfx-pixels` — ホスト窓から packed BGRA / RGBA への SIMD 変換（AVX2 / SSSE3 / SSE2 / NEON / scalar）

プラグインからは同じ git revision を参照してください。`openfx` と `openfx-pixels` を別コピーにすると型が一致しません。

```toml
openfx = { git = "https://github.com/MikanseiLaboratory/openfx-pixels", rev = "<commit>" }
openfx-pixels = { git = "https://github.com/MikanseiLaboratory/openfx-pixels", rev = "<commit>" }
```

## 要件

- Rust 1.97
- LLVM（bindgen / libclang）
- Windows x64（プラグインと同じ）

```powershell
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

## ライセンス

MIT。OpenFX ヘッダーは Academy Software Foundation の BSD-3-Clause です。詳細は `THIRD_PARTY_NOTICES.md`。
