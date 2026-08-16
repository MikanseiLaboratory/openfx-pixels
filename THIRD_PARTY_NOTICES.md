# Third-Party Notices

This file lists third-party software included with openfx-pixels.
Crate versions are pinned by `Cargo.lock`.

## OpenFX headers

Vendored from [AcademySoftwareFoundation/openfx](https://github.com/AcademySoftwareFoundation/openfx)
commit `3de640d6f645fe6e346acd57e568d8b0a5ae4574`.

BSD 3-Clause License. The full text is in `crates/openfx/vendor/LICENSE.md`.

```text
Copyright (c) 2025, OpenFX and contributors to the OpenFX project
SPDX-License-Identifier: BSD-3-Clause
```

## Remaining crates

See `Cargo.lock` for the complete, version-pinned dependency graph. Build crates include bindgen and clang-sys.

License texts for crates.io packages can be regenerated with `cargo about generate` when `cargo-about` is available.
