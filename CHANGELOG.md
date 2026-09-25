# Changelog

All notable changes to DOE Toolkit are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/).

## [0.3.0] - 2026-09-25

### New
- **Design-aware automatic model selection** on the Analyze page (Step 6): the app now suggests and applies a model matched to the design actually used (full/fractional factorial, response surface, split-plot), integrated with the BIC stepwise workflow.

### Fixed
- Fractional-factorial designs with natural-unit factor ranges no longer double-decode, which previously produced garbage factor levels (#45).
- The "orthogonal" CCD alpha is now truly orthogonal (correct standard alpha formula); the UI previously computed a non-orthogonal value (#46).
- Custom fractional generators with a base factor on the left-hand side are now rejected with a clear error message instead of crashing later during generation (#47).
- 2^(6-1) and 2^(7-1) half-fractions (Resolution VI/VII) are now reachable; these two designs no longer crash in the default generation path (#48).
- Box-Behnken designs actually generate Box-Behnken now (routing previously fell back to a CCD).
- D-optimal design generation no longer breaks (the polynomial builder returned an unexpected tuple).
- Profiler contour and 3D plots no longer have transposed axes.

### Changed
- Analysis engine robustness (PR #52): non-estimable/degenerate model terms are pruned before fitting with recorded reasons, missing response values are handled and reported per run, and a single numerically-guarded lack-of-fit test now backs both the Analyze page and the diagnostics engine.
- Algorithm reference docs (in-app Help page) rewritten to match the implemented behavior.

## [0.2.1] - 2026-09-18

### Fixed
- Packaged-app launch crash: the packaged Windows app now launches Streamlit via the bundled environment's Python interpreter.

## [0.2.0] - 2026-09-17

### New
- Replicate and block support for full factorial designs (#30).
- Design-Expert-style interaction plots tab (#30).
- Standardized-effects Pareto charts and categorical optimization (#30).
- `discrete_numeric` factors on the Optimize page, with snapping to declared levels (#30).
- JMP-style response definition grid with auto-sanitized names (#30).
- Box Plots tab (#33).
- Optimize-page rework: options moved to the top, per-response bounds support, and a "None" option to exclude a response from optimization (#34).
- License audit tool and bundled `THIRD_PARTY_NOTICES.txt` (#38).

### Fixed
- Categorical factors with numeric labels treated as continuous in ANOVA (#30).
- Trailing-commas crash when importing Excel-resaved design CSVs (#30).
- HTML report prediction profiler NaN and categorical-level crash (#35).
- Windows packaging portability: hardcoded paths removed; conda-pack ≥ 0.9.2 enforced and the packed environment validated by the build scripts (#38).
- `build.bat` cmd parsing crash (LF line endings + parenthesized blocks) (#38).

## [0.1.0] - 2026-03-19

### Added
- Initial release.