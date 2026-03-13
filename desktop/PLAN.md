# Desktop Plan

This document breaks the desktop side of ParallaxGen into one epic, milestone phases, and issue-sized work items. It is based on the current repository state in `desktop/` and the v2 project spec in `project.md`.

## Current State

The desktop scaffold already exists and includes:

- Python package layout under `desktop/src/parallaxgen/`
- CLI entrypoints in `desktop/src/parallaxgen/cli.py`
- starter data models in `desktop/src/parallaxgen/models.py`
- placeholder pipeline modules for depth, segmentation, composition, inpainting, preview, and packaging
- one basic manifest test in `desktop/tests/test_manifest.py`

What is still placeholder-only:

- depth inference returns a synthetic ramp
- subject extraction returns a synthetic rectangle mask
- layer planning is metadata-only and does not render real assets
- package writing creates empty `.webp` files
- preview is JSON text rather than visual output
- there is no config system, no quality scoring, and no real ML integration yet

## Epic

### Epic: Desktop Spatial Package Pipeline MVP

Build a production-shaped desktop pipeline that turns one photo into a valid `.parallax` package containing real layer assets, composition metadata, and preview output for the Android renderer.

Epic goals:

- replace all placeholder generation with real image processing
- keep the CLI stable while upgrading internals
- produce a deterministic package contract for Android
- make output quality measurable with validation and QA artifacts

Epic done definition:

- `parallaxgen process <image>` writes a complete package with real image assets
- `meta.json` and `index.json` match the v2 schema used by Android
- preview output is visual and useful for composition QA
- batch mode works on a folder of images
- basic tests cover manifest generation, package writing, and core pipeline invariants

## Milestones

## Milestone 1: Pipeline Foundations

Goal:

Stabilize the desktop package structure, config surface, and package contract before integrating heavy ML components.

Exit criteria:

- package schema is frozen for v1 MVP
- CLI options support config needed by downstream stages
- file writing creates non-empty image artifacts or well-defined placeholders for each stage
- tests cover schema and package layout

Issues:

### Issue D-1: Freeze desktop package contract

Scope:

- review `WallpaperMeta`, `WallpaperPackage`, and manifest models
- add any missing fields required by the Android loader
- define clear rules for required vs optional assets

Acceptance criteria:

- `models.py` contains the canonical package schema
- `meta.json` and `index.json` output are versioned and documented in code
- package asset names match the spec exactly

Dependencies:

- none

### Issue D-2: Add pipeline configuration layer

Scope:

- introduce typed config for model names, output size, overscan, clock safe rect policy, and quality thresholds
- thread config through `cli.py` and scene building

Acceptance criteria:

- process and batch commands accept config inputs without breaking current behavior
- defaults match `project.md`
- config is serializable for debugging

Dependencies:

- D-1

### Issue D-3: Replace empty asset writes with stage-aware outputs

Scope:

- ensure package writing generates actual files for depth map, masks, preview, and layers
- if a stage is still placeholder-driven, output a deterministic image artifact instead of an empty file

Acceptance criteria:

- no generated package contains zero-byte image files
- output package can be inspected manually without ambiguity

Dependencies:

- D-1

### Issue D-4: Expand desktop test baseline

Scope:

- add tests for package layout, metadata serialization, and CLI command behavior
- keep tests lightweight and not dependent on GPU or large model downloads

Acceptance criteria:

- tests cover package generation from a fake input image
- schema regressions fail fast in CI

Dependencies:

- D-1, D-3

## Milestone 2: Depth And Subject Extraction

Goal:

Integrate real depth and subject extraction so the pipeline stops generating synthetic geometry.

Exit criteria:

- real depth inference is available behind one stable interface
- real subject masking is available behind one stable interface
- CPU and CUDA execution paths are explicit

Issues:

### Issue D-5: Implement real depth model adapter

Scope:

- replace the synthetic ramp in `depth_runner.py`
- add model loading, preprocessing, inference, and normalized output
- support CPU fallback with explicit warnings about performance

Acceptance criteria:

- `DepthRunner.infer()` returns a real normalized depth map
- depth output preserves source image dimensions or documented export dimensions
- failure modes are explicit and actionable

Dependencies:

- D-2

### Issue D-6: Implement depth post-processing utilities

Scope:

- add normalization, smoothing, clipping, and edge-preserving cleanup helpers
- keep utilities deterministic and testable

Acceptance criteria:

- depth maps are stable enough for layer planning
- utility behavior is covered with unit tests on synthetic arrays

Dependencies:

- D-5

### Issue D-7: Implement real subject segmentation adapter

Scope:

- replace the rectangular mask in `subject_runner.py`
- produce subject alpha plus normalized bounding box
- choose one initial segmentation path that is practical on desktop hardware

Acceptance criteria:

- subject extraction returns a real mask for common portrait and object images
- bounding box is derived from the mask, not hard-coded

Dependencies:

- D-2

### Issue D-8: Implement matte refinement

Scope:

- replace pass-through alpha refinement with edge-aware cleanup
- improve thin contours and reduce mask chatter

Acceptance criteria:

- refined alpha is visibly cleaner than raw segmentation output
- refinement does not destroy coarse subject silhouette

Dependencies:

- D-7

## Milestone 3: Scene Composition And Layer Rendering

Goal:

Turn depth plus subject data into real 5-layer outputs with composition metadata and clock-safe placement.

Exit criteria:

- all five scene layers are generated from image content
- safe clock region and clock occlusion mask are derived from scene data
- scene planning is deterministic enough for repeatable exports

Issues:

### Issue D-9: Implement semantic layer planner

Scope:

- replace the fixed layer list planner with one driven by depth and subject data
- assign pixels into five authored scene layers

Acceptance criteria:

- planner outputs meaningful per-layer masks or composition regions
- hero subject and front accent handling are explicit

Dependencies:

- D-6, D-8

### Issue D-10: Implement real layer compositing and export

Scope:

- convert planned masks into RGBA layers
- export real WebP assets for layer images
- ensure alpha handling matches Android renderer expectations

Acceptance criteria:

- layer assets are valid images with correct dimensions
- foreground layers preserve transparency

Dependencies:

- D-9

### Issue D-11: Implement clock-safe composition analysis

Scope:

- derive `safe_clock_rect` from negative space and subject placement
- derive `clock_occlusion_mask` from foreground geometry

Acceptance criteria:

- clock region is data-driven instead of hard-coded
- bad images can be flagged when no viable clock zone exists

Dependencies:

- D-9

### Issue D-12: Add scene quality scoring

Scope:

- score images for depth separation, mask cleanliness, and clock readability
- surface warnings in CLI output and preview payloads

Acceptance criteria:

- processing surfaces quality warnings before export completes
- score thresholds are configurable

Dependencies:

- D-10, D-11

## Milestone 4: Inpainting, Packaging, And Preview

Goal:

Fill holes created by composition, package everything into a stable archive, and produce visual QA previews.

Exit criteria:

- background and midground holes are filled with real inpainting
- package archives are ready for Android ingestion
- preview output helps reject poor scenes early

Issues:

### Issue D-13: Implement inpainting integration

Scope:

- replace the static inpaint plan with real hole-filling execution
- apply stronger inpainting to background and lighter cleanup to mid layers

Acceptance criteria:

- exported layers do not contain obvious transparent holes where filled content is expected
- inpainting can be skipped or downgraded via config for slower machines

Dependencies:

- D-10

### Issue D-14: Finalize `.parallax` package writer

Scope:

- make package writing fully consistent with the finalized schema
- ensure `pack` produces archives the Android app can consume without translation

Acceptance criteria:

- archive contents mirror generated directories exactly
- package writer rejects incomplete packages with actionable errors

Dependencies:

- D-3, D-10, D-13

### Issue D-15: Implement visual preview generation

Scope:

- replace JSON-only preview with a real preview image or lightweight animated output
- include clock placement overlay and QA cues

Acceptance criteria:

- preview command produces an asset a human can inspect quickly
- preview reflects actual layer planning and clock zone decisions

Dependencies:

- D-10, D-11

## Milestone 5: Hardening And Developer Experience

Goal:

Make the desktop pipeline maintainable, debuggable, and practical for repeat use.

Exit criteria:

- logging, error handling, and tests are good enough for iteration
- performance bottlenecks are visible
- local setup is straightforward

Issues:

### Issue D-16: Add structured logging and progress reporting

Scope:

- add progress indicators for long-running stages
- log selected models, device choice, timings, and quality warnings

Acceptance criteria:

- CLI output clearly identifies which stage is running
- failures include context about the failing stage

Dependencies:

- D-5, D-7, D-13

### Issue D-17: Add benchmark and sample fixtures

Scope:

- create a small local sample set and benchmark script
- measure CPU versus CUDA timing where available

Acceptance criteria:

- one command can profile end-to-end processing time on sample inputs
- benchmark results are easy to compare across runs

Dependencies:

- D-14

### Issue D-18: Improve packaging and onboarding docs

Scope:

- document environment setup, model requirements, optional GPU path, and common failure cases
- document the package contract for Android consumers

Acceptance criteria:

- a new developer can install and run the starter pipeline from docs alone
- docs reflect the real CLI and schema

Dependencies:

- D-14

## Recommended Order

Implementation order for the desktop team:

1. D-1 through D-4
2. D-5 through D-8
3. D-9 through D-12
4. D-13 through D-15
5. D-16 through D-18

## Suggested GitHub Tracking Structure

Epic:

- `Desktop Spatial Package Pipeline MVP`

Milestones:

- `Desktop M1 - Foundations`
- `Desktop M2 - Depth And Subject Extraction`
- `Desktop M3 - Scene Composition`
- `Desktop M4 - Packaging And Preview`
- `Desktop M5 - Hardening`

Issue labels:

- `desktop`
- `epic`
- `milestone`
- `pipeline`
- `ml`
- `packaging`
- `preview`
- `testing`
