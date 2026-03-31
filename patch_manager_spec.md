# Patch Accumulation System — Design Specification
*ComfyUI-Wan-VACE-Prep · stuttlepress*

---

## 1. Overview

The Patch Accumulation System enables non-destructive, iterative VACE Inline smoothing of multiple transitions in a single source video. Each transition is processed in a separate queue run, with results stored as patch files on disk. When all desired transitions have been processed, a second workflow assembles the patches against the original source video in a single pass, with no intermediate re-encoding of the source.

The system consists of two new nodes and two associated workflow files:

- **Patch Manager** — bookkeeping, UI panel, conflict detection
- **Patch Assembler** — assembly of accumulated patches into a final IMAGE batch
- **smooth_accumulate.json** — workflow for generating patch files (queue once per transition)
- **smooth_assemble.json** — lightweight workflow for final assembly (no inference required)

---

## 2. Design Principles

- Every queue run of the accumulate workflow generates exactly one VACE Inline pass and saves exactly one patch.
- The source video is never modified. All patches reference the original source by path. Assembled output is never fed back into VFS.
- Patch files are saved lossless (FFV1/MKV) to avoid generational quality loss at assembly time.
- Video encoding is delegated to existing infrastructure (VHS Video Combine). Patch Manager and Patch Assembler do not implement video encoding.
- Assembly produces an IMAGE batch that flows into the standard output chain (Video Combine), preserving meta_batch support.
- The assembly workflow is model-free. No checkpoint, VAE, or CLIP loading is required to assemble patches.
- Patch Manager is a UI/bookkeeping node with no IMAGE batch output. Patch Assembler is a data-processing node with no UI panel.

---

## 3. Patch File Format

Each patch consists of two files in the work directory, sharing a common patch ID (a UTC timestamp-based string, e.g. `20260330_143022_001`).

### 3.1 Video clip

Filename: `{id}_vace.mkv`

- Container: Matroska (MKV) with FFV1 codec — lossless, single file, widely supported by PyAV
- Content: raw VACE Inline output, including context frames on both sides, unblended
- Frame dimensions and fps match source video
- Written by VHS Video Combine (configured for lossless output) in the accumulate workflow

### 3.2 JSON sidecar

Filename: `{id}.json`

| Field | Type | Description |
|---|---|---|
| `id` | string | Patch ID, matches filename stem |
| `source_video` | string | Absolute path to the original source video |
| `start_frame` | int | VFS `start_frame` value (0-indexed) |
| `end_frame` | int | VFS `end_frame` value (0-indexed, inclusive) |
| `context_frames` | int | Context frame count used by VACE Inline |
| `clip_path` | string | Absolute path to the saved MKV clip |
| `created_at` | string | ISO 8601 UTC timestamp |
| `enabled` | bool | Whether this patch is included in assembly (default: `true`) |
| `label` | string | Optional user-supplied label (default: empty string) |

---

## 4. Patch Manager Node

Patch Manager is present in both workflows. In the accumulate workflow it receives metadata from upstream nodes, writes the JSON sidecar, and updates its UI panel. In the assemble workflow it is present as a UI-only panel with no inference inputs connected.

### 4.1 Inputs

| Input | Type | Required | Description |
|---|---|---|---|
| `clip_path` | STRING | Yes | `filename` output from Video Combine — absolute path to the saved lossless clip |
| `source_video` | STRING | Yes | `video_path` output from VFS |
| `start_frame` | INT | Yes | `start_frame` output from VFS |
| `end_frame` | INT | Yes | `end_frame` output from VFS |
| `context_frames` | INT | Yes | `context_frames` value from VACE Inline (widget passthrough or direct connection) |
| `work_dir` | STRING | Widget | Path to the patch work directory. Must match the output directory configured on Video Combine |

### 4.2 Outputs

Patch Manager has no ComfyUI outputs into the data path. All functionality is through its UI panel and side-effect file writes.

### 4.3 Behavior on Queue Run

- Reads `work_dir` to build current patch list
- If `clip_path` is provided and is a new file (not already tracked), writes a JSON sidecar alongside it
- Refreshes the UI panel to reflect the current state of the work directory
- Does not perform assembly

### 4.4 UI Panel

The panel is rendered as a custom HTML widget in the node, consistent with VFS conventions.

**Patch list**

Patches are displayed sorted ascending by `start_frame`. Each row contains:

- Enabled toggle (checkbox)
- Label — editable inline text field, stored in JSON
- Frame range — displayed as `start_frame–end_frame`
- Source video filename (basename only)
- Created timestamp (local time, human-readable)
- Conflict indicator — orange warning icon if this patch overlaps any other enabled patch
- Delete button — removes the JSON sidecar and MKV clip from disk after confirmation

**Conflict detection**

Two patches conflict if their frame ranges overlap: patch A's `end_frame` >= patch B's `start_frame` and patch A's `start_frame` <= patch B's `end_frame`. Conflict indicators are shown on all patches involved in any conflict. The indicators update immediately when enabled toggles are changed.

**Work directory size**

Total size of all files in `work_dir` is displayed at the bottom of the panel to help users manage disk usage.

---

## 5. Patch Assembler Node

Patch Assembler is present only in the assemble workflow. It reads the work directory, loads the source video and each enabled patch clip via PyAV, assembles a complete IMAGE batch with crossfade applied at context frame boundaries, and outputs the result downstream to Video Combine.

### 5.1 Inputs

| Input | Type | Description |
|---|---|---|
| `work_dir` | Widget | Path to the patch work directory. Must point to the same directory used during accumulation. |
| `source_video` | STRING | `video_path` from VFS. Assembler validates that all enabled patches reference this same path. |
| `crossfade_frames` | INT widget | Number of frames over which to blend context frames at each patch boundary. Default: 8. |
| `crossfade_curve` | Combo widget | Blend curve applied to crossfade. Options: `linear`, `ease_in`, `ease_out`, `ease_in_out`. Default: `ease_in_out`. |

### 5.2 Outputs

| Output | Type | Description |
|---|---|---|
| `IMAGE` | IMAGE | Full assembled video as an IMAGE batch, suitable for downstream Batch Images → Video Combine. |

### 5.3 Assembly Algorithm

1. Load enabled patches from `work_dir`, sorted ascending by `start_frame`
2. Validate: all enabled patches share the same `source_video` path — raise error if not
3. Validate: no enabled patches overlap in frame range — raise error if conflicts exist
4. Load source video frames via PyAV
5. For each patch in order:
   - Load patch clip frames (FFV1 MKV) via PyAV
   - Identify context boundary frames in source video at `start_frame` and `end_frame`
   - Apply crossfade: blend patch context frames with source context frames over `crossfade_frames` using the selected curve
   - Splice: replace source frames in `[start_frame, end_frame]` with blended patch frames
6. Concatenate all segments into a single IMAGE batch tensor
7. Output IMAGE batch

> **Note:** Assembly operates entirely in memory on decoded frame tensors. No intermediate video files are written. Memory footprint is proportional to total assembled frame count at source resolution.

---

## 6. Workflow Files

### 6.1 smooth_accumulate.json

Used to generate and save one patch per queue run. Inference nodes are active.

**Node chain:**

```
VFS → VACE Inline → KSampler → VAE Decode → Video Combine (lossless) → Patch Manager
```

**Additional connections to Patch Manager:**

- `VFS.video_path` → `Patch Manager.source_video`
- `VFS.start_frame` → `Patch Manager.start_frame`
- `VFS.end_frame` → `Patch Manager.end_frame`
- `VACE Inline.context_frames` → `Patch Manager.context_frames`

**Video Combine configuration:**

- Codec: FFV1 (lossless)
- Output directory: same path as Patch Manager `work_dir` widget
- Filename prefix: `patch_` (Patch Manager uses this prefix to identify patch clips vs. other files in the directory)

### 6.2 smooth_assemble.json

Used to assemble all accumulated patches into a final video. No inference nodes are present.

**Node chain:**

```
VFS → Patch Assembler → Video Combine
```

Patch Manager is also present (connected to `VFS.video_path` and `work_dir` widget) to allow the user to review and manage the patch list before queuing assembly.

> **VFS configuration note:** VFS must point to the original source video. Patch Assembler will validate that its `source_video` input matches the source recorded in each patch's JSON sidecar.

---

## 7. Re-render Support (Deferred)

Re-render — the ability to redo a single patch's VACE inference pass with different settings — is not in scope for v1. The architecture supports it without structural changes: each patch's JSON sidecar includes the parameters needed to reconstruct the VACE Inline inputs, and the control video and mask could be saved alongside the VACE clip in a future iteration.

To enable re-render in a future version:

- Save `control_video` and `control_mask` as additional files per patch (two more MKV clips or a single multi-track file)
- Add `rerender_params` to the JSON sidecar: sampler, scheduler, steps, cfg, seed
- Add a **Re-render Selected** action to the Patch Manager UI panel
- The accumulate workflow already supports this: the user selects the patch to replace and queues with updated settings

---

## 8. Open Questions

| Question | Detail |
|---|---|
| **FFV1 write support** | Confirm PyAV write support for FFV1/MKV on all target platforms (Windows primary). Read support is universal; write support should be verified before committing to this format. |
| **Patch ID format** | Timestamp-based IDs (`20260330_143022_001`) are human-readable and naturally ordered. A counter suffix handles same-second collisions. Confirm this is preferable to UUID. |
| **Filename prefix** | Using `patch_` as a Video Combine filename prefix to distinguish patch clips from other files in the work directory. Confirm this convention is acceptable, or consider a dedicated subdirectory. |
| **`context_frames` input** | VACE Inline's `context_frames` is currently a widget. Confirm whether it exposes a passthrough output, or whether Patch Manager should read it via a direct INT connection from the widget value. |
| **Crossfade validation** | Assembler should validate `crossfade_frames <= context_frames` for each patch. Decide whether this is a hard error or a warning with clamping. |
