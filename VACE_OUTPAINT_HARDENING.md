# Harden Video Outpaint inline canvas (keep resizable flex-fill)

> **Persistence:** this plan is saved into the repo at `VACE_OUTPAINT_HARDENING.md` (project root, matching the `V3_DOC_GAPS.md` convention) so it lives alongside the code.

## Context

The V3 migration moved the Video Outpaint interactive editor onto ComfyUI's
Nodes-2.0 DOM-widget sizing contract. That contract is undocumented
(`getMinHeight` is merely an option the framework happens to read via
`DOMWidgetImpl.computeLayoutSize()`) and known-unstable — frontend issues
[#7942](https://github.com/Comfy-Org/ComfyUI_frontend/issues/7942) and
[#13068](https://github.com/Comfy-Org/ComfyUI_frontend/issues/13068) show
DOM-widget sizing is buggy in the framework itself. The gaps are logged in
`V3_DOC_GAPS.md`.

Decision (this session): the editor **stays inline** and **keeps its resizable
canvas UX** (drag node taller → canvas grows). So we harden the existing
flex-fill design rather than restructure. The worst historical bug (runaway
vertical growth, commit `7501641`) is already avoided; the remaining fragility
is (1) reliance on a single undocumented `getMinHeight` knob, (2) the init
**timing dance**, (3) the accepted min-width limit.

**Invariant to preserve (the lesson from the runaway-growth bug):** *never
derive the widget/node size from the rendered DOM and write it back to the DOM
in a layout path* — that closes a feedback loop (framework derives height from
DOM → we write DOM height → framework re-derives larger → loop). All hardening
keeps sizing strictly unidirectional: framework reads callbacks/CSS vars →
sizes node → writes height onto `dom.root` → flex column absorbs it. Nothing
reads `dom.wrap.offsetHeight` to decide a height.

## What we are NOT changing

- No UX change (still inline; still resizable; canvas still grows with node height).
- No backend / workflow-format change. `crop_state`, `mask_color`,
  `custom_color` remain the single source of truth; Python `execute` is untouched.
- Frame display still uses the existing `/vace_outpaint/info` + `/frame` routes.

## Changes — all in `web/vace_outpaint.js` unless noted

### 1. Sizing-contract hardening (`addDOMWidget` block, ~lines 919–923)

Belt-and-suspenders against the undocumented contract. All read by the
framework in the same unidirectional way as `getMinHeight` — they cannot
re-introduce the feedback loop because none derive from the DOM.

- Keep `getMinHeight: () => MIN_H`.
- Add `getHeight: () => MIN_H` (preferred/default height). `getMinHeight` +
  `getHeight` + the CSS var below are the safe core; `getMaxHeight`/a max CSS
  var are **optional** — add only with a large value (e.g. 8192) if manual
  testing shows growth is being capped; do not introduce a low cap that would
  defeat "drag taller → bigger canvas."
- Set the CSS-var fallback on the widget element right where `MIN_H` is
  computed in `onNodeCreated` (~line 910), since `dom` is in scope there:
  `dom.root.style.setProperty("--comfy-widget-min-height", MIN_H + "px");`
- **Feature-detect + warn (no force-height).** On the first valid layout (in
  the post-rAF init or first `ResizeObserver` fire), if `dom.root`'s rendered
  height is materially below `MIN_H` (e.g. `< MIN_H - 8`), emit one
  `console.warn` naming the node and saying the frontend did not honor the
  DOM-widget sizing options (likely incompatible frontend version). We do
  **not** set an explicit height in response — fighting the framework's layout
  pass is exactly the feedback-prone move to avoid. Degrade visibly, never fight.
  - **One-shot guard (required).** Both the post-rAF init and the
    `ResizeObserver` (including every subsequent resize) can be the "first valid
    layout," so the check must be gated by a `st._warnedSizing` flag: check +
    warn once, set the flag, and never re-evaluate. Without the flag the warning
    fires on every resize. The var target is `--comfy-widget-min-height` on
    `dom.root` (the element passed to `addDOMWidget`); if a future frontend stops
    reading the var off that element this same warn is the safety net that
    surfaces it.

### 2. Init convergence — replace value-latching with idempotent reconcile
*(the main robustness win)*

Today the code carefully *sequences* restored widget values: `onConfigure`
latches `crop_state`/`mask_color`/`custom_color` into `st._latched*` fields,
the double-rAF applies them in the right order, and `applyFrameData` re-reads
mask color defensively ("onConfigure may not have run yet, or the rAF may have
written stale state"). Replace this ordering-dependent dance with a single
**convergent reconcile** that is safe to call any time, any number of times.

- Add `refreshFromWidgets(st, dom, widgets)` that reads the three widget values
  and reconciles `st` + DOM to match:
  - Parse `crop_state` to source-space `{x,y,w,h,outW,outH,arLocked}` and store
    as `st.pendingCrop`. If `st.initialized`, apply to `st.cr`/`st.cropAR`/
    `st.outW`/`st.outH`/`st.arLocked` immediately (reuse the logic currently in
    `restoreCropFromWidgets`, lines 145–162 — generalize it into this helper)
    **and clear `st.pendingCrop`**; otherwise leave it stashed for the next
    layout to consume.
  - Apply `mask_color`/`custom_color` to `st` and call `applyMaskColorToDOM`
    (lines 825–829).
  - **Sync the AR-lock button DOM.** After `st.arLocked` is set, call
    `setArLocked(st, dom, st.arLocked)` (lines 495–503) so the 🔒/🔓 button text,
    border, and background reflect the restored lock state. This replaces the
    explicit `setArLocked` calls currently at lines 847 and 1011 — without it a
    restored `arLocked:false` reconciles into `st` but never shows in the UI.
    (Guard: `dom.arBtn` only exists after `buildUI`, which it always does by the
    time `refreshFromWidgets` is reachable, so no null-check is needed.)
  - Idempotent: calling it again with unchanged widget values is a no-op (it only
    ever writes derived-from-widget state, never DOM-derived state).
- **`st.pendingCrop` is consumed exactly once**, at the first successful layout,
  then it is `undefined` for the rest of the node's life. Make `initLayout`
  (lines 110–140) apply `st.pendingCrop` right after `st.sf`/`st.scale` are
  computed **and immediately clear it** (so `initLayout` overrides the default
  1280×720 crop it just computed at lines 131–138 only when a pending crop
  exists). Consequences of the once-only contract, spelled out because every
  `initLayout` caller now participates:
  - **Reset button** (line 571) calls `initLayout`: by the time the user can
    click reset, the initial layout has already consumed and cleared
    `pendingCrop`, so reset correctly produces the default crop. Do **not** have
    reset repopulate `pendingCrop`.
  - **Resize path** (lines 862–866) calls `initLayout` then overrides `st.cr`
    with the preserved `prevSrc`. `pendingCrop` is already `undefined` here
    (consumed at first init), so the two never conflict. The only path that
    consumes `pendingCrop` via `initLayout` is the very first layout — either the
    post-rAF init or the `ResizeObserver` deferred-init branch, whichever wins
    the race.
- Call `refreshFromWidgets` from: end of `onConfigure` (after `origOnConfigure`
  has restored widget values), and inside the post-rAF init.
- **`applyFrameData` precedence (explicit).** `applyFrameData` runs after a real
  server run, when `st.srcW`/`st.srcH` become the *true* source dims (may differ
  from the assumed 1280×720) and it has its own crop-preservation contract via
  its `preserveCrop` param. Do **not** blanket-call `refreshFromWidgets` here —
  it would fight the existing logic. Instead keep `applyFrameData`'s existing
  three-way precedence and only fold in the reconcile's *color* half:
  1. If `preserveCrop` and there is a live in-view crop, keep it by reprojecting
     `prevSrc` into the new source space (existing lines 962/967–970).
  2. Else if the `crop_state` widget holds a valid crop, restore from it (this is
     the generalized `restoreCropFromWidgets` path — a direct call, since
     `st.initialized` is already true here so there is no pending-stash step).
  3. Else default output (existing lines 975–977).
  Replace the defensive mask-color re-read (lines 984–987) with a call that
  applies `mask_color`/`custom_color` + `applyMaskColorToDOM` (factor that color
  half of `refreshFromWidgets` into a small `refreshColorFromWidgets` helper if
  it reads cleaner, or pass a flag so crop reconciliation is skipped). Net: crop
  precedence in `applyFrameData` is unchanged; only the mask-color handling moves
  from ad-hoc re-read to the shared reconcile.
- **Remove** `st._latchedCropState` / `_latchedMaskColor` / `_latchedCustomColor`
  and every site that applies them (lines 842–846, 953–955, 1004–1009), plus the
  defensive mask-color re-read lines in `applyFrameData` (984–987). All replaced
  by the reconcile.
- Collapse the double-rAF (lines 994–995) to a single `requestAnimationFrame`.
  Geometry-readiness is already uniformly owned by the `ResizeObserver`
  (including the offscreen → visible deferred-init case; `observe()` always
  delivers an initial callback), so the second frame's "wait for width" purpose
  is redundant. Keep wiring interactions + `setupResizeObserver` inside that
  single rAF so it still defers out of `onNodeCreated`.

### 3. min-width — no code change

Confirmed unfixable (`DOMWidgetImpl.computeLayoutSize()` hardcodes
`minWidth:0`); only `node.onResize` width-clamping was ever actually shipped and
removed (git history). Keep the existing CSS containment: `dom.root`
`overflow:hidden`, control rows `flex-wrap`/`overflow-x:auto`, buttons
`flex-shrink:0`; node still opens at `MIN_W` via the one-shot `setSize`.

Optional: correct the slight overstatement in `V3_DOC_GAPS.md` item #2 (it
lists four approaches as tried; only `onResize` clamping was actually shipped,
the rest were considered-and-rejected from reading the bundle).

### 4. Optional minor hardening — blob-URL leaks (two spots)

`fetchFrame` (lines 515–527) revokes the *prior* blob each time it swaps in a new
one, but two blobs still leak. Both are trivial; fix them together since they
share the same lifecycle:

1. **On node removal.** In `onRemoved` cleanup, revoke the current frame blob if
   `dom.frameImg.src.startsWith("blob:")` — the last scrubbed frame is never
   revoked otherwise.
2. **On post-run refresh.** `applyFrameData` overwrites `dom.frameImg.src` with a
   `data:` URL (line 978) without revoking a `blob:` URL a prior `fetchFrame`
   scrub may have installed. Sequence: scrub (blob installed) → re-run →
   `execution_success` → `applyFrameData` sets a `data:` URL → the scrub blob
   leaks. Before assigning the `data:` URL, revoke the old src if it
   `startsWith("blob:")` (`data:` URLs need no revocation, so the guard is
   sufficient).

Factor the "revoke if blob:" check into a one-line helper and call it from
`fetchFrame` (already there), `applyFrameData`, and `onRemoved`.

## Files

- `web/vace_outpaint.js` — all code changes above.
- `V3_DOC_GAPS.md` — optional one-line clarification in item #2.

## Verification (manual, in ComfyUI)

Can't be run here (project file-access policy forbids reading the ComfyUI
install, and there's no ComfyUI runtime in-repo), so verify by loading the node
in a real ComfyUI and walking this matrix:

1. **Fresh node** — canvas renders at `MIN_H`, all controls visible, default
   1280×720 centered crop.
2. **Resizable preserved** — drag node taller → canvas grows (flex-fill still
   works). Drag narrower → controls wrap/scroll, nothing spills past the node
   edge (min-width containment).
3. **Restore (init convergence)** — set a non-default crop, output size, AR-lock
   off, and custom pad color; save the workflow; reload → all four restore to
   the exact saved values, crop box in the right place. Repeat with the node
   placed offscreen at load → scroll into view → initializes correctly
   (deferred-init path).
4. **Run** — queue a run → first frame + scrubber populate; scrubbing changes
   the displayed frame.
5. **Layout-stress** — toggle the right-side panel open/closed and resize the
   browser window → no collapse/explosion (the #13068 / #7942 scenarios).
6. **Regression** — no runaway vertical growth on horizontal drag (the original
   bug); no console errors on node removal; `ResizeObserver` disconnected and
   `execution_success` listener removed. AR-lock button (🔒/🔓) matches the
   restored lock state after reload (Section 2 AR-button sync).
7. **Feature-detect** — temporarily comment out the sizing callbacks → confirm
   the `console.warn` fires **exactly once** (not per-resize — the
   `st._warnedSizing` one-shot guard) and the node degrades visibly instead of
   throwing or fighting the framework.
8. **Blob hygiene** — scrub to a mid-video frame, then re-run the node; with
   DevTools memory/`URL` tracking, confirm the scrubbed blob is revoked when the
   post-run `data:` frame replaces it, and the final blob is revoked on node
   removal (Section 4).
