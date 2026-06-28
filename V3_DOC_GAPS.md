# ComfyUI V3 / Nodes 2.0 - Documentation & API Gaps

A running list of things that cost real time during V3 migration because they were
undocumented, under-documented, or simply missing from the public API.

Each entry records: what we were trying to do, what we expected, what actually happened
(and what we had to do instead), and what would have prevented the friction.

Frontend version these were observed against: `comfyui_frontend_package` 1.45.19
(findings re-confirmed unchanged from 1.36.13).

---

## 1. DOM-widget vertical sizing contract is undocumented

**Goal:** A custom node with an interactive `addDOMWidget` canvas that resizes vertically
with the node.

**Expected:** Public guidance on how a DOM widget reports its desired/min height and how
the node height relates to it.

**What happened:** The V1 model (set the inner element height from `onResize`, override
`domWidget.computeSize`) is inverted under Nodes 2.0 and produces a feedback loop -
dragging the node horizontally caused runaway vertical growth. The actual contract had to
be reverse-engineered from the minified bundle: pass `getMinHeight` / `getHeight` /
`getMaxHeight` in the `addDOMWidget` options; these feed `DOMWidgetImpl.computeLayoutSize()`
(with CSS-var fallbacks `--comfy-widget-min-height` / `--comfy-widget-max-height`); the
framework then writes an explicit pixel height onto the widget root element. The correct
pattern is to make the root a flex column and let an inner `flex:1` child absorb the height,
with a `ResizeObserver` re-laying-out the content.

**Would fix it:** Document the `addDOMWidget` options (`getMinHeight`, `getHeight`,
`getMaxHeight`, `getMinWidth`?, the CSS vars) and the "framework sizes the element, you fill
it" model. A short migration note that `onResize`/`computeSize`-driven sizing is V1-only
would have saved the most time.

## 2. No way to set a minimum WIDTH on a DOM-widget node

**Goal:** Stop the node from being dragged narrower than its controls (buttons overhang the
node edge).

**Expected:** A `getMinWidth` option (parallel to `getMinHeight`), or a CSS var, or an
honoured node/widget min-width.

**What happened:** `DOMWidgetImpl.computeLayoutSize()` hardcodes `minWidth: 0`. There is no
option, no CSS var, and no instance override that the Vue layout honours (wrapping
`domWidget.computeLayoutSize` did not take - the layout reads an internal object, not what
`addDOMWidget` returns). `node.onResize` size mutation is ignored; `setSize` from `onResize`
loses to the drag controller and glitches the wires; overriding `node.computeSize` only
affects the legacy LiteGraph resize path, which the Vue renderer ignores. We gave up on a
hard floor and contained the controls with CSS (`overflow:hidden` on the root, wrap/scroll
on rows) instead.

**Would fix it:** Add a `getMinWidth` option (and `--comfy-widget-min-width` CSS var) to DOM
widgets, symmetric with the height handling that already exists.

## 3. Internal modules are the only source of truth for widget layout

**Goal:** Understand the widget layout/measurement model.

**What happened:** `scripts/domWidget.js` is a shim that warns it is "an internal module, not
part of the public API." The real `DOMWidgetImpl` / `computeLayoutSize` / layout-cell code
lives only in the minified `assets/index-*.js` bundle. Any non-trivial DOM-widget node has
to read that bundle to behave correctly, which is brittle across frontend releases.

**Would fix it:** A documented, stable public surface for DOM-widget sizing/layout so custom
nodes do not have to depend on internal implementation details.

## 4. V3 entrypoint vs NODE_CLASS_MAPPINGS mutual exclusivity is not obvious

**Goal:** Register V3 nodes via `comfy_entrypoint()` / `ComfyExtension`.

**What happened:** The loader is mutually exclusive: if a module defines
`NODE_CLASS_MAPPINGS`, `comfy_entrypoint()` is silently ignored (ComfyUI `nodes.py`,
loader: `if module has NODE_CLASS_MAPPINGS: use it; elif comfy_entrypoint: use it`). Having
both in one `__init__.py` during an incremental migration leads to the entrypoint nodes
silently not loading, with no error.

**Would fix it:** Document the precedence explicitly, ideally with a startup warning when
both are present in the same module.

---

## Template for new entries

```
## N. Short title

**Goal:** ...
**Expected:** ...
**What happened:** ...
**Would fix it:** ...
```
