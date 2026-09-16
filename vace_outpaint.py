import base64
import io as _io
import math

import numpy as np
import server
import torch
import torch.nn.functional as F
from aiohttp import web
from comfy_api.latest import io
from PIL import Image

# ── Consumer model table ────────────────────────────────────────────────
#
# `grid` is the model's mask cell measured in OUTPUT pixels: the VAE's spatial
# compression multiplied by the transformer's spatial patch size. A mask
# boundary that is not a multiple of the grid lands mid-cell, and the consumer
# must resolve the partial cell one of two bad ways — freeze it (pad colour is
# encoded as clean content, drawing a frame around the kept region) or
# regenerate it (up to grid-1 px of real footage replaced, which reads as edge
# blur). Emitting on the grid removes the choice: every cell is purely content
# or purely pad.
#
#   wan  8x VAE spatial compression x the DiT (1,2,2) patch          = 16
#        (comfy/ldm/wan/model.py: patch_size=(1, 2, 2))
#   ltx  32x VAE spatial compression x SymmetricPatchifier(1)        = 32
#        (comfy_extras/nodes_lt.py: height // 32; model.py: patchifier(1))
#   h3   16x VAE spatial compression x the DiT 2x2 patch             = 32
#        (see d:\work\h3-outpaint — CELL = 32 in its _quantize_mask)
#
# `rgb` is the pad colour. Grey (0.5) is the natural-image centre for a VAE
# normalised on ImageNet statistics; black sits ~2 sigma below the mean and
# writes the strongest possible edge feature into the preserved content next to
# the boundary (and is the letterbox colour in video training data). LTX keeps
# black because that is its trained convention.
_MODELS = {
    "wan": {"grid": 16, "rgb": (0.5, 0.5, 0.5)},
    "ltx": {"grid": 32, "rgb": (0.0, 0.0, 0.0)},
    "h3":  {"grid": 32, "rgb": (0.5, 0.5, 0.5)},
}
_DEFAULT_MODEL = "wan"
_MIN_GRID = 8           # smallest accepted custom_grid override
_MIN_DIM = 32           # smaller than this is not a real video frame
_DEFAULT_PAD_FACTOR = 0.3  # default upward padding as a fraction of source height

# Module-level frame cache: str(node_id) -> {"width", "height", "frame_count", "frames"}
# Populated on each node execution; consumed by the JS widget via API.
_frame_cache = {}


def _parse_color(s):
    s = s.strip().lstrip("#")
    if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return (r / 255.0, g / 255.0, b / 255.0)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"[VACE Outpaint] custom_color must be hex or 3 comma-separated values, got: {s!r}")
    vals = [float(p) for p in parts]
    if any(v > 1.0 for v in vals):
        vals = [v / 255.0 for v in vals]
    return tuple(max(0.0, min(1.0, v)) for v in vals)


def _resolve_model(model, pad_color, custom_grid, custom_color):
    """Resolve the (name, grid, fill_rgb) triple from the four widget values.

    Handles the legacy `mask_color` slot, which `model` reuses positionally:
    old workflows stored "wan" / "ltx" / "custom" there. "custom" (or anything
    unrecognised) meant "grid 16, pad colour from custom_color", so it maps to
    wan + a colour override, reproducing the old behaviour exactly.
    """
    name = model.strip() if isinstance(model, str) else ""
    if name not in _MODELS:
        name, pad_color = _DEFAULT_MODEL, "custom"
    grid = _MODELS[name]["grid"]
    try:
        override = int(custom_grid)
    except (TypeError, ValueError):
        override = 0
    if override >= _MIN_GRID:
        grid = override
    fill_rgb = _parse_color(custom_color) if pad_color == "custom" else _MODELS[name]["rgb"]
    return name, grid, fill_rgb


def _snap_dim(v, grid):
    """Nearest multiple of `grid`, at least one whole cell.

    Rounds halves up to match JS `Math.round`, so snapDim() in
    web/vace_outpaint.js and this agree on every value. Python's built-in
    round() is banker's rounding and would disagree on exact .5 cells (720 on a
    32px grid, say: 704 here vs 736 there).
    """
    return max(grid, math.floor(v / grid + 0.5) * grid)


def _align_inward(pos, length, grid):
    """Largest grid-aligned sub-interval of [pos, pos + length).

    Used where the content is copied 1:1 and must not be resampled: the kept
    region of the mask shrinks to whole cells, so the sliver of real content in
    the partial boundary cells stays in the control video (a good prior) but is
    labelled generate rather than freezing pad as content.
    """
    start = -(-pos // grid) * grid            # ceil to the grid
    end = ((pos + length) // grid) * grid     # floor to the grid
    return start, max(0, end - start)


def _snap_span(ideal_pos, ideal_len, limit, grid):
    """Snap a 1-D content span onto the grid inside [0, limit).

    Returns (pos, [len, ...]) with pos and every length a multiple of `grid`
    and pos + len <= limit. The lengths are the floor and ceil of the ideal
    length in whole cells; the caller picks between them so both axes can be
    chosen together, minimising the induced aspect error.
    """
    cells = limit // grid
    pos_cells = max(0, min(math.floor(ideal_pos / grid + 0.5), cells - 1))
    avail = cells - pos_cells
    lens = {max(1, min(int(c), avail))
            for c in (math.floor(ideal_len / grid), math.ceil(ideal_len / grid))}
    return pos_cells * grid, sorted(c * grid for c in lens)


def _require_aligned(where, x, y, w, h, grid):
    """Guard: every edge of the kept content rect must sit on the grid."""
    bad = [n for n, v in (("x", x), ("y", y), ("w", w), ("h", h)) if v % grid]
    if bad:
        raise RuntimeError(
            f"[VACE Outpaint] internal error: {where} content rect "
            f"{w}x{h} @ ({x},{y}) is not aligned to the {grid}px grid "
            f"(offending: {', '.join(bad)})"
        )


def _tensor_to_jpeg(frame_tensor):
    """Convert a (H, W, 3) float32 0-1 tensor to JPEG bytes."""
    arr = (frame_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class VACEOutpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VACEOutpaint",
            display_name="🪐 Video Outpaint",
            category="Wan VACE Prep/Video",
            description=(
                "Interactive outpaint layout: position a crop/output window over source frames "
                "to generate a VACE control video and binary outpaint mask. "
                "The output window may extend beyond the source frame on any side; "
                "overhanging regions become the mask (white = outpaint, black = source content)."
            ),
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Source video frames as an IMAGE batch.",
                ),
                io.String.Input(
                    "crop_state",
                    default="",
                    tooltip="Canvas-managed crop state: 'x,y,w,h[,ow,oh]' in source pixels. Set by the interactive widget.",
                ),
                io.String.Input(
                    "model",
                    default="wan",
                    tooltip=(
                        "Consumer model. Sets BOTH the quantisation grid (wan 16, ltx 32, "
                        "h3 32) and the default pad colour. Managed by the canvas widget."
                    ),
                ),
                io.String.Input(
                    "custom_color",
                    default="128,128,128",
                    tooltip="Pad color used when pad_color is 'custom'. Managed by the canvas widget.",
                ),
                io.String.Input(
                    "pad_color",
                    default="model",
                    tooltip="'model' uses the selected model's pad color; 'custom' uses custom_color. Managed by the canvas widget.",
                ),
                io.Int.Input(
                    "custom_grid",
                    default=0,
                    min=0,
                    max=256,
                    tooltip="Grid override in output pixels; 0 = use the model's grid. Managed by the canvas widget.",
                ),
            ],
            outputs=[
                io.Image.Output("control_video"),
                io.Mask.Output("control_mask"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("length"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    @classmethod
    def execute(cls, images, crop_state, model, custom_color, pad_color="model", custom_grid=0) -> io.NodeOutput:
        unique_id = cls.hidden.unique_id
        n, src_h, src_w, _ = images.shape

        if src_w < _MIN_DIM or src_h < _MIN_DIM:
            raise ValueError(
                f"[VACE Outpaint] Input frame dimensions ({src_w}×{src_h}) are too small to be valid. "
                "If using VFS 'all_frames', either enable output_all_frames=True or connect 'selected_frames' instead."
            )

        model_name, grid, fill_rgb = _resolve_model(model, pad_color, custom_grid, custom_color)

        # Parse crop state from widget value ("x,y,w,h" or "x,y,w,h,ow,oh").
        crop_x = crop_y = crop_w = crop_h = 0
        output_width = output_height = 0
        if crop_state:
            try:
                parts = [int(v) for v in crop_state.split(",")]
                if len(parts) >= 4:
                    crop_x, crop_y, crop_w, crop_h = parts[:4]
                if len(parts) >= 6:
                    output_width, output_height = parts[4], parts[5]
            except ValueError:
                pass

        # Fallback: widget hasn't set crop values yet (first run, uninitialized).
        if crop_w < grid or crop_h < grid:
            pad = max(grid, round(src_h * _DEFAULT_PAD_FACTOR / grid) * grid)
            crop_x = 0
            crop_y = -pad
            crop_w = round(src_w / grid) * grid
            crop_h = round((src_h + pad) / grid) * grid
            print(
                f"[VACE Outpaint] crop_state unset - using default: "
                f"x={crop_x} y={crop_y} w={crop_w} h={crop_h}"
            )

        crop_w_dim, crop_h_dim = crop_w, crop_h

        # Determine effective output resolution early so we can work at the
        # smallest possible buffer size throughout.
        eff_w = output_width  if output_width  >= grid else crop_w_dim
        eff_h = output_height if output_height >= grid else crop_h_dim

        # The output canvas itself must sit on the grid, otherwise no content
        # rect flush with the far edge can be grid-aligned. The widget snaps
        # both the crop box and the output resolution, so this only fires for a
        # hand-edited crop_state or a workflow saved against a coarser grid
        # (e.g. a 16-snapped layout reopened as h3).
        snap_w = _snap_dim(eff_w, grid)
        snap_h = _snap_dim(eff_h, grid)
        if (snap_w, snap_h) != (eff_w, eff_h):
            print(
                f"[VACE Outpaint] output {eff_w}×{eff_h} is not on the {grid}px "
                f"{model_name} grid - snapped to {snap_w}×{snap_h}"
            )
            eff_w, eff_h = snap_w, snap_h

        # Cache compressed frames so the JS widget can display them.
        _frame_cache[str(unique_id)] = {
            "width": src_w,
            "height": src_h,
            "frame_count": n,
            "frames": [_tensor_to_jpeg(images[i]) for i in range(n)],
        }

        # Build control video and mask.
        #
        # The output window top-left is (crop_x, crop_y) in source pixel space.
        # A negative crop_x / crop_y means the window extends to the left / top
        # of the source — those regions are padded (fill_rgb) and masked (white).
        #
        # For each output pixel (px, py), the corresponding source pixel is
        #   (crop_x + px, crop_y + py).
        # We copy the intersection and leave the rest as fill / masked.
        #
        # Memory-optimised path: when the crop box is much larger than the
        # requested output resolution (e.g. 4096×4096 crop → 1280×720 output),
        # we scale the source frames to the output resolution FIRST, then
        # compose at the small size.  This avoids allocating huge intermediate
        # buffers at the crop-box scale that would only be shrunk immediately
        # afterwards.
        #
        # Both paths emit a mask whose kept (black) rect is aligned to `grid`
        # on all four edges. See the model table above for why that matters.

        geom_note = ""

        if eff_w == crop_w_dim and eff_h == crop_h_dim:
            # No scaling — work directly at crop-box dimensions.
            # Top-left corner of the copied region in both spaces (same for all frames).
            dst_x = max(0, -crop_x)
            dst_y = max(0, -crop_y)
            src_x = max(0, crop_x)
            src_y = max(0, crop_y)
            copy_w = min(src_w - src_x, eff_w - dst_x)
            copy_h = min(src_h - src_y, eff_h - dst_y)

            # Content is copied 1:1 here, so the rect cannot be stretched onto
            # the grid without resampling it. Shrink the KEPT region inward to
            # whole cells instead: the partial boundary cells still hold real
            # pixels in the control video (a useful prior) but are labelled
            # generate, so no cell is part content and part pad. The content
            # edge is unaligned whenever the source dimensions are not grid
            # multiples (a 1080-tall source under a 32px grid, say) or the crop
            # origin was snapped against a coarser grid.
            keep_x, keep_w = _align_inward(dst_x, copy_w, grid) if copy_w > 0 else (0, 0)
            keep_y, keep_h = _align_inward(dst_y, copy_h, grid) if copy_h > 0 else (0, 0)

            # Build the mask once (same geometry for all frames).
            mask = np.ones((eff_h, eff_w), dtype=np.float32)  # 1 = outpaint
            if keep_w > 0 and keep_h > 0:
                _require_aligned("non-scaling", keep_x, keep_y, keep_w, keep_h, grid)
                mask[keep_y:keep_y + keep_h, keep_x:keep_x + keep_w] = 0.0

            if copy_w > 0 and copy_h > 0 and (keep_x, keep_y, keep_w, keep_h) != (dst_x, dst_y, copy_w, copy_h):
                lost = copy_w * copy_h - keep_w * keep_h
                geom_note = (
                    f" | keep {keep_w}×{keep_h} @ ({keep_x},{keep_y}) of "
                    f"{copy_w}×{copy_h} content ({lost} px in partial cells regenerated)"
                )

            control_frames = []
            for i in range(n):
                src_np = images[i].cpu().numpy()       # (src_h, src_w, 3) float32
                out    = np.full((eff_h, eff_w, 3), fill_rgb, dtype=np.float32)
                if copy_w > 0 and copy_h > 0:
                    out[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
                        src_np[src_y:src_y + copy_h, src_x:src_x + copy_w]
                control_frames.append(out)

            control_video = torch.from_numpy(np.stack(control_frames))  # (N, eff_h, eff_w, 3)
            control_mask  = torch.from_numpy(np.stack([mask] * n))      # (N, eff_h, eff_w)
        else:
            # Scaling path — work entirely at the small output resolution.
            #
            # Strategy: crop the relevant source region first, then scale it
            # directly to the output dimensions.  This avoids allocating a
            # huge crop-box-sized intermediate buffer.
            #
            # The original (non-optimised) approach would:
            #   1. Allocate (crop_w_dim × crop_h_dim) canvas, paste source region
            #   2. F.interpolate from crop-box size to (eff_h × eff_w)
            #
            # Instead we:
            #   1. Crop source to (copy_h × copy_w)
            #   2. F.interpolate directly to (out_copy_h × out_copy_w)
            #   3. Paste into the (eff_h × eff_w) output canvas
            #
            # Crop-then-scale vs scale-then-crop differ by at most a few pixels
            # of bilinear sampling offset — imperceptible for control-video use.

            # Where the source content sits in the crop-box canvas.
            dst_x = max(0, -crop_x)         # crop-box coords
            dst_y = max(0, -crop_y)
            src_x = max(0, crop_x)          # source coords
            src_y = max(0, crop_y)
            copy_w = min(src_w - src_x, crop_w_dim - dst_x)
            copy_h = min(src_h - src_y, crop_h_dim - dst_y)

            mask = np.ones((eff_h, eff_w), dtype=np.float32)  # 1 = outpaint

            # Pre-allocate output tensor filled with the pad colour.
            control_video = torch.full((n, eff_h, eff_w, 3), 0.0, device=images.device)
            control_video[:, :, :, 0] = fill_rgb[0]
            control_video[:, :, :, 1] = fill_rgb[1]
            control_video[:, :, :, 2] = fill_rgb[2]

            if copy_w > 0 and copy_h > 0:
                # Projecting the crop-box rect into output space with a plain
                # round() puts the content edges on arbitrary pixels whenever
                # the output resolution differs from the crop box, which is the
                # normal case. Snap the rect to the grid instead and resize the
                # source region to EXACTLY that rect, so mask and content stay
                # consistent by construction and no content is eroded.
                #
                # Snapping the two axes independently perturbs the preserved
                # region's aspect by up to grid/out_copy per axis, so pick the
                # floor/ceil combination whose aspect is closest to the EXACT
                # projection's, tie-broken on total scale error. The reference
                # is the projected rect, not the source region: the crop box is
                # stretched onto the output canvas whenever the two aspects
                # differ, and that stretch is intended, so measuring against
                # copy_w / copy_h would chase the wrong target.
                sx = eff_w / crop_w_dim
                sy = eff_h / crop_h_dim
                ideal_w = copy_w * sx
                ideal_h = copy_h * sy
                out_dst_x, w_cands = _snap_span(dst_x * sx, ideal_w, eff_w, grid)
                out_dst_y, h_cands = _snap_span(dst_y * sy, ideal_h, eff_h, grid)
                target_ar = ideal_w / ideal_h
                out_copy_w, out_copy_h = min(
                    ((w, h) for w in w_cands for h in h_cands),
                    key=lambda wh: (abs(wh[0] / wh[1] - target_ar),
                                    abs(wh[0] - ideal_w) + abs(wh[1] - ideal_h)),
                )
                _require_aligned("scaling", out_dst_x, out_dst_y, out_copy_w, out_copy_h, grid)

                mask[out_dst_y:out_dst_y + out_copy_h, out_dst_x:out_dst_x + out_copy_w] = 0.0

                for i in range(n):
                    # Crop the relevant region from the source frame,
                    # then scale it directly to the output sub-region size.
                    region = F.interpolate(
                        images[i: i + 1, src_y: src_y + copy_h,
                               src_x: src_x + copy_w, :].permute(0, 3, 1, 2),  # (1,3,copy_h,copy_w)
                        size=(out_copy_h, out_copy_w),
                        mode="bilinear", align_corners=False,
                    ).permute(0, 2, 3, 1).squeeze(0)  # (out_copy_h, out_copy_w, 3)
                    control_video[i, out_dst_y: out_dst_y + out_copy_h,
                                  out_dst_x: out_dst_x + out_copy_w] = region

                ar_err = (out_copy_w / out_copy_h) / target_ar - 1.0
                sc_err_x = out_copy_w / ideal_w - 1.0
                sc_err_y = out_copy_h / ideal_h - 1.0
                geom_note = (
                    f" | content {out_copy_w}×{out_copy_h} @ ({out_dst_x},{out_dst_y}) "
                    f"aspect {ar_err:+.2%} scale {sc_err_x:+.2%}/{sc_err_y:+.2%}"
                )

            control_mask = torch.from_numpy(np.stack([mask] * n))  # (N, eff_h, eff_w)

        pad_t = max(0, -crop_y)
        pad_b = max(0, crop_y + crop_h_dim - src_h)
        pad_l = max(0, -crop_x)
        pad_r = max(0, crop_x + crop_w_dim - src_w)
        scale_info = f" → output {eff_w}×{eff_h}" if (eff_w != crop_w_dim or eff_h != crop_h_dim) else ""
        print(
            f"[VACE Outpaint] {src_w}×{src_h} → crop {crop_w_dim}×{crop_h_dim}{scale_info} | "
            f"model={model_name} grid={grid} | "
            f"pad T={pad_t} B={pad_b} L={pad_l} R={pad_r} | frames={n}{geom_note}"
        )

        return io.NodeOutput(control_video, control_mask, eff_w, eff_h, n)




# ── API routes ──────────────────────────────────────────────────────────
# Serve cached frame data to the JS widget.

@server.PromptServer.instance.routes.get("/vace_outpaint/info")
async def vace_outpaint_info(request):
    """Return source dimensions, frame count, and the first frame as base64 JPEG."""
    node_id = request.query.get("node_id", "")
    if not node_id or node_id not in _frame_cache:
        return web.Response(status=404, text="No cached frame data for this node.")
    cache = _frame_cache[node_id]
    return web.json_response({
        "width":       cache["width"],
        "height":      cache["height"],
        "frame_count": cache["frame_count"],
        "frame":       base64.b64encode(cache["frames"][0]).decode(),
    })


@server.PromptServer.instance.routes.get("/vace_outpaint/frame")
async def vace_outpaint_frame(request):
    """Return a single frame by index as a JPEG image."""
    node_id = request.query.get("node_id", "")
    try:
        idx = int(request.query.get("idx", "0"))
    except ValueError:
        idx = 0
    if not node_id or node_id not in _frame_cache:
        return web.Response(status=404, text="No cached frame data for this node.")
    cache = _frame_cache[node_id]
    idx = max(0, min(idx, cache["frame_count"] - 1))
    return web.Response(body=cache["frames"][idx], content_type="image/jpeg")
