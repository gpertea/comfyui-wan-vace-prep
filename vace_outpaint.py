import base64
import io as _io

import numpy as np
import server
import torch
import torch.nn.functional as F
from aiohttp import web
from comfy_api.latest import io
from PIL import Image

_GRID = 16
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
                    "mask_color",
                    default="wan",
                    tooltip="Fill color preset for the outpainted region. Managed by the canvas widget.",
                ),
                io.String.Input(
                    "custom_color",
                    default="128,128,128",
                    tooltip="Custom fill color when mask_color is 'custom'. Managed by the canvas widget.",
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
    def execute(cls, images, crop_state, mask_color, custom_color) -> io.NodeOutput:
        unique_id = cls.hidden.unique_id
        n, src_h, src_w, _ = images.shape

        if src_w < _MIN_DIM or src_h < _MIN_DIM:
            raise ValueError(
                f"[VACE Outpaint] Input frame dimensions ({src_w}×{src_h}) are too small to be valid. "
                "If using VFS 'all_frames', either enable output_all_frames=True or connect 'selected_frames' instead."
            )

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
        if crop_w < _GRID or crop_h < _GRID:
            pad = max(_GRID, round(src_h * _DEFAULT_PAD_FACTOR / _GRID) * _GRID)
            crop_x = 0
            crop_y = -pad
            crop_w = round(src_w / _GRID) * _GRID
            crop_h = round((src_h + pad) / _GRID) * _GRID
            print(
                f"[VACE Outpaint] crop_state unset — using default: "
                f"x={crop_x} y={crop_y} w={crop_w} h={crop_h}"
            )

        crop_w_dim, crop_h_dim = crop_w, crop_h

        # Determine effective output resolution early so we can work at the
        # smallest possible buffer size throughout.
        eff_w = output_width  if output_width  >= _GRID else crop_w_dim
        eff_h = output_height if output_height >= _GRID else crop_h_dim

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

        _PRESETS = {"wan": (0.5, 0.5, 0.5), "ltx": (0.0, 0.0, 0.0)}
        fill_rgb = _PRESETS[mask_color] if mask_color in _PRESETS else _parse_color(custom_color)

        if eff_w == crop_w_dim and eff_h == crop_h_dim:
            # No scaling — work directly at crop-box dimensions.
            # Top-left corner of the copied region in both spaces (same for all frames).
            dst_x = max(0, -crop_x)
            dst_y = max(0, -crop_y)
            src_x = max(0, crop_x)
            src_y = max(0, crop_y)
            copy_w = min(src_w - src_x, eff_w - dst_x)
            copy_h = min(src_h - src_y, eff_h - dst_y)

            # Build the mask once (same geometry for all frames).
            mask = np.ones((eff_h, eff_w), dtype=np.float32)  # 1 = outpaint
            if copy_w > 0 and copy_h > 0:
                mask[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = 0.0

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

            # Destination of the content in the output canvas
            # (crop-box coords scaled to output coords).
            out_dst_x = round(dst_x * eff_w / crop_w_dim)
            out_dst_y = round(dst_y * eff_h / crop_h_dim)
            out_copy_w = max(1, round(copy_w * eff_w / crop_w_dim))
            out_copy_h = max(1, round(copy_h * eff_h / crop_h_dim))

            # Build the mask once at output resolution.
            mask = np.ones((eff_h, eff_w), dtype=np.float32)  # 1 = outpaint
            mask[out_dst_y:out_dst_y + out_copy_h, out_dst_x:out_dst_x + out_copy_w] = 0.0

            # Pre-allocate output tensor filled with the pad colour.
            control_video = torch.full((n, eff_h, eff_w, 3), 0.0, device=images.device)
            control_video[:, :, :, 0] = fill_rgb[0]
            control_video[:, :, :, 1] = fill_rgb[1]
            control_video[:, :, :, 2] = fill_rgb[2]

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

            control_mask = torch.from_numpy(np.stack([mask] * n))  # (N, eff_h, eff_w)

        pad_t = max(0, -crop_y)
        pad_b = max(0, crop_y + crop_h_dim - src_h)
        pad_l = max(0, -crop_x)
        pad_r = max(0, crop_x + crop_w_dim - src_w)
        scale_info = f" → output {eff_w}×{eff_h}" if (eff_w != crop_w_dim or eff_h != crop_h_dim) else ""
        print(
            f"[VACE Outpaint] {src_w}×{src_h} → crop {crop_w_dim}×{crop_h_dim}{scale_info} | "
            f"pad T={pad_t} B={pad_b} L={pad_l} R={pad_r} | frames={n}"
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
