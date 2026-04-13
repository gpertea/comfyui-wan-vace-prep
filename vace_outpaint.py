import base64
import io

import numpy as np
import server
import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image

_GRID = 16
_MIN_DIM = 32           # smaller than this is not a real video frame
_DEFAULT_PAD_FACTOR = 0.3  # default upward padding as a fraction of source height

# Module-level frame cache: str(node_id) -> {"width", "height", "frame_count", "frames"}
# Populated on each node execution; consumed by the JS widget via API.
_frame_cache = {}


def _tensor_to_jpeg(frame_tensor):
    """Convert a (H, W, 3) float32 0-1 tensor to JPEG bytes."""
    arr = (frame_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class VACEOutpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Source video frames as an IMAGE batch."
                }),
                "crop_state": ("STRING", {
                    "default": "",
                    "tooltip": "Canvas-managed crop state: 'x,y,w,h[,ow,oh]' in source pixels. Set by the interactive widget.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("control_video", "control_mask", "width", "height", "length")
    FUNCTION = "outpaint_prep"
    OUTPUT_NODE = True
    CATEGORY = "video/VACE"
    DESCRIPTION = (
        "Interactive outpaint layout: position a crop/output window over source frames "
        "to generate a VACE control video and binary outpaint mask. "
        "The output window may extend beyond the source frame on any side; "
        "overhanging regions become the mask (white = outpaint, black = source content)."
    )

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def outpaint_prep(self, images, crop_state, unique_id):
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

        out_w, out_h = crop_w, crop_h

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
        # of the source — those regions are padded (gray, 0.5) and masked (white).
        #
        # For each output pixel (px, py), the corresponding source pixel is
        #   (crop_x + px, crop_y + py).
        # We copy the intersection and leave the rest as black / masked.

        # Top-left corner of the copied region in both spaces (same for all frames).
        dst_x = max(0, -crop_x)
        dst_y = max(0, -crop_y)
        src_x = max(0, crop_x)
        src_y = max(0, crop_y)
        copy_w = min(src_w - src_x, out_w - dst_x)
        copy_h = min(src_h - src_y, out_h - dst_y)

        # Build the mask once (same geometry for all frames).
        mask = np.ones((out_h, out_w), dtype=np.float32)  # 1 = outpaint
        if copy_w > 0 and copy_h > 0:
            mask[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = 0.0

        control_frames = []
        mask_frames = []

        for i in range(n):
            src_np = images[i].cpu().numpy()       # (src_h, src_w, 3) float32
            out    = np.full((out_h, out_w, 3), 0.5, dtype=np.float32)

            if copy_w > 0 and copy_h > 0:
                out[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
                    src_np[src_y:src_y + copy_h, src_x:src_x + copy_w]

            control_frames.append(out)
            mask_frames.append(mask)

        control_video = torch.from_numpy(np.stack(control_frames))  # (N, out_h, out_w, 3)
        control_mask  = torch.from_numpy(np.stack(mask_frames))     # (N, out_h, out_w)

        # Scale to requested output resolution if different from crop box size.
        eff_w = output_width  if output_width  >= _GRID else out_w
        eff_h = output_height if output_height >= _GRID else out_h

        if eff_w != out_w or eff_h != out_h:
            cv = control_video.permute(0, 3, 1, 2)                               # (N,3,H,W)
            cv = F.interpolate(cv, size=(eff_h, eff_w), mode="bilinear", align_corners=False)
            control_video = cv.permute(0, 2, 3, 1)                               # (N,H,W,3)

            cm = control_mask.unsqueeze(1).float()                               # (N,1,H,W)
            cm = F.interpolate(cm, size=(eff_h, eff_w), mode="nearest")
            control_mask = cm.squeeze(1)                                          # (N,H,W)

        pad_t = max(0, -crop_y)
        pad_b = max(0, crop_y + crop_h - src_h)
        pad_l = max(0, -crop_x)
        pad_r = max(0, crop_x + crop_w - src_w)
        scale_info = f" → output {eff_w}×{eff_h}" if (eff_w != out_w or eff_h != out_h) else ""
        print(
            f"[VACE Outpaint] {src_w}×{src_h} → crop {out_w}×{out_h}{scale_info} | "
            f"pad T={pad_t} B={pad_b} L={pad_l} R={pad_r} | frames={n}"
        )

        return (control_video, control_mask, eff_w, eff_h, n)




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
