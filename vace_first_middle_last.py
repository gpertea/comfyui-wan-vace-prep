import torch
import torch.nn.functional as F


def _resize_frames(frames, width, height):
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    x = frames.movedim(-1, 1).float()
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.movedim(1, -1).to(frames.dtype)


class WanVACEFirstMiddleLast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {
                    "default": 81, "min": 1, "max": 16384, "step": 4,
                    "tooltip": "Total frame count. Must follow the 4n+1 pattern (1, 5, 9, ..., 81, ...)."
                }),
                "middle_position": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Where middle frames are centered in the video, as a fraction of total length."
                }),
            },
            "optional": {
                "first": ("IMAGE",),
                "middle": ("IMAGE",),
                "last": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("control_video", "control_mask", "width", "height", "length")
    FUNCTION = "run"
    CATEGORY = "video/VACE"
    DESCRIPTION = (
        "Builds a VACE control video and mask from optional first, middle, and last "
        "frame batches. Known frames are placed at their positions with mask=0; "
        "remaining frames are gray placeholders with mask=1 for generation."
    )
    EXPERIMENTAL = True

    def run(self, width, height, length, middle_position, first=None, middle=None, last=None):
        snapped_width = (width // 16) * 16
        snapped_height = (height // 16) * 16
        snapped_length = ((length - 1) // 4) * 4 + 1

        if snapped_width != width or snapped_height != height:
            print(
                f"[WanVACEFirstMiddleLast] Dimensions snapped to 16px grid: "
                f"{width}x{height} -> {snapped_width}x{snapped_height}"
            )
        if snapped_length != length:
            print(
                f"[WanVACEFirstMiddleLast] length snapped to 4n+1: "
                f"{length} -> {snapped_length}"
            )

        width = snapped_width
        height = snapped_height
        length = snapped_length

        control = torch.full((length, height, width, 3), 0.5, dtype=torch.float32)
        mask = torch.ones((length, height, width), dtype=torch.float32)

        n_first = 0
        n_last = 0

        if first is not None:
            first = _resize_frames(first, width, height)
            n_first = min(first.shape[0], length)
            control[:n_first] = first[:n_first, :, :, :3]
            mask[:n_first] = 0.0

        if last is not None:
            last = _resize_frames(last, width, height)
            n_last = min(last.shape[0], length - n_first)
            if n_last > 0:
                control[length - n_last:] = last[-n_last:, :, :, :3]
                mask[length - n_last:] = 0.0

        if middle is not None:
            middle = _resize_frames(middle, width, height)
            zone_start = n_first
            zone_end = length - n_last
            available = zone_end - zone_start

            if available > 0:
                n_mid = min(middle.shape[0], available)
                target_center = round(middle_position * (length - 1))
                mid_start = target_center - n_mid // 2
                mid_start = max(mid_start, zone_start)
                mid_start = min(mid_start, zone_end - n_mid)
                mid_end = mid_start + n_mid
                control[mid_start:mid_end] = middle[:n_mid, :, :, :3]
                mask[mid_start:mid_end] = 0.0

        return (control, mask, width, height, length)
