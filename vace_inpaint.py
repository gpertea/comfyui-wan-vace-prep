import torch


class WanVACEInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("control_video", "control_mask", "width", "height", "length")
    FUNCTION = "run"
    CATEGORY = "Wan VACE Prep"
    DESCRIPTION = (
        "Prepares a video for VACE inpainting. Masked regions (mask=1) are "
        "replaced with a gray placeholder so Wan VACE will regenerate them while "
        "preserving the rest."
    )
    EXPERIMENTAL = True
    
    def run(self, video, mask):
        N, H, W, C = video.shape

        if W % 16 != 0 or H % 16 != 0:
            raise ValueError(
                f"[WanVACEInpaint] Video dimensions ({W}x{H}) must both be "
                f"divisible by 16."
            )

        # Normalize mask to [N, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(N, -1, -1).contiguous()
        elif mask.ndim == 3:
            if mask.shape[0] == 1 and N > 1:
                mask = mask.expand(N, -1, -1).contiguous()
            elif mask.shape[0] != N:
                raise ValueError(
                    f"[WanVACEInpaint] Mask frame count ({mask.shape[0]}) does "
                    f"not match video frame count ({N})."
                )
        else:
            raise ValueError(
                f"[WanVACEInpaint] Unexpected mask shape: {list(mask.shape)}. "
                f"Expected [H, W] or [N, H, W]."
            )

        # Gray out masked pixels (1 = regenerate -> 0.5 placeholder)
        masked_video = video.clone()
        mask_bool = mask > 0.5  # [N, H, W]
        masked_video[mask_bool] = 0.5

        return (masked_video, mask, W, H, N)


