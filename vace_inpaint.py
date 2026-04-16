import torch
import torch.nn.functional as F


class WanVACEInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("control_video", "control_mask", "width", "height", "length")
    FUNCTION = "run"
    CATEGORY = "Wan VACE Prep"
    DESCRIPTION = (
        "Prepares a video for VACE inpainting. Masked regions (mask=1) are "
        "replaced with a gray placeholder so Wan VACE will regenerate them while "
        "preserving the rest. An optional reference image is prepended as a "
        "context frame (mask=0) to guide generation."
    )
    EXPERIMENTAL = True
    
    def run(self, video, mask, reference_image=None):
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

        if reference_image is not None:
            ref = reference_image[0:1]  # [1, H_ref, W_ref, C]
            ref_h, ref_w = ref.shape[1], ref.shape[2]
            if ref_h != H or ref_w != W:
                print(
                    f"[WanVACEInpaint] Resizing reference image from "
                    f"{ref_w}x{ref_h} to {W}x{H}"
                )
                ref = ref.permute(0, 3, 1, 2)  # [1, C, H_ref, W_ref]
                ref = F.interpolate(ref, size=(H, W), mode="bilinear", align_corners=False)
                ref = ref.permute(0, 2, 3, 1)  # [1, H, W, C]

            control_video = torch.cat([ref, masked_video], dim=0)
            ref_mask = torch.zeros(1, H, W, dtype=mask.dtype, device=mask.device)
            control_mask = torch.cat([ref_mask, mask], dim=0)
            length = N + 1
        else:
            control_video = masked_video
            control_mask = mask
            length = N

        return (control_video, control_mask, W, H, length)


