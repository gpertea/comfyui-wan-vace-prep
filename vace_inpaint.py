import torch
from comfy_api.latest import io


class WanVACEInpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanVACEInpaint",
            display_name="🪐 VACE Inpaint (Experimental)",
            category="Wan VACE Prep/VACE",
            description=(
                "Prepares a video for VACE inpainting. Masked regions (mask=1) are "
                "replaced with a gray placeholder so Wan VACE will regenerate them while "
                "preserving the rest."
            ),
            is_experimental=True,
            inputs=[
                io.Image.Input("video"),
                io.Mask.Input("mask"),
            ],
            outputs=[
                io.Image.Output("control_video"),
                io.Mask.Output("control_mask"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("length"),
            ],
        )

    @classmethod
    def execute(cls, video, mask) -> io.NodeOutput:
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

        return io.NodeOutput(masked_video, mask, W, H, N)
