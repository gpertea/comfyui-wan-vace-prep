import torch
import torch.nn.functional as F
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
                "preserving the rest. An optional reference image is prepended as a "
                "context frame with mask=0."
            ),
            is_experimental=True,
            inputs=[
                io.Image.Input("video"),
                io.Mask.Input("mask"),
                io.Image.Input("reference_image", optional=True),
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
    def execute(cls, video, mask, reference_image=None) -> io.NodeOutput:
        N, H, W, C = video.shape

        if W % 16 != 0 or H % 16 != 0:
            raise ValueError(
                f"[WanVACEInpaint] Video dimensions ({W}x{H}) must both be "
                f"divisible by 16."
            )

        ## normalize mask to [N, H, W].
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

        ## gray out masked pixels (1 = regenerate -> 0.5 placeholder).
        masked_video = video.clone()
        mask_bool = mask > 0.5
        masked_video[mask_bool] = 0.5

        if reference_image is not None:
            ref = reference_image[0:1]
            ref_h, ref_w = ref.shape[1], ref.shape[2]
            if ref_h != H or ref_w != W:
                print(
                    f"[WanVACEInpaint] Resizing reference image from "
                    f"{ref_w}x{ref_h} to {W}x{H}"
                )
                ref = ref.permute(0, 3, 1, 2)
                ref = F.interpolate(ref, size=(H, W), mode="bilinear", align_corners=False)
                ref = ref.permute(0, 2, 3, 1)

            control_video = torch.cat([ref, masked_video], dim=0)
            ref_mask = torch.zeros(1, H, W, dtype=mask.dtype, device=mask.device)
            control_mask = torch.cat([ref_mask, mask], dim=0)
            length = N + 1
        else:
            control_video = masked_video
            control_mask = mask
            length = N

        return io.NodeOutput(control_video, control_mask, W, H, length)
