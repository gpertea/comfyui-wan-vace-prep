import torch
from comfy_api.latest import io


class WanVACEExtend(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanVACEExtend",
            display_name="🪐 VACE Extend",
            category="Wan VACE Prep/VACE",
            description=(
                "Generates VACE control video and mask for extending a video from an\n"
                "arbitrary position using context frames."
            ),
            inputs=[
                io.Image.Input("video"),
                io.Int.Input("extend_from_idx", default=-1, min=-1000000, max=1000000, step=1,
                    tooltip="Frame index to extend from. Negative values count from the end of the video. e.g., -1 is last frame"),
                io.Int.Input("context_frames", default=8, min=4, max=120, step=4,
                    tooltip="Number of reference frames before extend_from_idx for VACE conditioning (multiple of 4)."),
                io.Int.Input("new_frames", default=25, min=1, max=241, step=4,
                    tooltip="Number of new frames to generate (4n+1)."),
            ],
            outputs=[
                io.Image.Output("control_video"),
                io.Mask.Output("control_mask"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("length"),
                io.Image.Output("start_images"),
                io.Int.Output("context_frames"),
                io.Int.Output("new_frames"),
            ],
        )

    @classmethod
    def execute(cls, video, extend_from_idx, context_frames, new_frames) -> io.NodeOutput:
        height = video.shape[1]
        width = video.shape[2]
        video_length = video.shape[0]

        if width % 16 != 0 or height % 16 != 0:
            raise ValueError(
                f"Video dimensions must be divisible by 16. "
                f"Current dimensions: {width}x{height}"
            )

        if (new_frames - 1) % 4 != 0:
            raise ValueError(
                f"new_frames must follow 4n+1 pattern (1, 5, 9, 13, 17, 21, 25, ...). "
                f"Got {new_frames}. "
                f"Nearest valid values: {((new_frames - 1) // 4) * 4 + 1} or {((new_frames - 1) // 4 + 1) * 4 + 1}"
            )

        if extend_from_idx < 0:
            actual_extend_idx = video_length + extend_from_idx
        else:
            actual_extend_idx = extend_from_idx

        if actual_extend_idx < 0 or actual_extend_idx >= video_length:
            raise ValueError(
                f"extend_from_idx resolves to {actual_extend_idx}, which is out of bounds for video length {video_length}. "
                f"Valid range: 0 to {video_length - 1} (or -{video_length} to -1 for negative indexing)."
            )

        if context_frames > actual_extend_idx:
            raise ValueError(
                f"context_frames ({context_frames}) requires at least {context_frames} frames before extend_from_idx, "
                f"but extend_from_idx is at position {actual_extend_idx}. "
                f"Maximum context_frames for this position: {actual_extend_idx}"
            )

        if context_frames > 0:
            context_start = actual_extend_idx - context_frames
            video_context = video[context_start:actual_extend_idx]
        else:
            video_context = torch.empty((0, height, width, video.shape[3]), dtype=video.dtype, device=video.device)

        channels = video.shape[3]

        vace_frames = torch.full((new_frames, height, width, channels), 0.5, dtype=video.dtype, device=video.device)

        if context_frames > 0:
            control_video = torch.cat([video_context, vace_frames], dim=0)
        else:
            control_video = vace_frames

        vace_count = context_frames + new_frames
        mask = torch.zeros((vace_count, height, width), dtype=torch.float32, device=video.device)
        mask[context_frames:] = 1.0

        cut_point = actual_extend_idx - context_frames
        start_images = video[:cut_point]

        length = int(control_video.shape[0])

        return io.NodeOutput(control_video, mask, width, height, length, start_images, context_frames, new_frames)
