import torch


class WanVACEPrepBase:
    def _validate_dimensions(self, video_1, video_2):
        """Validate that both videos have matching, 16-divisible dimensions."""
        height = video_1.shape[1]
        width = video_1.shape[2]

        if video_2.shape[1] != height or video_2.shape[2] != width:
            raise ValueError(
                f"Video dimensions must match. "
                f"video_1 is {width}x{height}, "
                f"video_2 is {video_2.shape[2]}x{video_2.shape[1]}"
            )

        if width % 16 != 0 or height % 16 != 0:
            raise ValueError(
                f"Video dimensions must be divisible by 16. "
                f"Current dimensions: {width}x{height}"
            )

        return width, height

    def _extract_context(self, video_1, video_2, context_frames, replace_frames):
        """Extract context frames from each video edge for the control video."""
        if replace_frames > 0:
            v1_context = video_1[-(context_frames + replace_frames):-replace_frames]
            v2_context = video_2[replace_frames:context_frames + replace_frames]
        else:
            v1_context = video_1[-context_frames:]
            v2_context = video_2[:context_frames]
        return v1_context, v2_context

    def _build_control_video_and_mask(self, video_1, v1_context, v2_context,
                                      context_frames, replace_frames, new_frames,
                                      height, width):
        """Build the control video tensor and mask."""
        channels = video_1.shape[3]

        # Wan wants to generate 4n+1 frames. If we don't provide that,
        # it will quietly round down to the nearest 4n+1. So we add 1 here.
        vace_count = (replace_frames * 2) + new_frames + 1
        vace_frames = torch.full(
            (vace_count, height, width, channels), 0.5,
            dtype=video_1.dtype, device=video_1.device
        )

        control_video = torch.cat([v1_context, vace_frames, v2_context], dim=0)

        total_frames = (context_frames * 2) + vace_count
        mask = torch.zeros((total_frames, height, width), dtype=torch.float32, device=video_1.device)
        mask[context_frames:context_frames + vace_count] = 1.0

        return control_video, mask, vace_count
