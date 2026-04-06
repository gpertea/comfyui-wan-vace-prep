import torch
from .vace_base import WanVACEPrepBase


class VACEInline(WanVACEPrepBase):
    """Single-video VACE prep: splits at a VFS selection boundary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Full source video as an IMAGE batch (e.g. from Load Video)."
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "tooltip": "First frame of the VFS selection (0-based index)."
                }),
                "end_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "tooltip": "Last frame of the VFS selection, exclusive (0-based index)."
                }),
                "context_frames": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 120,
                    "step": 4,
                    "tooltip": "Reference frames from each video edge for VACE interpolation (multiple of 4)."
                }),
                "new_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 240,
                    "step": 4,
                    "tooltip": "Additional frames to insert within the selected range, expanding the video length (multiple of 4)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("control_video", "control_mask", "width", "height", "length", "start_images", "end_images", "context_frames", "new_frames")
    FUNCTION = "vace_prep_inline"
    CATEGORY = "video/VACE"
    DESCRIPTION = """
    Generates VACE control video and mask for smooth in-place regeneration
    of a selected frame range within a single video. The selected range
    (start_frame to end_frame) defines what VACE regenerates. context_frames
    on each side condition the generation. new_frames optionally inserts
    additional frames within the selected range, expanding the video length.
    """
    EXPERIMENTAL = True

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def vace_prep_inline(self, images, start_frame, end_frame, context_frames, new_frames):
        total_frames = images.shape[0]

        # ── 1. Validate selection bounds ────────────────────────────────
        self._validate_selection(start_frame, end_frame, total_frames)

        # ── 2. Split at selection boundary ─────────────────────────────
        # video_1: everything up to (but not including) the selection
        # video_2: everything from end_frame onwards
        # The selected range itself is treated as the gap VACE will fill/smooth.
        video_1 = images[:start_frame]
        video_2 = images[end_frame:]

        # ── 3. Validate dimensions (delegates to base) ──────────────────
        width, height = self._validate_dimensions(video_1, video_2)

        # ── 4. Validate each half has enough frames for context ──────────
        if video_1.shape[0] < context_frames:
            raise ValueError(
                f"The region before start_frame ({start_frame}) contains only "
                f"{video_1.shape[0]} frame(s), but context_frames ({context_frames}) are required. "
                f"Move start_frame further into the clip or reduce context_frames."
            )

        if video_2.shape[0] < context_frames:
            frames_after = total_frames - end_frame
            raise ValueError(
                f"The region after end_frame ({end_frame}) contains only "
                f"{video_2.shape[0]} frame(s) ({frames_after} in the original clip), "
                f"but context_frames ({context_frames}) are required. "
                f"Move end_frame earlier or reduce context_frames."
            )

        # ── 5. Build control video and mask ─────────────────────────────
        # replace_frames=0: context is taken from the outermost context_frames
        # of each side. The selected range length drives vace_count via new_frames
        # augmented by the selection size.
        #
        # Wan requires 4n+1 frames. The base class adds 1 to vace_count, so we
        # need new_frames to be a multiple of 4. Snap selection_size up to the
        # nearest 4n+1 before subtracting 1 so that new_frames ends up as 4n.
        #
        # Example: selection_size=10 → remainder=10%4=2 → (1-2)%4=3 → snapped=13 (4*3+1)
        #          new_frames = snapped - 1 + user_new_frames = 12 + 0 = 12 (4n) ✓
        selection_size = end_frame - start_frame
        remainder = selection_size % 4
        snapped_size = selection_size if remainder == 1 else selection_size + ((1 - remainder) % 4)
        if snapped_size != selection_size:
            print(
                f"[VACE Inline] Selection size {selection_size} is not 4n+1. "
                f"Snapping up to {snapped_size} frames. "
                f"Output will be {snapped_size - selection_size} frame(s) longer than input."
            )
        v1_context, v2_context = self._extract_context(video_1, video_2, context_frames, replace_frames=0)
        control_video, mask, _ = self._build_control_video_and_mask(
            video_1, v1_context, v2_context,
            context_frames, replace_frames=0,
            new_frames=snapped_size - 1 + new_frames,
            height=height, width=width
        )

        # ── 6. Compute pass-through frame ranges ─────────────────────────
        start_images = video_1[:-context_frames]
        end_images = video_2[context_frames:]
        length = int(control_video.shape[0])

        return (control_video, mask, width, height, length, start_images, end_images, context_frames, new_frames)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_selection(self, start_frame, end_frame, total_frames):
        """Ensure the VFS selection is sane relative to the video length."""
        if start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {start_frame}.")

        if end_frame <= start_frame:
            raise ValueError(
                f"end_frame ({end_frame}) must be greater than start_frame ({start_frame})."
            )

        if end_frame > total_frames:
            raise ValueError(
                f"end_frame ({end_frame}) exceeds video length ({total_frames} frames)."
            )

        if start_frame == 0:
            raise ValueError(
                f"start_frame is 0, which leaves no frames before the selection. "
                f"The split requires at least context_frames frames on each side."
            )

        if end_frame == total_frames:
            raise ValueError(
                f"end_frame ({end_frame}) equals the total video length ({total_frames}), "
                f"which leaves no frames after the selection. "
                f"The split requires at least context_frames frames on each side."
            )

NODE_CLASS_MAPPINGS = {
    "VACEInline": VACEInline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VACEInline": "🪐 VACE Inline (Experimental)"
}
