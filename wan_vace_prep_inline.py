import torch
from .wan_vace_prep import WanVACEPrepBase


class VACESmooth(WanVACEPrepBase):
    """
    VACE prep for a single video with an inline selection boundary.

    Instead of taking two separate video clips, this node accepts one
    continuous video (an IMAGE batch from Load Video) plus a frame
    selection from Visual Frame Selector, then splits the video at
    the selection boundary and runs the same control-building logic
    as WanVACEPrep.

    Typical use-case: you have one long clip and want VACE to smooth
    a transition at a specific point inside it without pre-splitting
    the clip manually.

    Parameters
    ----------
    images      : IMAGE batch — the full source video
    start_frame : INT  — first frame of the selected range (from VFS)
    end_frame   : INT  — last frame of the selected range, exclusive (from VFS)
    context_frames : frames of context to pull from each side of the split
    replace_frames : frames to regenerate at each transition edge
    add_frames     : new frames to insert between the two sides
                     (equivalent to new_frames in WanVACEPrep)

    Output signature mirrors WanVACEPrep exactly so the downstream
    sampler wiring is identical.
    """

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
                    "max": 99999,
                    "tooltip": "First frame of the VFS selection (0-based index)."
                }),
                "end_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "tooltip": "Last frame of the VFS selection, exclusive (0-based index)."
                }),
                "context_frames": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 120,
                    "step": 4,
                    "tooltip": "Reference frames from each video edge for VACE interpolation (multiple of 4)."
                }),
                "replace_frames": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 120,
                    "step": 4,
                    "tooltip": "Number of frames to regenerate at each transition edge (multiple of 4)."
                }),
                "add_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 240,
                    "step": 4,
                    "tooltip": "Number of new transition frames to generate in addition to replace_frames (multiple of 4)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("control_video", "control_mask", "width", "height", "length", "start_images", "end_images")
    FUNCTION = "vace_prep_inline"
    CATEGORY = "video/VACE"
    DESCRIPTION = (
        "Inline VACE prep: splits a single video at the VFS selection boundary "
        "and generates the control video and mask for smooth in-clip transitions. "
        "Output signature matches Wan VACE Prep for drop-in compatibility."
    )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def vace_prep_inline(self, images, start_frame, end_frame, context_frames, replace_frames, add_frames):
        total_frames = images.shape[0]

        # ── 1. Validate selection bounds ────────────────────────────────
        self._validate_selection(start_frame, end_frame, total_frames)

        # ── 2. Split at selection boundary ─────────────────────────────
        # video_1: everything up to (but not including) the selection
        # video_2: everything from end_frame onwards
        # The selected range itself is treated as the gap VACE will fill/smooth.
        split_point = start_frame          # last frame of video_1 group
        video_1 = images[:split_point] if split_point > 0 else images[:1]
        video_2 = images[end_frame:] if end_frame < total_frames else images[-1:]

        # ── 3. Validate dimensions (delegates to base) ──────────────────
        width, height = self._validate_dimensions(video_1, video_2)

        # ── 4. Validate each half has enough frames ─────────────────────
        required = context_frames + replace_frames

        if video_1.shape[0] < required:
            raise ValueError(
                f"The region before start_frame ({start_frame}) contains only "
                f"{video_1.shape[0]} frame(s), but context_frames ({context_frames}) + "
                f"replace_frames ({replace_frames}) = {required} are required. "
                f"Move start_frame further into the clip or reduce context/replace values."
            )

        if video_2.shape[0] < required:
            frames_after = total_frames - end_frame
            raise ValueError(
                f"The region after end_frame ({end_frame}) contains only "
                f"{video_2.shape[0]} frame(s) ({frames_after} in the original clip), "
                f"but context_frames ({context_frames}) + replace_frames ({replace_frames}) "
                f"= {required} are required. "
                f"Move end_frame earlier or reduce context/replace values."
            )

        # ── 5. Build control video and mask (shared base logic) ──────────
        v1_context, v2_context = self._extract_context(video_1, video_2, context_frames, replace_frames)
        control_video, mask, _ = self._build_control_video_and_mask(
            video_1, v1_context, v2_context,
            context_frames, replace_frames, add_frames,
            height, width
        )

        # ── 6. Compute pass-through frame ranges ─────────────────────────
        # start_images: the portion of video_1 before the context/replace window
        # end_images:   the portion of video_2 after the context/replace window
        start_images = video_1[:-(context_frames + replace_frames)]
        end_images = video_2[context_frames + replace_frames:]
        length = int(control_video.shape[0])

        return (control_video, mask, width, height, length, start_images, end_images)

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
                f"The split requires at least context_frames + replace_frames frames on each side."
            )

        if end_frame == total_frames:
            raise ValueError(
                f"end_frame ({end_frame}) equals the total video length ({total_frames}), "
                f"which leaves no frames after the selection. "
                f"The split requires at least context_frames + replace_frames frames on each side."
            )


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "VACESmooth": VACESmooth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VACESmooth": "🪐 VACE Smooth",
}
