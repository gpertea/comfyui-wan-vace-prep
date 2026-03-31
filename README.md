# Visual Frame Selector

ComfyUI nodes for interactive frame selection and single-video VACE preparation.

> **Experimental.** These nodes are functional but may have rough edges, and their interfaces may change between releases.

## Nodes

### Visual Frame Selector

Interactive video frame selector with a visual scrubber, transport controls, and in/out markers. Designed for selecting frame ranges to feed into VACE Inline or other frame-range-aware nodes.

- Upload a video and scrub through it visually
- Set start/end markers by dragging or using transport buttons
- Outputs both the selected frame range and the full video

**Parameters:**

| Parameter | Default | Description |
|-|-|-|
| video | | Video file (upload via ComfyUI's file selector) |
| current_frame | 0 | Current scrubber position (driven by the visual controls) |
| start_frame | 0 | Start of selected range |
| end_frame | 0 | End of selected range |

**Outputs:**

| Output | Description |
|-|-|
| selected_frames | Frames within the selected range |
| all_frames | Every frame in the video |
| selected_count | Number of frames in the selection |
| frame_count | Total frames in the video |
| start_frame | Resolved start frame index |
| end_frame | Resolved end frame index |
| fps | Video frame rate |
| audio | Audio track (silent placeholder if none) |

**Known limitations:**
- Requires ComfyUI's legacy renderer. Nodes 2.0 / Vue renderer is not supported; the node will display a warning if it detects an unsupported renderer.
- ComfyUI does not garbage-collect orphaned node metadata from workflow JSON. If widget values seem stale after deleting and re-adding the node, manually clean the JSON or start from a fresh workflow.

---

### VACE Inline

Single-video VACE prep node. Instead of joining two separate clips, this node takes one continuous video plus a frame selection (typically from Visual Frame Selector) and regenerates the selected range in place using VACE.

Typical use case: you have a long clip and want to smooth or regenerate a specific section without splitting it manually.

**Parameters:**

| Parameter | Default | Description |
|-|-|-|
| images | | Full source video as an IMAGE batch (e.g., from Load Video or Visual Frame Selector) |
| start_frame | 0 | First frame of the selection (0-based) |
| end_frame | 0 | Last frame of the selection, exclusive (0-based) |
| context_frames | 8 | Reference frames from each side of the selection for VACE conditioning. Must be a multiple of 4. |
| new_frames | 0 | Additional frames to insert within the selected range, expanding the video length. Must be a multiple of 4. |

**Outputs:**

| Output | Description |
|-|-|
| control_video | VACE control video input |
| control_mask | VACE control mask input |
| width, height, length | Control video dimensions |
| start_images | Video segment before the context + selection region |
| end_images | Video segment after the selection + context region |
| context_frames, new_frames | Parameter passthrough for downstream wiring |

**Known limitations:**
- If the selected frame range is not 4n+1, the node snaps up to the nearest 4n+1 and logs a warning. The output will be 1–3 frames longer than the input selection. A future update may enforce 4n+1 selection in Visual Frame Selector to avoid this.

---

## Technical Notes

**4n+1 frame rule.** The Wan model generates 4n+1 frames at a time. If you request a different count, it silently rounds down to the nearest 4n+1. For this reason, parameters are restricted to multiples of 4 or 4n+1, and when necessary the nodes add +1 to the generated frame count.

## License

MIT License — feel free to use, modify, and distribute.
