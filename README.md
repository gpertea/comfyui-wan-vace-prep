# Wan VACE Prep

ComfyUI nodes for Wan VACE workflows. Control video and mask creation for transitions and extensions, plus workflow utilities.

## Quick Start

**Install via ComfyUI Manager:** Search for "Wan VACE Prep"

**Or clone this repository:**

```bash
cd /path/to/comfyui/custom_nodes
git clone https://github.com/stuttlepress/ComfyUI-Wan-VACE-Prep
```

## Nodes

### VACE Join	
For smoothly joining two video clips together. Builds VACE controls for the transition using context frames from each clip to guide frame generation. 

![VACE Join Node](assets/vace-join.png)

**Parameters:**

| Parameter | Default | Description |
|-|-|-|
| context_frames | 8 | Reference frames from each video edge that VACE uses for interpolation. These frames guide the model and are preserved in the output. Must be a multiple of 4. |
| replace_frames | 8 | Number of frames at each transition edge to discard and regenerate. These create the actual transition blend zone. Must be a multiple of 4. |
| new_frames | 0 | Number of completely new frames to generate between the two clips, extending the transition duration. Must be 0 or a multiple of 4. |

**Outputs:**

| Output | Description |
|-|-|
| control_video | VACE control video input |
| control_mask | VACE control mask input |
| width, height, length | Control video dimensions |
| start_images | Video 1 segment that precedes context frames and the transition |
| end_images | Video 2 segment that comes after the transition and context frames |
| context_frames, replace_frames, new_frames | Parameter passthrough for optional downstream wiring |



https://github.com/user-attachments/assets/5b70b5af-8d88-4c0f-a5a7-cdbe0657ce8c



---

### VACE Join (Batch)

Batch-aware version of VACE Join for processing multiple video pairs. Handles first/last iteration edge cases.

![VACE Join Batch Node](assets/vace-join-batch.png)

**Parameters:**

| Parameter | Default | Description |
|-|-|-|
| video_1 | | First video in the pair (IMAGE type) |
| video_2 | | Second video in the pair (IMAGE type) |
| is_first | false | Set true for first iteration (index=0) — includes full beginning of video_1 in start_images |
| is_last | false | Set true for last iteration — includes full ending of video_2 in end_images |
| context_frames | 8 | Reference frames from each video edge for VACE interpolation. Must be a multiple of 4. |
| replace_frames | 8 | Frames at each transition edge to discard and regenerate. Must be a multiple of 4. |
| new_frames | 0 | New frames to generate between clips. Must be 0 or a multiple of 4. |
| debug | false | Log diagnostic information to the console |

**Outputs:**

| Output | Description |
|-|-|
| control_video | VACE control video input (context frames + placeholder for generation) |
| control_mask | VACE control mask input (masks generation region) |
| width, height, length | Control video dimensions |
| start_images | Video segment from video_1 to preserve (excludes transition region) |
| end_images | Video segment from video_2 to preserve (only populated on last iteration) |
| context_frames, replace_frames, new_frames | Parameter passthrough for optional downstream wiring |

---

### VACE Batch Context

Establishes iteration context for batch video processing workflows. Manages file paths, iteration tracking, and provides first/last flags for proper handling of video sequence boundaries.

![VACE Batch Context Node](assets/vace-batch-context.png)

**Parameters:**

| Parameter | Default | Description |
|-|-|-|
| input_list | | List of video filenames to process (STRING, force input) |
| input_dir | | Directory containing input videos |
| project_name | . | Project name — workflow files created under ComfyUI/output/project_name. Use period (.) for no project name. |
| index | 0 | Current iteration index (0-based). Valid range: 0 to (number of videos - 2) |
| debug | false | Log iteration details to the console |

**Outputs:**

| Output | Description |
|-|-|
| work_dir | Working directory path for intermediate files |
| workfile_prefix | Filename prefix for this iteration's work files |
| video_1_filename | Full path to first video in current pair |
| video_2_filename | Full path to second video in current pair |
| is_first | True if this is the first iteration (index=0) |
| is_last | True if this is the last iteration |

---

### VACE Extend

Extends a video from an arbitrary frame position. Context frames preceding the extension point build a VACE control video for conditioning.

![VACE Extend Node](assets/vace-extend.png)

**Parameters:**

| Parameter | Default | Description |
|-|-|-|
| extend_from_idx | -1 | Frame to extend from (negative counts from end, e.g., -1 = last frame) |
| context_frames | 8 | Reference frames preceding extend_from_idx for VACE conditioning. Must be a multiple of 4. |
| new_frames | 25 | Number of new frames to generate (must be 4n+1: 1, 5, 9, 13, 17, 21, 25...) |

**Outputs:**

| Output | Description |
|-|-|
| control_video | VACE control video input |
| control_mask | VACE control mask input |
| width, height, length | Control video dimensions |
| start_images | Video segment that precedes the context frames and the extension |
| context_frames, new_frames | Parameter passthrough for downstream wiring |

---

### Load Videos From Folder (Simple)

Loads all videos from a folder, concatenated into a single image batch.

Optionally connect a **VideoHelperSuite** *Meta Batch Manager* node to process large collections in RAM-safe chunks. If you are joining a large number of video files and running out of system memory as they concatenate, this is the solution. From the VHS Meta Batch Manager node documentation:

> The Meta Batch Manager allows for extremely long input videos to be processed when all other methods for fitting the content in RAM fail. It does not affect VRAM usage. It must be connected to at least one Input (a Load Video or Load Images) AND at least one Video Combine.

See the VHS Meta Batch Manager node documentation for more information.

*Meta Batch Manager* rule of thumb: set `frames_per_batch` to roughly 10× your available RAM (not VRAM) in GB — so 32 GB → 320 frames, 64 GB → 640 frames, 128 GB → 1280 frames.

- **Formats:** webm, mp4, mkv, gif, mov
- All videos must have identical resolution
- No external dependencies

![Load Videos From Folder (Simple) Node](assets/load-videos-from-folder-simple.png)

**Parameters:**

| Parameter | Default | Description |
|-|-|-|
| folder_path | | Full pathname of the directory holding input videos |
| debug | false | Log video details and progress to the console |
| meta_batch | *(optional)* | Connect to VideoHelperSuite Meta Batch Manager to load videos in batches |

**Outputs:**

| Output | Description |
|-|-|
| images | Concatenated image batch ready for video creation |

---

## Experimental Nodes

The following nodes are functional but may have rough edges, and their interfaces may change between releases.

---

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

**Class names vs. display names.** Some internal class names (e.g., `WanVACEPrep`) don't match the current display names (e.g., "VACE Join"). This is intentional — renaming classes would break existing workflows that reference them. Once ComfyUI's node renaming API is stable, a refactoring pass will align them.

## License

MIT License — feel free to use, modify, and distribute.
