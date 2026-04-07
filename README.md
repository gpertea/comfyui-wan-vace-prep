# Wan VACE Prep

This is a small collection of ComfyUI nodes designed to help make common Wan VACE tasks less complicated.

## Quick Start

**Install via ComfyUI Manager:** Search for "Wan VACE Prep"

**Or clone this repository:**

```bash
cd /path/to/comfyui/custom_nodes
git clone https://github.com/stuttlepress/ComfyUI-Wan-VACE-Prep
```

## Nodes

- [VACE Outpaint](#vace-outpaint)
- [VACE Join](#vace-join)
- [VACE Join (Batch)](#vace-join-batch)
- [VACE Batch Context](#vace-batch-context)
- [VACE Extend](#vace-extend)
- [Load Videos From Folder (Simple)](#load-videos-from-folder-simple)

### VACE Outpaint

Prepares a video for outpainting using an interactive canvas widget. Position and size an output window over your source frames. Regions outside the source become the outpaint area. Builds a VACE control video and mask ready for generation.

![VACE Outpaint Node](assets/vace-outpaint.png)

> **Nodes 2.0 incompatibility:** This node is known to malfunction under ComfyUI's Nodes 2.0 renderer. No fix is planned until ComfyUI publishes node development documentation.

**Parameters:**

| Parameter | Default | Description |
|-|-|-|
| images | | Source video frames |

**Outputs:**

| Output | Description |
|-|-|
| control_video | VACE control video input. Source content placed within the output window; overhanging regions filled with gray (0.5). |
| control_mask | VACE control mask input. White (1) where outpainting should occur, black (0) over source content. |
| width, height | Output video dimensions |
| length | Frame count |

https://github.com/user-attachments/assets/a233832a-6630-4e7b-8d05-9e048d2e97a4

---

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
| is_first | false | Set true for first iteration (index=0). Includes full beginning of video_1 in start_images |
| is_last | false | Set true for last iteration. Includes full ending of video_2 in end_images |
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

*This node drives my [Wan VACE Video Joiner](https://github.com/stuttlepress/ComfyUI-Wan-VACE-Video-Joiner) workflow. It may not be useful outside of that context.*  
Establishes iteration context for batch video processing workflows. Manages file paths, iteration tracking, and provides first/last flags for proper handling of video sequence boundaries. Supports an optional loop mode that generates a wrap-around transition between the last and first video for seamless looping output.

![VACE Batch Context Node](assets/vace-batch-context.png)

**Parameters:**

| Parameter | Default | Description |
|-|-|-|
| input_list | | List of video filenames to process (STRING, force input) |
| input_dir | | Directory containing input videos |
| project_name | . | Workflow files are created under ComfyUI/output/project_name. Use period (.) for no project name. |
| index | 0 | Current iteration index (0-based). Valid range: 0 to (number of videos - 2) normally, or 0 to (number of videos - 1) when `make_loop=true` |
| debug | false | Log iteration details to the console |
| make_loop | false | Enable loop mode. Adds one extra iteration that pairs the last video with the first, creating a seamless loop. When true, `is_first` and `is_last` are always false. |

**Outputs:**

| Output | Description |
|-|-|
| work_dir | Working directory path for intermediate files |
| workfile_prefix | Filename prefix for this iteration's work files |
| video_1_filename | Full path to first video in current pair |
| video_2_filename | Full path to second video in current pair |
| is_first | True if this is the first iteration (index=0). Always false when `make_loop=true`. |
| is_last | True if this is the last iteration. Always false when `make_loop=true`. |
| assemble_video | True on the final iteration. Used to gate the assembly step. Equivalent to `is_last` when `make_loop=false`; fires on the loop-closing iteration when `make_loop=true`. |

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

Optionally connect a **[VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** *Meta Batch Manager* node to process large collections in RAM-safe chunks. If you are joining a large number of video files and running out of system memory as they concatenate, this is the solution. From the VHS Meta Batch Manager node documentation:

> The Meta Batch Manager allows for extremely long input videos to be processed when all other methods for fitting the content in RAM fail. It does not affect VRAM usage. It must be connected to at least one Input (a Load Video or Load Images) AND at least one Video Combine.

See the VHS Meta Batch Manager node documentation for more information.

*Meta Batch Manager* rule of thumb: set `frames_per_batch` to roughly 10× your available RAM (not VRAM) in GB. So 32 GB -> 320 frames, 64 GB -> 640 frames, 128 GB -> 1280 frames.

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

## Technical Notes

**4n+1 frame rule.** The Wan model generates 4n+1 frames at a time. If you request a different count, it silently rounds down to the nearest 4n+1. For this reason, parameters are restricted to multiples of 4 or 4n+1, and when necessary the nodes add +1 to the generated frame count.

**Class names vs. display names.** Some internal class names (e.g., `WanVACEPrep`) don't match the current display names (e.g., "VACE Join"). This is intentional: renaming classes would break existing workflows that reference them. Once ComfyUI's node renaming API is stable, a refactoring pass will align them.

**Nodes 2.0 renderer.** These nodes have not been tested under ComfyUI's Nodes 2.0 renderer and may or may not work correctly with it. Testing and support will be added sometime after ComfyUI publishes documentation for node developers to work from.

## License

MIT License. Feel free to use, modify, and distribute.
