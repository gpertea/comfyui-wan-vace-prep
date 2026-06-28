# V3 Migration Test Checklist

## 1. Extension loads

- [x] ComfyUI starts without errors in the console (no import errors, no `AttributeError` on `comfy_entrypoint`)
- [x] No `NODE_CLASS_MAPPINGS` warning appears (would mean V1/V3 conflict)
- [x] All 9 nodes appear in the "Add Node" menu

## 2. Category / menu placement

| Node | Expected menu path |
|------|--------------------|
| VACE Join | Wan VACE Prep > VACE |
| VACE Join (Batch) | Wan VACE Prep > VACE |
| VACE Batch Context | Wan VACE Prep > VACE |
| VACE Extend | Wan VACE Prep > VACE |
| VACE Inpaint | Wan VACE Prep > VACE |
| VACE First/Middle/Last | Wan VACE Prep > VACE |
| Load Videos From Folder (Simple) | Wan VACE Prep > utility |
| Frame Number Overlay | Wan VACE Prep > utility |
| Wan First/Middle/Last Frame to Video | Wan VACE Prep > conditioning |

- [x] Video Outpaint does **not** appear (intentionally unregistered)

## 3. Display names and experimental flag

- [x] These 4 nodes show an experimental badge/indicator in the UI: VACE Inpaint, Frame Number Overlay, Wan First/Middle/Last Frame to Video, VACE First/Middle/Last
- [x] All display names show correctly (spot-check: "VACE Join", "VACE Join (Batch)", "Load Videos From Folder (Simple)")

## 4. Input widgets render correctly

- [x] **VACE Join**: `context_frames`, `replace_frames`, `new_frames` show as integer sliders; `video_1` and `video_2` show as IMAGE sockets with no widget
- [x] **VACE Join (Batch)**: `is_first`, `is_last`, `debug` show as checkboxes
- [x] **VACE Batch Context**: `input_list` shows as a socket only with no text box (`force_input=True`); `make_loop` and `debug` are checkboxes
- [x] **Frame Number Overlay**: `position` shows as a dropdown with four options (top-left, top-right, bottom-left, bottom-right)
- [x] **Load Videos From Folder**: `meta_batch` socket is optional (node executes without it connected)
- [x] **Wan First/Middle/Last Frame to Video**: optional image inputs (`start_image`, `middle_image`, `end_image`) show as sockets only with no widget; `middle_frame` shows as a float slider
- [x] **VACE First/Middle/Last**: `first`, `middle`, `last` optional image inputs present

## 5. Output sockets

- [x] **VACE Join** outputs 10 sockets in order: control_video (IMAGE), control_mask (MASK), width (INT), height (INT), length (INT), start_images (IMAGE), end_images (IMAGE), context_frames (INT), replace_frames (INT), new_frames (INT)
- [x] **VACE Extend** outputs 8 sockets in order: control_video (IMAGE), control_mask (MASK), width (INT), height (INT), length (INT), start_images (IMAGE), context_frames (INT), new_frames (INT)
- [x] **Load Videos From Folder**: outputs both `images` (IMAGE) and `audio` (AUDIO) sockets
- [x] **Wan First/Middle/Last Frame to Video** outputs: positive (CONDITIONING), negative (CONDITIONING), latent (LATENT)

## 6. Execution - smoke tests (minimal inputs)

- [x] **VACE Join**: run with two short matching-resolution image batches (e.g., 16 frames each). Verify control_video and mask shapes are correct, no runtime errors.
- [x] **VACE Extend**: run with a short video, `extend_from_idx=-1`, `new_frames=9`. Verify shapes.
- [x] **VACE Inpaint**: run with a video and a single-frame mask. Verify mask broadcasts to all frames.
- [x] **Frame Number Overlay**: run with a small image batch. Confirm frame numbers are visible in output.
- [x] **Load Videos From Folder**: point at a folder with 2+ mp4 files. Verify frames concatenate and audio socket has data.
- [x] **VACE First/Middle/Last**: run with only `first` connected, nothing else. Verify gray placeholder output for remaining frames.
- [x] **Wan First/Middle/Last Frame to Video**: run with only `positive`, `negative`, `vae` connected, no optional images. Verify conditioning and latent output.

## 7. is_input_list behavior (VACE Batch Context)

- [x] Connect a filename list to `input_list`. Verify index 0 and 1 return distinct filenames.
- [x] Set `index=0`. Verify `is_first=True`, `is_last=False`.
- [x] Set `index` to the maximum valid value. Verify `is_last=True`, `assemble_video=True`.
- [x] Set `index` out of range. Verify a clear error is raised rather than silent misbehavior.

## 8. Hidden input (Load Videos From Folder with meta_batch)

- [x] Connect a VHS Meta Batch Manager. Verify the node uses `cls.hidden.unique_id` correctly: multiple batch passes yield sequential frame chunks without resetting the generator.

## 9. Error handling

- [x] **VACE Join**: mismatched video resolutions -> clear error message showing both dimensions
- [x] **VACE Join**: `context_frames + replace_frames` exceeds video length -> clear error
- [x] **VACE Extend**: `new_frames=10` (not 4n+1) -> error naming the two nearest valid values
- [x] **VACE Extend**: `context_frames` exceeds frames before extend point -> clear error
- [x] **Load Videos From Folder**: non-existent folder path -> "Folder does not exist" error
- [x] **Load Videos From Folder**: folder with mixed-resolution videos -> resolution mismatch error naming the offending file
