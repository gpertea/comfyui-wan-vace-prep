import os
import torch
import numpy as np
import av
import folder_paths


class VisualFrameSelector:
    DESCRIPTION = """
    Interactive video frame selection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "video":         (sorted(files), {"video_upload": True}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "start_frame":   ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "end_frame":     ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "INT",  "INT",  "INT",  "INT",  "FLOAT", "AUDIO")
    RETURN_NAMES  = ("selected_frames", "all_frames", "selected_count", "frame_count",
                     "start_frame", "end_frame", "fps", "audio")
    FUNCTION = "load_frames"
    CATEGORY = "video/utility"
    EXPERIMENTAL = True

    @classmethod
    def IS_CHANGED(cls, video, start_frame, end_frame, **kwargs):
        video_path = folder_paths.get_annotated_filepath(video)
        mtime = os.path.getmtime(video_path) if os.path.exists(video_path) else 0
        return mtime

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_video_frames(container, video_stream, start=0, end=None):
        """Decode frames [start, end] inclusive. Returns list of RGB uint8 ndarrays."""
        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        codec_name = video_stream.codec_context.name if video_stream.codec_context else "unknown"

        if start > 0:
            target_pts = int(start / fps / video_stream.time_base)
            container.seek(target_pts, stream=video_stream)

        frames = []
        fallback_index = None  # used only when frame.pts is unavailable
        for frame in container.decode(video=0):
            # Derive the absolute frame index from PTS so that seeks landing
            # on a keyframe before `start` don't corrupt the count.
            if frame.pts is not None:
                frame_index = round(float(frame.pts) * float(video_stream.time_base) * fps)
                fallback_index = frame_index + 1
            else:
                if fallback_index is None:
                    fallback_index = start
                frame_index = fallback_index
                fallback_index += 1

            if frame_index < start:
                continue
            if end is not None and frame_index > end:
                break
            try:
                rgb = frame.to_ndarray(format="rgb24")
            except Exception:
                raise ValueError(
                    f"Cannot decode frame {frame_index}: codec '{codec_name}' "
                    f"is not supported. Re-encode with h264, h265, or vp9."
                )
            frames.append(rgb)

        return frames

    @staticmethod
    def _extract_audio(video_path):
        """Return a ComfyUI-compatible audio dict, or None if no audio track."""
        try:
            container = av.open(video_path)
            if not container.streams.audio:
                container.close()
                return None

            audio_stream = container.streams.audio[0]
            sample_rate  = audio_stream.sample_rate

            pcm_frames = []
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()          # (channels, samples) or (samples,)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                pcm_frames.append(arr.astype(np.float32))

            container.close()

            if not pcm_frames:
                return None

            waveform = np.concatenate(pcm_frames, axis=-1)   # (C, N)
            max_val  = np.abs(waveform).max()
            if max_val > 1.0:
                waveform = waveform / max_val

            # ComfyUI AUDIO format: {"waveform": Tensor[B,C,N], "sample_rate": int}
            waveform_t = torch.from_numpy(waveform).unsqueeze(0)   # (1, C, N)
            return {"waveform": waveform_t, "sample_rate": sample_rate}

        except Exception as e:
            print(f"[VisualFrameSelector] Audio extraction skipped: {e}")
            return None

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def load_frames(self, video, current_frame=0, start_frame=0, end_frame=0):
        video_path = folder_paths.get_annotated_filepath(video)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        container = None
        try:
            container = av.open(video_path)

            if not container.streams.video:
                raise ValueError("No video stream found in file")

            video_stream = container.streams.video[0]
            fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0

            # --- Total frame count -------------------------------------------
            total_frames = video_stream.frames
            if total_frames == 0:
                total_frames = sum(1 for _ in container.decode(video=0))
                container.seek(0)

            # --- Log Inputs --------------------------------------------------
            print(f"[VisualFrameSelector] Inputs: video='{os.path.basename(video)}', current_frame={current_frame}, start_frame={start_frame}, end_frame={end_frame}")

            # --- Resolve selected range (inclusive, 0-based) -----------------
            original_start = start_frame
            original_end = end_frame
            
            actual_start = max(0, min(start_frame, total_frames - 1))
            actual_end   = end_frame if end_frame > 0 else total_frames - 1
            actual_end   = min(actual_end, total_frames - 1)
            
            if actual_end <= actual_start:
                actual_end = min(actual_start + 1, total_frames - 1)

            # --- Determine Adjustment Reasons --------------------------------
            start_reason = None
            if original_start < 0:
                start_reason = "clamped to 0"
            elif original_start >= total_frames:
                start_reason = "clamped to last frame"
            
            end_reason = None
            if original_end <= original_start:
                end_reason = "adjusted to ensure range > 0"
            elif original_end >= total_frames:
                end_reason = "clamped to last frame"
            elif original_end <= 0 and actual_end != original_end:
                 end_reason = "defaulted to last frame"

            # --- Decode selected frames --------------------------------------
            selected_rgb = self._decode_video_frames(
                container, video_stream, actual_start, actual_end
            )

            if not selected_rgb:
                raise ValueError("No frames were loaded from the video")

            selected_tensor = torch.stack([
                torch.from_numpy(f.astype(np.float32) / 255.0) for f in selected_rgb
            ])

            # --- Decode ALL frames -------------------------------------------
            container.seek(0)
            all_rgb = self._decode_video_frames(container, video_stream, 0, None)

            all_tensor = torch.stack([
                torch.from_numpy(f.astype(np.float32) / 255.0) for f in all_rgb
            ])

            container.close()
            container = None

            # --- Audio -------------------------------------------------------
            audio = self._extract_audio(video_path)
            audio_status = "extracted" if audio and audio.get("waveform").shape[2] > 1 else "none/placeholder"

            # --- Log Outputs -------------------------------------------------
            start_log = f"start_frame: {actual_start}"
            if start_reason:
                start_log = f"start_frame: {original_start} -> {actual_start} (Reason: {start_reason})"
            
            end_log = f"end_frame: {actual_end}"
            if end_reason:
                end_log = f"end_frame: {original_end} -> {actual_end} (Reason: {end_reason})"

            print(f"[VisualFrameSelector] Outputs: {start_log}, {end_log}, selected_count={len(selected_rgb)}, total_frames={total_frames}, fps={fps:.2f}, audio={audio_status}")

            if audio is None:
                # Silent placeholder so the AUDIO output is always valid
                audio = {"waveform": torch.zeros(1, 2, 1), "sample_rate": 44100}

            return (
                selected_tensor,    # images        – selected range
                all_tensor,         # all_frames    – full video (or capped)
                len(selected_rgb),  # selected_count
                total_frames,       # total_frames
                actual_start,       # start_frame
                actual_end,         # end_frame
                fps,                # fps
                audio,              # audio
            )

        except (ValueError, FileNotFoundError):
            raise
        except Exception as e:
            raise ValueError(f"Error processing video: {str(e)}")
        finally:
            if container is not None:
                container.close()


NODE_CLASS_MAPPINGS = {
    "VisualFrameSelector": VisualFrameSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualFrameSelector": "🪐 Visual Frame Selector (Experimental)"
}
