import torch
import numpy as np
import os
import av
import folder_paths

# Video extensions supported for the file selector
VIDEO_EXTENSIONS = ['mp4', 'webm', 'mkv', 'mov', 'avi', 'gif']


class VisualFrameSelector:
    """
    Interactive video frame selection with visual preview.
    
    Load a video and use the visual scrubber to select a range of frames.
    The selected frames are output as an IMAGE tensor.
    
    Frame range is inclusive at both ends, 0-based indexing.
    start_frame=0, end_frame=4 outputs 5 frames (0, 1, 2, 3, 4).
    
    The UI enforces end_frame >= start_frame + 1 (minimum 2-frame selection).
    end_frame=0 means "last frame of the video".
    """

    DESCRIPTION = """# Visual Frame Selector

Interactive video frame selection with visual preview.

**Controls:**
- Drag markers to set start/end frames
- Click scrubber to seek
- Use transport buttons for playback
- Right-click video for volume control
"""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT", "INT", "AUDIO")
    RETURN_NAMES = ("images", "selected_frames", "fps", "total_frames", "start_frame", "end_frame", "audio")
    FUNCTION = "load_frames"
    CATEGORY = "image/video"

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

    def load_frames(self, video, current_frame=0, start_frame=0, end_frame=0):
        video_path = folder_paths.get_annotated_filepath(video)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        container = None
        try:
            container = av.open(video_path)

            if len(container.streams.video) == 0:
                raise ValueError("No video stream found in file")

            video_stream = container.streams.video[0]
            codec_name = video_stream.codec_context.name if video_stream.codec_context else "unknown"

            fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
            total_frames = video_stream.frames
            if total_frames == 0:
                total_frames = sum(1 for _ in container.decode(video=0))
                container.seek(0)

            # Resolve frame range (inclusive, 0-based)
            actual_start = max(0, min(start_frame, total_frames - 1))
            actual_end = end_frame if end_frame > 0 else total_frames - 1
            actual_end = min(actual_end, total_frames - 1)

            if actual_end <= actual_start:
                actual_end = min(actual_start + 1, total_frames - 1)

            # Seek to start frame
            if actual_start > 0:
                target_pts = int(actual_start / fps / video_stream.time_base)
                container.seek(target_pts, stream=video_stream)

            # Decode frames
            frames = []
            frame_index = 0
            for frame in container.decode(video=0):
                if frame_index < actual_start:
                    frame_index += 1
                    continue
                if frame_index > actual_end:
                    break

                try:
                    rgb = frame.to_ndarray(format="rgb24")
                except Exception:
                    raise ValueError(
                        f"Cannot decode frame {frame_index}: codec '{codec_name}' "
                        f"is not supported. Re-encode the video with a compatible "
                        f"codec (e.g. h264, h265, vp9)."
                    )
                frames.append(torch.from_numpy(rgb.astype(np.float32) / 255.0))
                frame_index += 1

            container.close()
            container = None

            if len(frames) == 0:
                raise ValueError("No frames were loaded from the video")

            frames_tensor = torch.stack(frames)

            audio = self._extract_audio(video_path, actual_start, actual_end, fps)

            return (
                frames_tensor,
                len(frames),
                fps,
                total_frames,
                actual_start,
                actual_end,
                audio,
            )

        except (ValueError, FileNotFoundError):
            raise  # Let these propagate cleanly to ComfyUI

        except Exception as e:
            raise ValueError(f"Error processing video: {str(e)}")

        finally:
            if container is not None:
                container.close()

    def _extract_audio(self, video_path, start_frame, end_frame, fps):
        """Extract audio for the selected frame range using PyAV.
        
        Returns audio dict compatible with ComfyUI AUDIO type, or None if
        no audio track exists.
        """
        if fps <= 0:
            return None

        try:
            container = av.open(video_path)

            if len(container.streams.audio) == 0:
                container.close()
                return None

            audio_stream = container.streams.audio[0]
            sample_rate = audio_stream.rate
            channels = audio_stream.channels

            start_time = start_frame / fps
            duration = (end_frame - start_frame + 1) / fps
            end_time = start_time + duration

            # Seek to start time
            if start_time > 0:
                target_pts = int(start_time / audio_stream.time_base)
                container.seek(target_pts, stream=audio_stream)

            # Decode audio samples
            audio_frames = []
            for frame in container.decode(audio=0):
                frame_time = float(frame.pts * audio_stream.time_base) if frame.pts is not None else 0.0
                if frame_time > end_time:
                    break
                # Convert to float32 numpy array
                arr = frame.to_ndarray()
                audio_frames.append(arr)

            container.close()

            if len(audio_frames) == 0:
                return None

            # Concatenate all decoded audio
            audio_np = np.concatenate(audio_frames, axis=-1)

            # Ensure shape is [channels, samples]
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)

            # Trim to exact time range
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            total_samples = audio_np.shape[-1]

            # After seeking, the first decoded frame may start before our target
            # Just use what we have, trim to duration
            max_samples = end_sample - start_sample
            if audio_np.shape[-1] > max_samples:
                audio_np = audio_np[:, :max_samples]

            # ComfyUI AUDIO format: {"waveform": tensor [batch, channels, samples], "sample_rate": int}
            audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dim

            return {"waveform": audio_tensor, "sample_rate": sample_rate}

        except Exception:
            return None

NODE_CLASS_MAPPINGS = {
    "VisualFrameSelector": VisualFrameSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualFrameSelector": "🪐 Visual Frame Selector",
}