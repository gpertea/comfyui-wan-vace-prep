import os
import torch
import numpy as np
import av
import folder_paths


class VisualFrameSelector:
    DESCRIPTION = """
    Interactive video frame selection.
    - Drag markers to set start/end frames
    - Click scrubber to seek
    - Use transport buttons for playback
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

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "selected_frames", "total_frames", "start_frame", "end_frame")
    FUNCTION = "load_frames"
    CATEGORY = "video/utility"

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

            return (
                frames_tensor,
                len(frames),
                total_frames,
                actual_start,
                actual_end,
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
    "VisualFrameSelector": "🪐 Visual Frame Selector"
}
