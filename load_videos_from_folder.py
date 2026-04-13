import os
import re
import av
import numpy as np
import torch


class LoadVideosFromFolderSimple:
    VIDEO_EXTENSIONS = ['webm', 'mp4', 'mkv', 'gif', 'mov']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log some details to the console"
                }),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load_videos"
    CATEGORY = "video/utility"
    DESCRIPTION = """
    Load all videos from a folder, concatenated into
    a single image batch.  Optionally connect a
    VideoHelperSuite Meta Batch Manager to process
    large collections in smaller RAM-safe chunks. See
    VHS Meta Batch Manager documentation for more information.
    """

    def load_videos(self, folder_path, debug, meta_batch=None, unique_id=None):
        folder_path = folder_path.strip().strip('"').strip("'")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")

        video_files = self._get_video_files(folder_path)

        if not video_files:
            raise ValueError(
                f"No video files found in {folder_path}\n"
                f"Supported formats: {', '.join(self.VIDEO_EXTENSIONS)}"
            )

        if meta_batch is None:
            return self._load_all(video_files, folder_path, debug)
        else:
            return self._load_batched(video_files, folder_path, debug, meta_batch, unique_id)

    def _load_all(self, video_files, folder_path, debug):
        """Load all videos at once into a single tensor."""
        if debug:
            print(f"[Load Videos] Loading {len(video_files)} videos from {folder_path}")

        all_frames = []
        expected_shape = None

        for idx, video_path in enumerate(video_files):
            if debug:
                print(f"[Load Videos] [{idx+1}/{len(video_files)}]: {os.path.basename(video_path)}", end=" ... ")

            frames = self._load_video_frames(video_path)
            expected_shape = self._check_resolution(frames, expected_shape, video_path)
            all_frames.append(frames)

            if debug:
                print(f"{frames.shape[0]} frames")

        if debug:
            print(f"[Load Videos] Concatenating {len(video_files)} videos...")
        output = torch.cat(all_frames, dim=0)

        if debug:
            print(f"[Load Videos] Done")
        return (output,)

    def _load_batched(self, video_files, folder_path, debug, meta_batch, unique_id):
        """
        Load frames in chunks coordinated by the VHS BatchManager.
        The BatchManager drives re-execution of the workflow. Each pass
        picks up where the generator left off, yields up to frames_per_batch
        frames, then returns. The BatchManager re-queues until exhausted.
        """
        if unique_id not in meta_batch.inputs:
            total_frames = self._count_total_frames(video_files)
            meta_batch.total_frames = min(meta_batch.total_frames, total_frames)
            if debug:
                print(f"[Load Videos] Batched: Starting new generator for {len(video_files)} videos ({total_frames} frames) in {folder_path}")
            meta_batch.inputs[unique_id] = self._frame_generator(video_files, debug)

        generator = meta_batch.inputs[unique_id]
        frames_per_batch = meta_batch.frames_per_batch

        batch_frames = []
        expected_shape = None
        frames_collected = 0

        while frames_collected < frames_per_batch:
            try:
                frame_tensor, video_path = next(generator)
            except StopIteration:
                if debug:
                    print(f"[Load Videos] Batched: Generator exhausted, cleaning up")
                meta_batch.inputs.pop(unique_id)
                meta_batch.has_closed_inputs = True
                break

            expected_shape = self._check_resolution(
                frame_tensor.unsqueeze(0), expected_shape, video_path
            )
            batch_frames.append(frame_tensor)
            frames_collected += 1

        if not batch_frames:
            raise RuntimeError("Batched loader produced no frames")

        output = torch.stack(batch_frames, dim=0)

        if debug:
            print(f"[Load Videos] Batched: Yielding {output.shape[0]} frames  shape={tuple(output.shape)}")

        return (output,)

    def _get_video_files(self, folder_path):
        """Return naturally-sorted list of video file paths in folder_path."""
        video_files = []
        for f in os.listdir(folder_path):
            full_path = os.path.join(folder_path, f)
            if os.path.isfile(full_path) and self._is_video_file(f):
                video_files.append(full_path)

        def natural_sort_key(path):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', os.path.basename(path))]

        return sorted(video_files, key=natural_sort_key)

    def _is_video_file(self, filename):
        """Return True if filename has a supported video extension."""
        _, ext = os.path.splitext(filename)
        return ext.lstrip('.').lower() in self.VIDEO_EXTENSIONS

    def _load_video_frames(self, video_path):
        """Load all frames from a video file. Returns [N, H, W, C] float32 tensor."""
        container = av.open(video_path)
        try:
            if len(container.streams.video) == 0:
                raise ValueError(f"No video stream found in: {video_path}")

            frames = []
            for frame in container.decode(video=0):
                rgb = frame.to_ndarray(format="rgb24")
                frame_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
                frames.append(frame_tensor)

            if not frames:
                raise RuntimeError(f"No frames extracted from {video_path}")

            return torch.stack(frames, dim=0)
        finally:
            container.close()

    def _check_resolution(self, frames, expected_shape, video_path):
        """
        Verify frames match expected_shape (H, W) from tensor shape[1:3].
        Returns expected_shape, initialising it from frames if not yet set.
        Raises ValueError with filename on mismatch.
        """
        if expected_shape is None:
            return frames.shape[1:3]
        if frames.shape[1:3] != expected_shape:
            raise ValueError(
                f"\nResolution mismatch\n"
                f"  Expected: {expected_shape[1]}x{expected_shape[0]} (from first video)\n"
                f"  Got: {frames.shape[2]}x{frames.shape[1]} in {os.path.basename(video_path)}"
            )
        return expected_shape

    def _count_total_frames(self, video_files):
        """Fast frame count estimate using container metadata.

        Used by VHS BatchManager for progress display. Falls back to
        full decode if the container doesn't report a frame count.
        """
        total = 0
        for video_path in video_files:
            container = None
            try:
                container = av.open(video_path)
                if len(container.streams.video) > 0:
                    count = container.streams.video[0].frames
                    if count > 0:
                        total += count
                    else:
                        # Container doesn't report frame count; decode to count
                        total += sum(1 for _ in container.decode(video=0))
            except Exception as e:
                print(f"[Load Videos] Warning: could not count frames in {os.path.basename(video_path)}: {e}")
            finally:
                if container is not None:
                    container.close()
        return total

    def _frame_generator(self, video_files, debug):
        """
        Generator that yields (frame_tensor, video_path) one frame at a time
        across all videos. Keeps at most one video open at a time.
        """
        for idx, video_path in enumerate(video_files):
            if debug:
                print(f"[Load Videos] Batched: Opening [{idx+1}/{len(video_files)}]: {os.path.basename(video_path)}")

            container = av.open(video_path)
            try:
                if len(container.streams.video) == 0:
                    raise ValueError(f"No video stream found in: {video_path}")

                frame_count = 0
                for frame in container.decode(video=0):
                    rgb = frame.to_ndarray(format="rgb24")
                    frame_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
                    frame_count += 1
                    yield frame_tensor, video_path
            finally:
                container.close()

            if debug:
                print(f"[Load Videos] Batched: Finished {os.path.basename(video_path)}: {frame_count} frames")
