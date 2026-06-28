import os
import re
import av
import numpy as np
import torch
from comfy_api.latest import io


class LoadVideosFromFolderSimple(io.ComfyNode):
    VIDEO_EXTENSIONS = ['webm', 'mp4', 'mkv', 'gif', 'mov']

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadVideosFromFolderSimple",
            display_name="🪐 Load Videos From Folder (Simple)",
            category="Wan VACE Prep/utility",
            description=(
                "Load all videos from a folder, concatenated into\n"
                "a single image batch with their audio tracks combined.\n"
                "Optionally connect a VideoHelperSuite Meta Batch Manager\n"
                "to process large collections in smaller RAM-safe chunks. See\n"
                "VHS Meta Batch Manager documentation for more information."
            ),
            inputs=[
                io.String.Input("folder_path", default=""),
                io.Boolean.Input("debug", default=False,
                    tooltip="Log some details to the console"),
                io.Custom("VHS_BatchManager").Input("meta_batch", optional=True),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Audio.Output("audio"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, folder_path, debug, meta_batch=None) -> io.NodeOutput:
        folder_path = folder_path.strip().strip('"').strip("'")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")

        video_files = cls._get_video_files(folder_path)

        if not video_files:
            raise ValueError(
                f"No video files found in {folder_path}\n"
                f"Supported formats: {', '.join(cls.VIDEO_EXTENSIONS)}"
            )

        if meta_batch is None:
            return io.NodeOutput(*cls._load_all(video_files, folder_path, debug))
        else:
            return io.NodeOutput(*cls._load_batched(video_files, folder_path, debug, meta_batch, cls.hidden.unique_id))

    @classmethod
    def _load_all(cls, video_files, folder_path, debug):
        """Load all videos at once into a single tensor with concatenated audio."""
        if debug:
            print(f"[Load Videos] Loading {len(video_files)} videos from {folder_path}")

        all_frames = []
        all_audio = []
        expected_shape = None
        target_sample_rate = None
        expected_channels = None
        has_audio = True

        for idx, video_path in enumerate(video_files):
            if debug:
                print(f"[Load Videos] [{idx+1}/{len(video_files)}]: {os.path.basename(video_path)}", end=" ... ")

            frames, audio_dict, _ = cls._load_video_with_audio(video_path, target_sample_rate)
            expected_shape = cls._check_resolution(frames, expected_shape, video_path)
            all_frames.append(frames)

            if debug:
                print(f"{frames.shape[0]} frames")

            if has_audio:
                if audio_dict is None:
                    print(f"[Load Videos] Warning: no audio in {os.path.basename(video_path)}, audio output disabled")
                    has_audio = False
                else:
                    if target_sample_rate is None:
                        target_sample_rate = audio_dict["sample_rate"]
                        expected_channels = audio_dict["waveform"].shape[1]
                    else:
                        audio_dict = cls._normalize_audio_channels(audio_dict, expected_channels, video_path)
                    all_audio.append(audio_dict["waveform"])

        if debug:
            print(f"[Load Videos] Concatenating {len(video_files)} videos...")
        output = torch.cat(all_frames, dim=0)

        audio_output = None
        if has_audio and all_audio:
            audio_output = {
                "waveform": torch.cat(all_audio, dim=2),
                "sample_rate": target_sample_rate,
            }

        if debug:
            print(f"[Load Videos] Done")
        return (output, audio_output)

    @classmethod
    def _load_batched(cls, video_files, folder_path, debug, meta_batch, unique_id):
        """
        Load frames in chunks coordinated by the VHS BatchManager.
        Audio is pre-loaded in full on the first call and returned unchanged on
        every batch execution so the downstream node always receives the complete
        audio track (it overwrites the audio slot each pass, keeping the last value).
        """
        if unique_id not in meta_batch.inputs:
            total_frames = cls._count_total_frames(video_files)
            meta_batch.total_frames = min(meta_batch.total_frames, total_frames)
            if debug:
                print(f"[Load Videos] Batched: Starting new generator for {len(video_files)} videos ({total_frames} frames) in {folder_path}")

            audio_output = cls._preload_audio(video_files, debug)
            meta_batch.inputs[unique_id] = {
                'generator': cls._frame_generator(video_files, debug),
                'audio_output': audio_output,
            }

        state = meta_batch.inputs[unique_id]
        generator = state['generator']
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

            expected_shape = cls._check_resolution(
                frame_tensor.unsqueeze(0), expected_shape, video_path
            )
            batch_frames.append(frame_tensor)
            frames_collected += 1

        if not batch_frames:
            raise RuntimeError("Batched loader produced no frames")

        output = torch.stack(batch_frames, dim=0)

        if debug:
            print(f"[Load Videos] Batched: Yielding {output.shape[0]} frames  shape={tuple(output.shape)}")

        return (output, state['audio_output'])

    @classmethod
    def _load_video_with_audio(cls, video_path, target_sample_rate):
        """
        Decode all video frames and extract matching audio.
        Returns (frames_tensor, audio_dict_or_none, frame_count).
        """
        container = av.open(video_path)
        try:
            if len(container.streams.video) == 0:
                raise ValueError(f"No video stream found in: {video_path}")

            video_stream = container.streams.video[0]
            fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0

            frames = []
            for frame in container.decode(video=0):
                rgb = frame.to_ndarray(format="rgb24")
                frames.append(torch.from_numpy(rgb.astype(np.float32) / 255.0))

            if not frames:
                raise RuntimeError(f"No frames extracted from {video_path}")

        finally:
            container.close()

        frame_count = len(frames)
        frames_tensor = torch.stack(frames, dim=0)
        audio_dict = cls._extract_audio(video_path, frame_count, fps, target_sample_rate)
        return frames_tensor, audio_dict, frame_count

    @classmethod
    def _scan_video_timing(cls, video_path):
        """
        Decode video stream (discarding frame data) to count frames and get fps.
        Returns (frame_count, fps).
        Used by _preload_audio to avoid storing frame tensors during audio pre-loading.
        """
        container = av.open(video_path)
        try:
            if len(container.streams.video) == 0:
                raise ValueError(f"No video stream found in: {video_path}")

            video_stream = container.streams.video[0]
            fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0
            frame_count = sum(1 for _ in container.decode(video=0))

        finally:
            container.close()

        return frame_count, fps

    @classmethod
    def _extract_audio(cls, video_path, frame_count, fps, target_sample_rate):
        """
        Extract audio from the start of the file, trimmed to exactly frame_count / fps seconds.
        No seeking or PTS offset - audio and video both start at the beginning of the clip.
        Returns {"waveform": [1, channels, samples], "sample_rate": int} or None.
        """
        container = av.open(video_path)
        try:
            if len(container.streams.audio) == 0:
                return None

            audio_stream = container.streams.audio[0]
            actual_rate = int(audio_stream.sample_rate) if audio_stream.sample_rate else 44100
            out_rate = target_sample_rate if target_sample_rate is not None else actual_rate

            resampler = av.AudioResampler(format='fltp', rate=out_rate)
            audio_frames = []

            for frame in container.decode(audio_stream):
                for rf in resampler.resample(frame):
                    audio_frames.append(rf.to_ndarray())

            for rf in resampler.resample(None):
                audio_frames.append(rf.to_ndarray())

        finally:
            container.close()

        if not audio_frames:
            return None

        waveform = np.concatenate(audio_frames, axis=1)  # [channels, all_samples]

        # Trim to exactly the number of samples that matches the decoded frame count.
        # Using frame_count / fps rather than PTS avoids drift caused by non-zero first_pts
        # (common when encoders start at pts=1 rather than pts=0).
        expected_samples = round(frame_count / fps * out_rate) if fps > 0 else waveform.shape[1]
        waveform = waveform[:, :expected_samples]

        if waveform.shape[1] < expected_samples:
            waveform = np.pad(waveform, ((0, 0), (0, expected_samples - waveform.shape[1])))

        return {
            "waveform": torch.from_numpy(waveform).float().unsqueeze(0),
            "sample_rate": out_rate,
        }

    @classmethod
    def _preload_audio(cls, video_files, debug):
        """
        Pre-extract and concatenate audio for all videos.
        Returns a complete audio dict or None if any video lacks audio.
        """
        all_audio = []
        target_sample_rate = None
        expected_channels = None

        for video_path in video_files:
            frame_count, fps = cls._scan_video_timing(video_path)
            audio_dict = cls._extract_audio(video_path, frame_count, fps, target_sample_rate)

            if audio_dict is None:
                print(f"[Load Videos] Warning: no audio in {os.path.basename(video_path)}, audio output disabled")
                return None

            if target_sample_rate is None:
                target_sample_rate = audio_dict["sample_rate"]
                expected_channels = audio_dict["waveform"].shape[1]
            else:
                audio_dict = cls._normalize_audio_channels(audio_dict, expected_channels, video_path)
            all_audio.append(audio_dict["waveform"])

        if not all_audio:
            return None

        return {
            "waveform": torch.cat(all_audio, dim=2),
            "sample_rate": target_sample_rate,
        }

    @classmethod
    def _normalize_audio_channels(cls, audio_dict, expected_channels, video_path):
        """Normalize channel count to match expected_channels. Handles mono/stereo mismatches."""
        waveform = audio_dict["waveform"]  # [1, channels, samples]
        actual_channels = waveform.shape[1]

        if actual_channels == expected_channels:
            return audio_dict

        if expected_channels == 2 and actual_channels == 1:
            waveform = torch.cat([waveform, waveform], dim=1)
        elif expected_channels == 1 and actual_channels == 2:
            waveform = waveform.mean(dim=1, keepdim=True)
        else:
            raise ValueError(
                f"Audio channel count mismatch in {os.path.basename(video_path)}: "
                f"expected {expected_channels}, got {actual_channels}"
            )

        return {"waveform": waveform, "sample_rate": audio_dict["sample_rate"]}

    @classmethod
    def _get_video_files(cls, folder_path):
        """Return naturally-sorted list of video file paths in folder_path."""
        video_files = []
        for f in os.listdir(folder_path):
            full_path = os.path.join(folder_path, f)
            if os.path.isfile(full_path) and cls._is_video_file(f):
                video_files.append(full_path)

        def natural_sort_key(path):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', os.path.basename(path))]

        return sorted(video_files, key=natural_sort_key)

    @classmethod
    def _is_video_file(cls, filename):
        """Return True if filename has a supported video extension."""
        _, ext = os.path.splitext(filename)
        return ext.lstrip('.').lower() in cls.VIDEO_EXTENSIONS

    @classmethod
    def _check_resolution(cls, frames, expected_shape, video_path):
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

    @classmethod
    def _count_total_frames(cls, video_files):
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

    @classmethod
    def _frame_generator(cls, video_files, debug):
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
