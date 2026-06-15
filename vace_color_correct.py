import math

import torch
import torch.nn.functional as F
from comfy_api.latest import io


class WanVACETransitionColorCorrect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanVACETransitionColorCorrect",
            display_name="VACE Transition Color Correct",
            category="Wan VACE Prep/VACE",
            description=(
                "Applies bounded luma/chroma correction to a VACE transition batch "
                "using the original control-video context frames as anchors."
            ),
            inputs=[
                io.Image.Input("control_video"),
                io.Image.Input("vace_output"),
                io.Int.Input("context_frames", default=8, min=1, max=120, step=1),
                io.Boolean.Input("enabled", default=True),
                io.Float.Input("correction_strength", default=0.75, min=0.0, max=2.0, step=0.05),
                io.Float.Input("luma_strength", default=0.75, min=0.0, max=2.0, step=0.05),
                io.Float.Input("chroma_strength", default=0.60, min=0.0, max=2.0, step=0.05),
                io.Float.Input("smooth_window", default=12.0, min=0.0, max=40.0, step=0.5),
                io.Float.Input("min_gain", default=0.75, min=0.1, max=1.0, step=0.01),
                io.Float.Input("max_gain", default=1.25, min=1.0, max=3.0, step=0.01),
                io.Boolean.Input("debug", default=False),
            ],
            outputs=[io.Image.Output("vace_output")],
        )

    @staticmethod
    def _log(debug, message):
        if debug:
            print(f"[VACE Transition Color Correct] {message}")

    @staticmethod
    def _smooth_series(values, sigma):
        if sigma <= 0 or values.shape[0] < 3:
            return values

        radius = max(1, int(math.ceil(float(sigma) * 3.0)))
        positions = torch.arange(
            -radius,
            radius + 1,
            dtype=values.dtype,
            device=values.device,
        )
        kernel = torch.exp(-0.5 * (positions / float(sigma)) ** 2)
        kernel = kernel / kernel.sum()

        ## convolve each RGB channel independently over frame time.
        series = values.transpose(0, 1).unsqueeze(0)
        padded = F.pad(series, (radius, radius), mode="replicate")
        weight = kernel.view(1, 1, -1).repeat(values.shape[1], 1, 1)
        smoothed = F.conv1d(padded, weight, groups=values.shape[1])
        return smoothed.squeeze(0).transpose(0, 1)

    @classmethod
    def _interior_targets(cls, control_means, vace_means, context_frames, smooth_window):
        frame_count = vace_means.shape[0]
        context = min(int(context_frames), frame_count // 2, control_means.shape[0] // 2)
        generated_start = context
        generated_end = frame_count - context

        target = cls._smooth_series(vace_means, smooth_window)
        target[:context] = control_means[:context]
        target[generated_end:] = control_means[-context:]

        ## linearly bridge from the first source context to the second source context.
        anchor_count = min(4, context)
        before_anchor = control_means[context - anchor_count:context].mean(dim=0)
        after_anchor = control_means[-context:][:anchor_count].mean(dim=0)
        generated_count = generated_end - generated_start
        if generated_count > 0:
            progress = torch.linspace(
                0.0,
                1.0,
                generated_count,
                dtype=vace_means.dtype,
                device=vace_means.device,
            ).unsqueeze(1)
            linear = before_anchor * (1.0 - progress) + after_anchor * progress
            target[generated_start:generated_end] = (
                0.5 * target[generated_start:generated_end] + 0.5 * linear
            )

        return target, context, generated_start, generated_end

    @staticmethod
    def _calculate_gain(
        current_rgb,
        target_rgb,
        correction_strength,
        luma_strength,
        chroma_strength,
        min_gain,
        max_gain,
    ):
        current_rgb = current_rgb.clamp_min(1e-4)
        target_rgb = target_rgb.clamp_min(1e-4)
        channel_gain = target_rgb / current_rgb

        luma_weights = torch.tensor(
            [0.2126, 0.7152, 0.0722],
            dtype=current_rgb.dtype,
            device=current_rgb.device,
        )
        current_luma = (current_rgb * luma_weights).sum(dim=1, keepdim=True).clamp_min(1e-4)
        target_luma = (target_rgb * luma_weights).sum(dim=1, keepdim=True).clamp_min(1e-4)
        luma_gain = target_luma / current_luma
        chroma_gain = channel_gain / luma_gain.clamp_min(1e-6)

        combined_gain = (1.0 + luma_strength * (luma_gain - 1.0)) * (
            1.0 + chroma_strength * (chroma_gain - 1.0)
        )
        gain = 1.0 + correction_strength * (combined_gain - 1.0)
        return gain.clamp(min_gain, max_gain)

    @classmethod
    def execute(
        cls,
        control_video,
        vace_output,
        context_frames,
        enabled,
        correction_strength,
        luma_strength,
        chroma_strength,
        smooth_window,
        min_gain,
        max_gain,
        debug,
    ):
        if not enabled or correction_strength <= 0:
            cls._log(debug, "disabled; returning input unchanged")
            return io.NodeOutput(vace_output)

        if vace_output.ndim != 4 or control_video.ndim != 4:
            raise ValueError("control_video and vace_output must be IMAGE tensors")

        if vace_output.shape[-1] < 3 or control_video.shape[-1] < 3:
            raise ValueError("control_video and vace_output must have RGB channels")

        frame_count = int(vace_output.shape[0])
        control_count = int(control_video.shape[0])
        context = min(int(context_frames), frame_count // 2, control_count // 2)
        if context < 1 or frame_count <= context * 2:
            cls._log(debug, "not enough frames for context correction; returning input unchanged")
            return io.NodeOutput(vace_output)

        if min_gain > max_gain:
            raise ValueError("min_gain must be less than or equal to max_gain")

        source = vace_output.to(dtype=torch.float32)
        control = control_video.to(device=source.device, dtype=torch.float32)
        control_means = control[:, :, :, :3].mean(dim=(1, 2))
        vace_means = source[:, :, :, :3].mean(dim=(1, 2))
        target_means, context, generated_start, generated_end = cls._interior_targets(
            control_means,
            vace_means,
            context,
            float(smooth_window),
        )

        gains = cls._calculate_gain(
            vace_means,
            target_means,
            float(correction_strength),
            float(luma_strength),
            float(chroma_strength),
            float(min_gain),
            float(max_gain),
        )

        corrected = source.clone()
        corrected[:, :, :, :3] = (source[:, :, :, :3] * gains[:, None, None, :]).clamp(0.0, 1.0)
        corrected = corrected.to(dtype=vace_output.dtype)

        cls._log(
            debug,
            (
                f"frames={frame_count}, context={context}, "
                f"generated={generated_start}:{generated_end}, "
                f"gain_min={float(gains.min()):.4f}, gain_max={float(gains.max()):.4f}"
            ),
        )
        return io.NodeOutput(corrected)
