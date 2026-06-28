import torch
import comfy.model_management
import comfy.utils
import comfy.clip_vision
import node_helpers
from comfy_api.latest import io


class WanFirstMiddleLastFrameToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanFMLF2V",
            display_name="🪐 Wan First/Middle/Last Frame to Video (Experimental)",
            category="Wan VACE Prep/conditioning",
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=16384, step=16),
                io.Int.Input("height", default=480, min=16, max=16384, step=16),
                io.Int.Input("length", default=81, min=1, max=16384, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("middle_image", optional=True),
                io.Float.Input("middle_frame", default=0.5, min=0.0, max=1.0, step=0.01, optional=True),
                io.Image.Input("end_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_middle_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                start_image=None, middle_image=None, middle_frame=0.5, end_image=None,
                clip_vision_start_image=None, clip_vision_middle_image=None,
                clip_vision_end_image=None) -> io.NodeOutput:
        spacial_scale = vae.spacial_compression_encode()
        latent = torch.zeros(
            [batch_size, vae.latent_channels, ((length - 1) // 4) + 1,
             height // spacial_scale, width // spacial_scale],
            device=comfy.model_management.intermediate_device()
        )

        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(
                middle_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0

        if middle_image is not None:
            mid_start = int(((middle_frame * length) // 4) * 4)
            mid_start = min(mid_start, length - 1)
            mid_end = min(mid_start + middle_image.shape[0], length)
            image[mid_start:mid_end] = middle_image[:mid_end - mid_start]
            mask[:, :, mid_start + 3:mid_end + 3] = 0.0

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        clip_vision_output = None
        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image

        if clip_vision_middle_image is not None:
            if clip_vision_output is not None:
                states = torch.cat(
                    [clip_vision_output.penultimate_hidden_states,
                     clip_vision_middle_image.penultimate_hidden_states],
                    dim=-2
                )
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_middle_image

        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat(
                    [clip_vision_output.penultimate_hidden_states,
                     clip_vision_end_image.penultimate_hidden_states],
                    dim=-2
                )
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(
                positive, {"clip_vision_output": clip_vision_output}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"clip_vision_output": clip_vision_output}
            )

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)
