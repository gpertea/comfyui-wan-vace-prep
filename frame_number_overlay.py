import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from comfy_api.latest import io


def _load_font(font_size):
    paths = [
        "/System/Library/Fonts/Supplemental/Courier New.ttf",       # macOS
        "/System/Library/Fonts/Menlo.ttc",                           # macOS fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",       # Linux
        "C:/Windows/Fonts/cour.ttf",                                  # Windows
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, font_size)
    try:
        return ImageFont.load_default(size=font_size)  # PIL >= 10
    except TypeError:
        return ImageFont.load_default()


class FrameNumberOverlay(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FrameNumberOverlay",
            display_name="🪐 Frame Number Overlay (Experimental)",
            category="Wan VACE Prep/utility",
            description="Burns the frame number into every frame of an IMAGE batch as a text overlay.",
            is_experimental=True,
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("font_size", default=32, min=8, max=256),
                io.Int.Input("start_index", default=0, min=0, max=99999),
                io.Combo.Input("position", options=["top-left", "top-right", "bottom-left", "bottom-right"]),
                io.Int.Input("padding", default=10, min=0, max=256),
                io.String.Input("font_color", default="white"),
                io.String.Input("prefix", default=""),
                io.String.Input("suffix", default=""),
            ],
            outputs=[
                io.Image.Output("images"),
            ],
        )

    @classmethod
    def execute(cls, images, font_size=32, start_index=0, position="top-left", padding=10,
                font_color="white", prefix="", suffix="") -> io.NodeOutput:
        font = _load_font(font_size)
        result = []

        for i, frame_tensor in enumerate(images):
            # [H, W, C] float32 [0,1] → uint8 PIL image
            frame_np = (frame_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(frame_np, mode="RGB")
            draw = ImageDraw.Draw(img)

            label = f"{prefix}{start_index + i}{suffix}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            W, H = img.size
            x = padding if "left" in position else (W - text_w - padding)
            y = padding if "top" in position else (H - text_h - padding)

            # Black outline for contrast
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text((x + dx, y + dy), label, font=font, fill=(0, 0, 0))
            draw.text((x, y), label, font=font, fill=font_color)

            frame_out = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
            result.append(frame_out)

        return io.NodeOutput(torch.stack(result))

