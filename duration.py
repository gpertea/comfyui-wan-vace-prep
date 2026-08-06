from comfy_api.latest import io


class Duration(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Duration",
            display_name="🪐 MiniMax H3 Duration",
            category="Wan VACE Prep/MinimaxH3",
            description="Converts a requested duration in seconds to a MiniMax H3-compatible frame count "
                         "(rounded up to the nearest 17k+5 at 24 fps) and the resulting revised duration.",
            inputs=[
                io.Float.Input("seconds", default=6.0, min=0.0, step=0.01),
            ],
            outputs=[
                io.Int.Output("frames"),
                io.Float.Output("revised_seconds"),
            ],
        )

    @classmethod
    def execute(cls, seconds: float) -> io.NodeOutput:
        raw_frames = max(5, round(seconds * 24))
        frames = raw_frames + (5 - (raw_frames % 17)) % 17
        revised_seconds = frames / 24.0
        return io.NodeOutput(frames, revised_seconds)
