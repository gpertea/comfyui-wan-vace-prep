import os

from comfy_api.latest import io


class WanVACEBatchContext(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanVACEBatchContext",
            display_name="🪐 VACE Batch Context",
            category="Wan VACE Prep/VACE",
            description="Establishes iteration context for batch video join processing.",
            is_input_list=True,
            inputs=[
                io.String.Input("input_list", force_input=True),
                io.String.Input("input_dir", default="",
                    tooltip="Directory containing input videos"),
                io.String.Input("project_name", default="",
                    tooltip="Project name - workflow files will be created under ComfyUI/output/project_name."),
                io.Int.Input("index", default=0, min=0,
                    tooltip="Current iteration index (0 based)"),
                io.Boolean.Input("debug", default=False,
                    tooltip="Log some details to the console"),
                io.Boolean.Input("make_loop", default=False,
                    tooltip="Generate an extra loop-closing transition between the last and first video"),
            ],
            outputs=[
                io.String.Output("work_dir"),
                io.String.Output("workfile_prefix"),
                io.String.Output("video_1_filename"),
                io.String.Output("video_2_filename"),
                io.Boolean.Output("is_first"),
                io.Boolean.Output("is_last"),
                io.Boolean.Output("assemble_video"),
            ],
        )

    @classmethod
    def execute(cls, input_list, input_dir, project_name, index, debug, make_loop) -> io.NodeOutput:
        # is_input_list=True makes ComfyUI deliver every input as a list, including
        # scalar widgets. input_list is the multi-element upstream list; the rest are
        # single-element lists we unwrap with [0].
        input_dir = input_dir[0]
        project_name = project_name[0].strip()
        index = index[0]
        debug = debug[0]
        make_loop = make_loop[0]

        # Validate input list
        list_length = len(input_list)

        if list_length < 2:
            raise ValueError(
                f"Need at least 2 videos to create transitions, found {list_length}"
            )

        # Validate index bounds
        max_index = list_length - 1 if make_loop else list_length - 2
        if index < 0 or index > max_index:
            raise ValueError(
                f"Index {index} out of range (valid: 0-{max_index} for {list_length} videos)"
            )

        # Construct paths — uses forward slashes intentionally; these are
        # relative paths for ComfyUI's output system, not OS filesystem paths.
        if project_name:
            work_dir = f"{project_name}/vace-work"
        else:
            work_dir = "vace-work"
        padded_index = f"{index:03d}"
        workfile_prefix = f"{work_dir}/index{padded_index}"

        # Set iteration flags
        if make_loop:
            is_first = False
            is_last = False
        else:
            is_first = (index == 0)
            is_last = (index == max_index)

        # Extract filenames
        if make_loop and index == list_length - 1:
            # Loop-closing iteration: last video → first video
            v1 = input_list[-1]
            v2 = input_list[0]
        else:
            v1 = input_list[index]
            v2 = input_list[index + 1]

        video_1_filename = os.path.join(input_dir, v1) if input_dir else v1
        video_2_filename = os.path.join(input_dir, v2) if input_dir else v2

        assemble_video = (not make_loop and index == list_length - 2) or (make_loop and index == list_length - 1)

        if make_loop and index == list_length - 2:
            print(f"[VACE Batch Context] Loop enabled: run once more at index {list_length - 1} to generate loop transition.")

        if debug:
            is_loop_iter = make_loop and index == list_length - 1
            print(f"\n[VACE Batch Context] === Start ===")
            print(f"[VACE Batch Context] Index: {index} (videos {index+1}-{index+2 if not is_loop_iter else 1} of {list_length}){' [LOOP TRANSITION]' if is_loop_iter else ''}")
            if not make_loop:
                print(f"[VACE Batch Context] {'[FIRST]' if is_first else ''} {'[LAST]' if is_last else ''}")
            print(f"[VACE Batch Context] Input directory: {input_dir}")
            print(f"[VACE Batch Context] Video 1: {v1}")
            print(f"[VACE Batch Context] Video 2: {v2}")
            print(f"[VACE Batch Context] Work prefix: {workfile_prefix}")
            print(f"[VACE Batch Context] === End ===")

        return io.NodeOutput(work_dir, workfile_prefix, video_1_filename, video_2_filename, is_first, is_last, assemble_video)

