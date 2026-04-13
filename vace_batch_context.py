import os

class WanVACEBatchContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": ("STRING", {"forceInput": True}),
                "input_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing input videos"
                }),
                "project_name": ("STRING", {
                    "default": "",
                    "tooltip": "Project name - workflow files will be created under ComfyUI/output/project_name."
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Current iteration index (0 based)"
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log some details to the console"
                }),
                "make_loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generate an extra loop-closing transition between the last and first video"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("work_dir", "workfile_prefix", "video_1_filename", "video_2_filename", "is_first", "is_last", "assemble_video")
    FUNCTION = "setup_context"
    CATEGORY = "video/VACE"
    DESCRIPTION = """
    Establishes iteration context for batch video join processing.
    """
    INPUT_IS_LIST = True

    def setup_context(self, **kwargs):
        # INPUT_IS_LIST=True makes ComfyUI deliver every input as a list.
        # The **kwargs signature is required because list-mode inputs can't
        # be declared as positional parameters — ComfyUI wraps each value
        # in a list, including scalars, so we unwrap with [0] here.
        input_dir = kwargs.get('input_dir', [""])[0]
        input_list = kwargs.get('input_list', [])
        project_name = kwargs.get('project_name', [""])[0].strip()
        index = kwargs.get('index', [0])[0]
        debug = kwargs.get('debug', [False])[0]
        make_loop = kwargs.get('make_loop', [False])[0]

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

        return (work_dir, workfile_prefix, video_1_filename, video_2_filename, is_first, is_last, assemble_video)

