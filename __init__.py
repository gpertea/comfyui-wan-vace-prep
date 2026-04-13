"""
Wan VACE Prep - Custom nodes for preparing videos for Wan VACE generation
"""

from .vace_join import WanVACEPrep, WanVACEPrepBatch
from .vace_extend import WanVACEExtend
from .load_videos_from_folder import LoadVideosFromFolderSimple
from .vace_batch_context import WanVACEBatchContext
from .frame_number_overlay import FrameNumberOverlay
from .vace_outpaint import VACEOutpaint

NODE_CLASS_MAPPINGS = {
    "WanVACEPrep": WanVACEPrep,
    "WanVACEPrepBatch": WanVACEPrepBatch,
    "WanVACEExtend": WanVACEExtend,
    "LoadVideosFromFolderSimple": LoadVideosFromFolderSimple,
    "WanVACEBatchContext": WanVACEBatchContext,
    "FrameNumberOverlay": FrameNumberOverlay,
    "VACEOutpaint": VACEOutpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVACEPrep": "🪐 VACE Join",
    "WanVACEPrepBatch": "🪐 VACE Join (Batch)",
    "WanVACEExtend": "🪐 VACE Extend",
    "LoadVideosFromFolderSimple": "🪐 Load Videos From Folder (Simple)",
    "WanVACEBatchContext": "🪐 VACE Batch Context",
    "FrameNumberOverlay": "🪐 Frame Number Overlay",
    "VACEOutpaint": "🪐 VACE Outpaint",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]