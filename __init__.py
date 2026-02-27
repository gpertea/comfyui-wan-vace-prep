"""
Wan VACE Prep - Custom nodes for preparing videos for Wan VACE generation
"""

import os

import av
import folder_paths
import server
from aiohttp import web

from .wan_vace_prep import NODE_CLASS_MAPPINGS as PREP_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PREP_DISPLAY
from .wan_vace_prep_inline import NODE_CLASS_MAPPINGS as SMOOTH_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SMOOTH_DISPLAY
from .wan_vace_extend import NODE_CLASS_MAPPINGS as EXTEND_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as EXTEND_DISPLAY
from .load_videos import NODE_CLASS_MAPPINGS as LOAD_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LOAD_DISPLAY
from .wan_vace_batch_context import NODE_CLASS_MAPPINGS as CONTEXT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CONTEXT_DISPLAY
from .frame_selector import NODE_CLASS_MAPPINGS as VFS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VFS_DISPLAY

NODE_CLASS_MAPPINGS = {**PREP_MAPPINGS, **SMOOTH_MAPPINGS, **EXTEND_MAPPINGS, **LOAD_MAPPINGS, **CONTEXT_MAPPINGS, **VFS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**PREP_DISPLAY, **SMOOTH_DISPLAY, **EXTEND_DISPLAY, **LOAD_DISPLAY, **CONTEXT_DISPLAY, **VFS_DISPLAY}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


# ── Helpers ─────────────────────────────────────────────────────────

def _resolve_video_path(filename):
    """Resolve a video filename to its full path within managed directories.

    Returns (path, error_response) tuple. If error_response is not None,
    the path is invalid and the error should be returned to the client.
    """
    if not filename:
        return None, web.Response(status=400, text="No filename provided")

    if not folder_paths.exists_annotated_filepath(filename):
        return None, web.Response(status=404, text="Video file not found")

    video_path = folder_paths.get_annotated_filepath(filename)

    if not os.path.isfile(video_path):
        return None, web.Response(status=404, text="Video file not found")

    return video_path, None


# ── API Routes ──────────────────────────────────────────────────────
# Note: Video serving and upload are handled by ComfyUI's native
# /api/view and upload systems (triggered by video_upload: True).
# We only provide query_video for PyAV-based metadata extraction.

@server.PromptServer.instance.routes.get("/visual_frame_selector/v2/query_video")
async def query_video(request):
    """Return video metadata (fps, total_frames, dimensions, has_audio, file_size).

    Also performs a decode test on the first frame to catch codec issues
    (e.g. ffv1 in mkv) early, before the user tries to run the workflow.
    """
    filename = request.query.get("filename", "")
    video_path, error = _resolve_video_path(filename)
    if error:
        return error

    container = None
    try:
        container = av.open(video_path)

        if len(container.streams.video) == 0:
            return web.json_response({
                "error": "No video stream found in file",
            }, status=422)

        video_stream = container.streams.video[0]
        codec_name = video_stream.codec_context.name if video_stream.codec_context else "unknown"

        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        total_frames = video_stream.frames
        width = video_stream.width
        height = video_stream.height

        if total_frames == 0:
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)

        try:
            container.seek(0)
            for frame in container.decode(video=0):
                frame.to_ndarray(format="rgb24")
                break
        except Exception as decode_err:
            container.close()
            container = None
            return web.json_response({
                "error": f"Codec '{codec_name}' is not supported for preview/processing: {str(decode_err)}",
                "codec": codec_name,
            }, status=422)

        has_audio = len(container.streams.audio) > 0
        container.close()
        container = None

        return web.json_response({
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "has_audio": has_audio,
            "file_size": os.path.getsize(video_path),
        })

    except av.error.InvalidDataError as e:
        return web.json_response({
            "error": f"Cannot read video file: {str(e)}",
        }, status=422)

    except Exception as e:
        return web.json_response({
            "error": f"Error reading video: {str(e)}",
        }, status=500)

    finally:
        if container is not None:
            container.close()