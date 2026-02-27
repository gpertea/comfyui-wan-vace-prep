import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ── Constants ──────────────────────────────────────────────────────

const NODE_CLASS = "VisualFrameSelector";
const API_PREFIX = "/visual_frame_selector/v2";

const SVG = {
    play: `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M3 2 L3 14 L13 8 Z"/></svg>`,
    pause: `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2" width="3" height="12"/><rect x="10" y="2" width="3" height="12"/></svg>`,
    first: `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><rect x="2" y="2" width="2" height="12"/><path d="M6 8 L14 2 L14 14 Z"/></svg>`,
    prev: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor"><path stroke-width="2" d="M10 4 L4 8 L10 12"/></svg>`,
    next: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor"><path stroke-width="2" d="M6 4 L12 8 L6 12"/></svg>`,
    last: `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M2 2 L10 8 L2 14 Z"/><rect x="12" y="2" width="2" height="12"/></svg>`,
    jumpIn: `<svg width="16" height="16" viewBox="0 0 16 16" fill="#55f"><path d="M8 2 L4 2 L4 14 L8 14 L8 12 L6 12 L6 4 L8 4 Z"/><line x1="10" y1="8" x2="16" y2="8" stroke="#55f" stroke-width="1"/><polygon points="11,5 9,8 11,11"/></svg>`,
    setIn: `<svg width="16" height="16" viewBox="0 0 16 16" fill="#55f"><path d="M8 2 L4 2 L4 14 L8 14 L8 12 L6 12 L6 4 L8 4 Z"/></svg>`,
    playRange: `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M2 2 L2 14" stroke="#55f" stroke-width="2"/><path d="M14 2 L14 14" stroke="red" stroke-width="2"/><path d="M5 4 L11 8 L5 12 Z"/></svg>`,
    setOut: `<svg width="16" height="16" viewBox="0 0 16 16" fill="red"><path d="M8 2 L12 2 L12 14 L8 14 L8 12 L10 12 L10 4 L8 4 Z"/></svg>`,
    jumpOut: `<svg width="16" height="16" viewBox="0 0 16 16" fill="red"><path d="M8 2 L12 2 L12 14 L8 14 L8 12 L10 12 L10 4 L8 4 Z"/><line x1="0" y1="8" x2="6" y2="8" stroke="red" stroke-width="1"/><polygon points="5,5 7,8 5,11"/></svg>`,
};

// ── State Management ───────────────────────────────────────────────

function createState() {
    return {
        video: {
            filename: "",
            fps: 0,
            totalFrames: 0,
            width: 0,
            height: 0,
            hasAudio: false,
            loaded: false,
        },
        selection: {
            currentFrame: 0,
            startFrame: 0,
            endFrame: 0,
        },
        ui: {
            isPlaying: false,
            isDragging: false,
            draggingMarker: null,
        },
    };
}

function updateState(ctx, changes, source) {
    const { node, state, dom, widgets } = ctx;

    const prevStart = state.selection.startFrame;
    const prevEnd = state.selection.endFrame;
    const prevCurrent = state.selection.currentFrame;

    if (changes.video) Object.assign(state.video, changes.video);
    if (changes.selection) Object.assign(state.selection, changes.selection);
    if (changes.ui) Object.assign(state.ui, changes.ui);

    // Enforce constraints
    if (state.video.totalFrames > 0) {
        const maxFrame = state.video.totalFrames - 1;
        state.selection.currentFrame = clamp(state.selection.currentFrame, 0, maxFrame);
        state.selection.startFrame = clamp(state.selection.startFrame, 0, maxFrame);
        state.selection.endFrame = clamp(state.selection.endFrame, 0, maxFrame);

        if (state.selection.endFrame <= state.selection.startFrame) {
            state.selection.endFrame = Math.min(state.selection.startFrame + 1, maxFrame);
        }
    }

    // Sync widgets (guard against re-entry from widget source)
    if (source !== "widget") {
        syncWidgetValue(widgets.currentFrame, state.selection.currentFrame);
        syncWidgetValue(widgets.startFrame, state.selection.startFrame);
        syncWidgetValue(widgets.endFrame, state.selection.endFrame);
    }

    // Update video seek
    const video = dom.video;
    if (video && source !== "video-timeupdate" && state.selection.currentFrame !== prevCurrent && video.duration) {
        // Map frame to time through video.duration rather than frame/fps.
        // The browser's duration (from container metadata) can differ slightly
        // from totalFrames/fps, so using duration directly avoids drift.
        const total = state.video.totalFrames;
        const time = total > 1
            ? (state.selection.currentFrame / (total - 1)) * video.duration
            : 0;
        if (Math.abs(video.currentTime - time) > 0.01) {
            video.currentTime = time;
        }
    }

    // Update scrubber markers
    if (state.video.totalFrames > 0) {
        const total = state.video.totalFrames;
        if (state.selection.startFrame !== prevStart || changes.video) {
            setMarkerPosition(dom.inMarker, dom.inLine, state.selection.startFrame, total);
        }
        if (state.selection.endFrame !== prevEnd || changes.video) {
            setMarkerPosition(dom.outMarker, dom.outLine, state.selection.endFrame, total);
        }
    }

    // Update progress bar
    if (video && video.duration && dom.progress) {
        dom.progress.style.width = (video.currentTime / video.duration) * 100 + "%";
    }

    updateStatusLine(ctx);

    if (node.graph) {
        node.graph.setDirtyCanvas(true, true);
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
}

let _syncing = false;
function syncWidgetValue(widget, value) {
    if (_syncing) return;
    if (widget && widget.value !== value) {
        _syncing = true;
        widget.value = value;
        // Setting .value alone may not trigger reactivity in all renderers.
        // Fire the callback to ensure the UI updates.
        if (widget.callback) {
            widget.callback(value);
        }
        _syncing = false;
    }
}

function setMarkerPosition(marker, line, frame, totalFrames) {
    if (!totalFrames || totalFrames <= 1) return;
    const percent = (frame / (totalFrames - 1)) * 100;
    marker.style.left = percent + "%";
    line.style.left = percent + "%";
}

function updateStatusLine(ctx) {
    const { state, dom } = ctx;
    const wasVisible = dom.statusRow.style.display === "flex";

    if (!state.video.loaded) {
        dom.statusRow.style.display = "none";
        if (wasVisible && ctx._remeasureControls) ctx._remeasureControls();
        return;
    }
    dom.statusRow.style.display = "flex";
    if (!wasVisible && ctx._remeasureControls) ctx._remeasureControls();

    const count = state.selection.endFrame - state.selection.startFrame + 1;
    const total = state.video.totalFrames;

    dom.statusText.textContent = `Total frames: ${total} | Selected frames: ${count}`;
}

// ── DOM Construction ───────────────────────────────────────────────

function buildScrubber() {
    const scrubber = document.createElement("div");
    scrubber.style.cssText = "position:relative;height:30px;background:#222;cursor:pointer;margin-top:8px;";

    const progress = document.createElement("div");
    progress.style.cssText = "height:100%;width:0%;background:#445566;pointer-events:none;";
    scrubber.appendChild(progress);

    const inMarker = document.createElement("div");
    inMarker.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" style="display:block;"><path d="M6 2 L10 10 L2 10 Z" fill="#55f"/></svg>`;
    inMarker.style.cssText = "position:absolute;left:0%;bottom:0;transform:translateX(-50%) translateY(80%);pointer-events:auto;cursor:ew-resize;z-index:2;";

    const inLine = document.createElement("div");
    inLine.style.cssText = "position:absolute;left:0%;top:0;width:2px;height:100%;background:#55f;pointer-events:none;opacity:0.5;transform:translateX(-1px);";

    const outMarker = document.createElement("div");
    outMarker.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" style="display:block;"><path d="M6 10 L2 2 L10 2 Z" fill="red"/></svg>`;
    outMarker.style.cssText = "position:absolute;left:100%;top:0;transform:translateX(-50%) translateY(-80%);pointer-events:auto;cursor:ew-resize;z-index:2;";

    const outLine = document.createElement("div");
    outLine.style.cssText = "position:absolute;left:100%;top:0;width:2px;height:100%;background:red;pointer-events:none;opacity:0.5;transform:translateX(-1px);";

    scrubber.append(inLine, outLine, inMarker, outMarker);

    return { scrubber, progress, inMarker, inLine, outMarker, outLine };
}

function buildButton(svgContent, title) {
    const btn = document.createElement("button");
    btn.innerHTML = svgContent;
    btn.title = title;
    btn.style.cssText = `
        display:flex;align-items:center;justify-content:center;flex-shrink:0;
        width:28px;height:24px;padding:0;
        background-color:#2A2A2A;border:1px solid #444;border-radius:4px;
        color:#ddd;cursor:pointer;outline:none;
    `;
    btn.onmouseenter = () => { btn.style.backgroundColor = "#444"; btn.style.borderColor = "#666"; };
    btn.onmouseleave = () => { btn.style.backgroundColor = "#2A2A2A"; btn.style.borderColor = "#444"; };
    return btn;
}

function buildControlBar() {
    const controlBar = document.createElement("div");
    controlBar.style.cssText = "margin-top:8px;display:flex;flex-direction:column;gap:4px;align-items:center;justify-content:center;";

    const row1 = document.createElement("div");
    row1.style.cssText = "display:flex;gap:4px;align-items:center;justify-content:center;";

    const firstBtn = buildButton(SVG.first, "First frame");
    const prevBtn = buildButton(SVG.prev, "Previous frame");
    const playPauseBtn = buildButton(SVG.play, "Play/Pause");
    const nextBtn = buildButton(SVG.next, "Next frame");
    const lastBtn = buildButton(SVG.last, "Last frame");
    row1.append(firstBtn, prevBtn, playPauseBtn, nextBtn, lastBtn);

    const row2 = document.createElement("div");
    row2.style.cssText = "display:flex;gap:4px;align-items:center;justify-content:center;";

    const jumpInBtn = buildButton(SVG.jumpIn, "Jump to start_frame");
    const setInBtn = buildButton(SVG.setIn, "Set start_frame");
    const playRangeBtn = buildButton(SVG.playRange, "Play from start_frame to end_frame");
    const setOutBtn = buildButton(SVG.setOut, "Set end_frame");
    const jumpOutBtn = buildButton(SVG.jumpOut, "Jump to end_frame");
    row2.append(jumpInBtn, setInBtn, playRangeBtn, setOutBtn, jumpOutBtn);

    const statusRow = document.createElement("div");
    statusRow.style.cssText = "display:none;gap:8px;align-items:center;justify-content:center;margin-top:4px;";

    const statusText = document.createElement("span");
    statusText.style.cssText = "color:#aaa;font-size:12px;white-space:nowrap;";
    statusText.textContent = "";
    statusRow.appendChild(statusText);

    controlBar.append(row1, row2, statusRow);

    return {
        element: controlBar,
        buttons: { firstBtn, prevBtn, playPauseBtn, nextBtn, lastBtn, jumpInBtn, setInBtn, playRangeBtn, setOutBtn, jumpOutBtn },
        statusRow,
        statusText,
    };
}

// ── Native Video Element Discovery ─────────────────────────────────

/**
 * Find the native video element created by ComfyUI's video-preview widget.
 * Once found, we configure it (disable controls/loop) and wire our events.
 *
 * Uses a MutationObserver on the widget container so that controls are
 * suppressed the instant the <video> element enters the DOM — before the
 * browser renders it. This eliminates the flash of native controls on
 * workflow restore, where ComfyUI recreates the <video> synchronously.
 *
 * Falls back to polling for the widget container itself if it isn't
 * present yet (e.g. video-preview widget added after onNodeCreated).
 *
 * Uses the standard widget API which works in both legacy mode and
 * compatibility mode.
 */
function findNativeVideoElement(node, abortController, callback) {
    let settled = false;

    const onFound = (video, widget) => {
        if (settled) return;
        settled = true;
        // Suppress native controls immediately — before any paint — so they
        // never flash on workflow restore. hookVideoElement will also set
        // these, but doing it here closes the race window.
        video.controls = false;
        video.muted = true;
        callback({ video, widget });
    };

    // Attach a MutationObserver to a widget container element so we catch
    // the <video> element the moment it's inserted, before paint.
    const observeContainer = (widget) => {
        const container = widget.element;

        // Already present?
        const existing = container.querySelector("video");
        if (existing) {
            onFound(existing, widget);
            return;
        }

        const mo = new MutationObserver(() => {
            const video = container.querySelector("video");
            if (video) {
                mo.disconnect();
                onFound(video, widget);
            }
        });
        mo.observe(container, { childList: true, subtree: true });
        abortController.signal.addEventListener("abort", () => mo.disconnect());
    };

    // Find the video-preview widget. It may already exist or may be added
    // to node.widgets after onNodeCreated returns, so we poll for it briefly.
    const tryAttach = () => {
        for (const w of (node.widgets || [])) {
            if (w.name === "video-preview" && w.element) {
                observeContainer(w);
                return true;
            }
        }
        return false;
    };

    if (tryAttach()) return;

    let attempts = 0;
    const interval = setInterval(() => {
        attempts++;
        if (tryAttach() || settled) {
            clearInterval(interval);
        } else if (attempts > 100) { // 5 seconds
            clearInterval(interval);
            console.warn("[VisualFrameSelector] Could not find video-preview widget after 5s");
        }
    }, 50);

    abortController.signal.addEventListener("abort", () => clearInterval(interval));
}

// ── Event Wiring ───────────────────────────────────────────────────

function wireVideoEvents(ctx) {
    const video = ctx.dom.video;
    if (!video) return;

    // Tag so we don't double-wire
    if (video._vfsWired) return;
    video._vfsWired = true;

    // Video timeupdate -> state
    // Derive frame from position within video.duration (consistent with seek).
    video.addEventListener("timeupdate", () => {
        if (!ctx.state.video.fps || ctx.state.video.fps <= 0) return;
        const total = ctx.state.video.totalFrames;
        const maxFrame = Math.max(0, total - 1);
        let frame;
        if (video.duration > 0 && total > 1) {
            frame = Math.round((video.currentTime / video.duration) * (total - 1));
        } else {
            frame = Math.round(video.currentTime * ctx.state.video.fps);
        }
        frame = clamp(frame, 0, maxFrame);
        if (frame !== ctx.state.selection.currentFrame) {
            updateState(ctx, { selection: { currentFrame: frame } }, "video-timeupdate");
        }
        if (video.duration && ctx.dom.progress) {
            ctx.dom.progress.style.width = (video.currentTime / video.duration) * 100 + "%";
        }
    });

    video.addEventListener("play", () => {
        ctx.state.ui.isPlaying = true;
        ctx.dom.buttons.playPauseBtn.innerHTML = SVG.pause;
    });

    video.addEventListener("pause", () => {
        ctx.state.ui.isPlaying = false;
        ctx.dom.buttons.playPauseBtn.innerHTML = SVG.play;
    });
}

/**
 * Take over a video element: disable native controls, store reference, wire events.
 * Called on initial discovery and after each video file change (ComfyUI may
 * replace the <video> element when loading a new source).
 */
function hookVideoElement(ctx, video) {
    if (!video || video === ctx.dom.video) return false;

    video.controls = false;
    video.loop = false;
    video.muted = true;
    ctx.dom.video = video;
    wireVideoEvents(ctx);
    return true;
}

/**
 * Re-find and re-hook the native video element after a video file change.
 * ComfyUI may create a new <video> element when loading a new source.
 * Polls briefly since the new element may appear asynchronously.
 */
function rehookVideo(ctx) {
    const node = ctx.node;
    let attempts = 0;
    const check = () => {
        let video = null;
        for (const w of (node.widgets || [])) {
            if (w.name === "video-preview" && w.element) {
                video = w.element.querySelector("video");
                if (video) break;
            }
        }

        if (video && video !== ctx.dom.video) {
            hookVideoElement(ctx, video);
            return true;
        }
        return false;
    };

    // Try immediately
    if (check()) return;

    // Poll briefly — new element may appear after a tick
    const interval = setInterval(() => {
        attempts++;
        if (check() || attempts > 20) { // 1 second max
            clearInterval(interval);
        }
    }, 50);

    ctx.abortController.signal.addEventListener("abort", () => clearInterval(interval));
}

function wireControlEvents(ctx) {
    const { state, dom } = ctx;
    const { scrubber, progress, inMarker, outMarker, buttons } = dom;
    const abort = ctx.abortController;

    // Scrubber click-to-seek
    scrubber.addEventListener("mousedown", (e) => {
        if (e.target === scrubber || e.target === progress) {
            state.ui.isDragging = true;
            seekFromMouseEvent(ctx, e);
        }
    });

    // Marker dragging
    inMarker.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        state.ui.draggingMarker = "in";
    });

    outMarker.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        state.ui.draggingMarker = "out";
    });

    document.addEventListener("mousemove", (e) => {
        if (state.ui.draggingMarker) {
            handleMarkerDrag(ctx, e);
        } else if (state.ui.isDragging) {
            seekFromMouseEvent(ctx, e);
        }
    }, { signal: abort.signal });

    document.addEventListener("mouseup", () => {
        state.ui.isDragging = false;
        state.ui.draggingMarker = null;
    }, { signal: abort.signal });

    // Transport controls
    const seekToFrame = (frame) => {
        updateState(ctx, { selection: { currentFrame: frame } }, "transport");
    };

    buttons.firstBtn.onclick = () => seekToFrame(0);
    buttons.prevBtn.onclick = () => { dom.video?.pause(); seekToFrame(state.selection.currentFrame - 1); };
    buttons.playPauseBtn.onclick = () => {
        const v = dom.video;
        if (v) { if (v.paused) v.play(); else v.pause(); }
    };
    buttons.nextBtn.onclick = () => { dom.video?.pause(); seekToFrame(state.selection.currentFrame + 1); };
    buttons.lastBtn.onclick = () => seekToFrame(state.video.totalFrames - 1);

    // Marker controls
    buttons.jumpInBtn.onclick = () => seekToFrame(state.selection.startFrame);
    buttons.setInBtn.onclick = () => {
        updateState(ctx, { selection: { startFrame: state.selection.currentFrame } }, "transport");
    };

    buttons.playRangeBtn.onclick = () => {
        const v = dom.video;
        if (!v || !v.duration) return;
        seekToFrame(state.selection.startFrame);
        v.play();
        const checkEnd = () => {
            if (v.paused) { v.removeEventListener("timeupdate", checkEnd); return; }
            const total = state.video.totalFrames;
            if (total <= 1) return;
            // Map endFrame to time through duration, then add one frame's worth
            const endTime = (state.selection.endFrame / (total - 1)) * v.duration;
            const oneFrameTime = v.duration / (total - 1);
            if (v.currentTime >= endTime + oneFrameTime * 0.5) {
                v.pause();
                v.currentTime = endTime;
                v.removeEventListener("timeupdate", checkEnd);
            }
        };
        v.addEventListener("timeupdate", checkEnd);
    };

    buttons.setOutBtn.onclick = () => {
        updateState(ctx, { selection: { endFrame: state.selection.currentFrame } }, "transport");
    };
    buttons.jumpOutBtn.onclick = () => seekToFrame(state.selection.endFrame);

    // Widget callbacks
    wireWidgetCallbacks(ctx);
}

function seekFromMouseEvent(ctx, e) {
    if (!ctx.state.video.totalFrames) return;
    const rect = ctx.dom.scrubber.getBoundingClientRect();
    const percent = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const frame = Math.round(percent * (ctx.state.video.totalFrames - 1));
    updateState(ctx, { selection: { currentFrame: frame } }, "programmatic");
}

function handleMarkerDrag(ctx, e) {
    if (!ctx.state.video.totalFrames) return;
    const rect = ctx.dom.scrubber.getBoundingClientRect();
    const percent = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const frame = Math.round(percent * (ctx.state.video.totalFrames - 1));

    if (ctx.state.ui.draggingMarker === "in") {
        updateState(ctx, { selection: { startFrame: frame } }, "marker-drag");
    } else if (ctx.state.ui.draggingMarker === "out") {
        updateState(ctx, { selection: { endFrame: frame } }, "marker-drag");
    }
}

function wireWidgetCallbacks(ctx) {
    const { widgets } = ctx;

    if (widgets.currentFrame) {
        widgets.currentFrame.callback = (value) => {
            updateState(ctx, { selection: { currentFrame: value } }, "widget");
        };
    }
    if (widgets.startFrame) {
        widgets.startFrame.callback = (value) => {
            updateState(ctx, { selection: { startFrame: value } }, "widget");
        };
    }
    if (widgets.endFrame) {
        widgets.endFrame.callback = (value) => {
            updateState(ctx, { selection: { endFrame: value } }, "widget");
        };
    }
}

// ── Video Loading ──────────────────────────────────────────────────

async function loadVideo(ctx, filename) {
    const { state, dom, widgets } = ctx;

    if (!filename) {
        updateState(ctx, {
            video: { filename: "", fps: 0, totalFrames: 0, width: 0, height: 0, hasAudio: false, loaded: false },
            selection: { currentFrame: 0, startFrame: 0, endFrame: 0 },
        }, "api");
        return;
    }

    // Detect if this is a different video (new load) vs same video (workflow restore)
    const isNewVideo = filename !== state.video.filename;

    try {
        const resp = await fetch(`${API_PREFIX}/query_video?filename=${encodeURIComponent(filename)}`);
        if (!resp.ok) return;
        const meta = await resp.json();
        if (meta.error) return; // PyAV-level failure, silently skip

        const defaultEnd = Math.min(meta.total_frames - 1, Math.round(meta.fps * 2));
        const maxFrame = meta.total_frames - 1;

        if (widgets.currentFrame) widgets.currentFrame.options.max = maxFrame;
        if (widgets.startFrame) widgets.startFrame.options.max = maxFrame;
        if (widgets.endFrame) widgets.endFrame.options.max = maxFrame;

        if (isNewVideo) {
            // New video selected by user — reset to defaults
            updateState(ctx, {
                video: {
                    filename,
                    fps: meta.fps,
                    totalFrames: meta.total_frames,
                    width: meta.width,
                    height: meta.height,
                    hasAudio: meta.has_audio,
                    loaded: true,
                },
                selection: { currentFrame: 0, startFrame: 0, endFrame: defaultEnd },
            }, "api");

            // Sync seek position once the video element is ready
            const syncWhenReady = () => {
                const v = dom.video;
                if (!v || ctx.state.video.filename !== filename) return;
                if (v.readyState >= 1) {
                    updateState(ctx, {
                        selection: { currentFrame: 0, startFrame: 0, endFrame: defaultEnd },
                    }, "programmatic");
                } else {
                    v.addEventListener("loadedmetadata", () => {
                        if (ctx.state.video.filename === filename) {
                            updateState(ctx, {
                                selection: { currentFrame: 0, startFrame: 0, endFrame: defaultEnd },
                            }, "programmatic");
                        }
                    }, { once: true });
                }
            };
            setTimeout(syncWhenReady, 200);

        } else {
            // Workflow restore — set video metadata now, defer reading
            // widget values until the video element has loaded. By the time
            // loadedmetadata fires, ComfyUI will have finished restoring
            // all widget values from the serialized workflow.
            updateState(ctx, {
                video: {
                    filename,
                    fps: meta.fps,
                    totalFrames: meta.total_frames,
                    width: meta.width,
                    height: meta.height,
                    hasAudio: meta.has_audio,
                    loaded: true,
                },
            }, "api");

            const restoreFromWidgets = () => {
                // Use values saved by onConfigure (by name) rather than
                // reading widgets, which may have been clobbered by
                // ComfyUI's positional restore.
                const saved = ctx._savedFrameValues;
                const sf = saved ? clamp(saved.start_frame, 0, maxFrame) : 0;
                const ef = saved ? clamp(saved.end_frame, 0, maxFrame) : defaultEnd;
                const cf = saved ? clamp(saved.current_frame, 0, maxFrame) : 0;
                updateState(ctx, {
                    selection: {
                        currentFrame: cf,
                        startFrame: sf,
                        endFrame: ef === 0 ? defaultEnd : ef,
                    },
                }, "programmatic");
            };

            const syncWhenReady = () => {
                const v = dom.video;
                if (!v || ctx.state.video.filename !== filename) return;
                if (v.readyState >= 1) {
                    restoreFromWidgets();
                } else {
                    v.addEventListener("loadedmetadata", () => {
                        if (ctx.state.video.filename === filename) {
                            restoreFromWidgets();
                        }
                    }, { once: true });
                }
            };
            setTimeout(syncWhenReady, 200);
        }

    } catch (err) {
        console.error("[VisualFrameSelector] Error loading video:", err);
    }
}

// ── Resize Logic ───────────────────────────────────────────────────

function setupResize(ctx) {
    const { node, dom } = ctx;
    const controlsWidget = dom.controlsWidget;
    const controlsContainer = dom.controlsContainer;

    // Cache the measured controls height to avoid feedback loops.
    // computeSize is called during layout — if it reads offsetHeight and
    // returns a new value, ComfyUI re-layouts, changing offsetHeight again.
    // We measure once initially and re-measure only on explicit events.
    let cachedControlsHeight = 130; // sensible default before first measurement

    const measureControls = () => {
        const h = controlsContainer.offsetHeight;
        if (h > 0) cachedControlsHeight = h + 30; // generous padding for widget spacing
    };

    // Measure after first render
    requestAnimationFrame(() => {
        requestAnimationFrame(measureControls);
    });

    // Re-measure when status line appears/disappears (video load/clear)
    ctx._remeasureControls = () => {
        requestAnimationFrame(measureControls);
    };

    // Minimum width: 5 buttons * 28px + 4 gaps * 4px + padding = ~180px
    const MIN_WIDTH = 315;
    const MIN_HEIGHT = 150;

    controlsWidget.computeSize = function () {
        return [MIN_WIDTH, cachedControlsHeight];
    };

    // Apply minimum size immediately
    node.setSize([Math.max(node.size[0], MIN_WIDTH), Math.max(node.size[1], MIN_HEIGHT)]);

    requestAnimationFrame(() => {
        node.minWidth = MIN_WIDTH;
        node.minHeight = MIN_HEIGHT;
        node.lastResizePos = [...node.pos];

        const origOnResize = node.onResize;
        node.onResize = function (size) {
            const xChanged = this.pos[0] !== this.lastResizePos[0];
            const yChanged = this.pos[1] !== this.lastResizePos[1];

            if (size[0] < this.minWidth) {
                size[0] = this.minWidth;
                if (xChanged) this.pos[0] = this.lastResizePos[0];
            }
            if (size[1] < this.minHeight) {
                size[1] = this.minHeight;
                if (yChanged) this.pos[1] = this.lastResizePos[1];
            }

            this.lastResizePos = [...this.pos];
            if (origOnResize) return origOnResize.call(this, size);
            return size;
        };
    });
}

// ── Node Extension ─────────────────────────────────────────────────

app.registerExtension({
    name: "VisualFrameSelector.v2",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            // ── Remove input connectors ──
            const inputsToRemove = ["current_frame", "start_frame", "end_frame"];
            if (node.inputs) {
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    if (inputsToRemove.includes(node.inputs[i].name)) {
                        node.removeInput(i);
                    }
                }
            }

            // ── Create state and abort controller ──
            const state = createState();
            const abortController = new AbortController();

            // ── Build controls DOM ──
            const controlsContainer = document.createElement("div");
            controlsContainer.style.cssText = "width:100%;padding:0 10px 10px 10px;box-sizing:border-box;";

            const { scrubber, progress, inMarker, inLine, outMarker, outLine } = buildScrubber();
            const controlBar = buildControlBar();

            controlsContainer.appendChild(scrubber);
            controlsContainer.appendChild(controlBar.element);

            // ── Add controls as DOM widget ──
            const controlsWidget = node.addDOMWidget("vfs_controls", "customvideo", controlsContainer, {
                serialize: false,
                hideOnZoom: false,
            });

            // ── Find ComfyUI widgets ──
            const widgets = {
                video: node.widgets.find(w => w.name === "video"),
                currentFrame: node.widgets.find(w => w.name === "current_frame"),
                startFrame: node.widgets.find(w => w.name === "start_frame"),
                endFrame: node.widgets.find(w => w.name === "end_frame"),
            };

            // ── Widget ordering ──
            // Desired order:
            //   1. video combo (file selector)
            //   2. video-preview (native preview — may be added after onNodeCreated)
            //   3. vfs_controls (our scrubber + buttons)
            //   4. current_frame, start_frame, end_frame
            //   5. any other widgets we don't recognize
            //
            // The video-preview widget may be appended to
            // node.widgets AFTER onNodeCreated returns, so we re-run ordering
            // whenever we detect the native video element.
            //
            // Because the widget count differs between save (with video-preview)
            // and restore (without it yet), ComfyUI's positional value
            // restoration assigns values to the wrong widgets. We fix this
            // with onConfigure below, which restores frame values by name.
            const frameWidgetNames = new Set(["current_frame", "start_frame", "end_frame"]);

            function reorderWidgets() {
                const combo = [];
                const preview = [];
                const controls = [];
                const frameW = [];
                const other = [];

                for (const w of node.widgets) {
                    if (w.name === "video" || w.name === "upload") {
                        combo.push(w);
                    } else if (w.name === "video-preview") {
                        preview.push(w);
                    } else if (w === controlsWidget) {
                        controls.push(w);
                    } else if (frameWidgetNames.has(w.name)) {
                        frameW.push(w);
                    } else {
                        other.push(w);
                    }
                }

                node.widgets = [...combo, ...preview, ...controls, ...frameW, ...other];
            }

            reorderWidgets();

            // ── Build context (video element added later when found) ──
            const ctx = {
                node,
                state,
                dom: {
                    video: null, // Set when native video element is found
                    scrubber,
                    progress,
                    inMarker,
                    inLine,
                    outMarker,
                    outLine,
                    controlsContainer,
                    controlsWidget,
                    controlBar,
                    statusRow: controlBar.statusRow,
                    statusText: controlBar.statusText,
                    buttons: controlBar.buttons,
                },
                widgets,
                abortController,
            };

            // ── Serialize/restore frame values by name ──
            // ComfyUI serializes widget values by positional index, but the
            // video-preview widget only exists after onNodeCreated returns.
            // This means the widget count (and therefore positions) differ
            // between save and restore, causing values to land on the wrong
            // widgets. We work around this by saving frame values by name
            // in onSerialize and restoring them in onConfigure.
            const origOnSerialize = node.onSerialize;
            node.onSerialize = function (o) {
                if (origOnSerialize) origOnSerialize.call(this, o);
                o.vfs_frame_values = {
                    current_frame: widgets.currentFrame?.value ?? 0,
                    start_frame: widgets.startFrame?.value ?? 0,
                    end_frame: widgets.endFrame?.value ?? 0,
                };
            };

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                if (origOnConfigure) origOnConfigure.call(this, info);
                if (info.vfs_frame_values) {
                    const v = info.vfs_frame_values;
                    ctx._savedFrameValues = v;
                    // Write to widgets immediately to override ComfyUI's
                    // positional restore, which may have put wrong values here.
                    if (widgets.currentFrame) widgets.currentFrame.value = v.current_frame ?? 0;
                    if (widgets.startFrame) widgets.startFrame.value = v.start_frame ?? 0;
                    if (widgets.endFrame) widgets.endFrame.value = v.end_frame ?? 0;
                }
            };

            // ── Wire control events (these work even before video is found) ──
            wireControlEvents(ctx);
            setupResize(ctx);

            // ── Find native video element and wire to it ──
            findNativeVideoElement(node, abortController, (result) => {
                const { video, widget } = result;

                // Detect unsupported renderer: if the widget's video element
                // is not in the DOM, the renderer (e.g. Node 2.0 Vue) is managing
                // its own elements and our controls cannot interact with them.
                if (!video.isConnected) {
                    console.warn("[VisualFrameSelector] Video element is detached from DOM — renderer not supported. Controls disabled.");
                    ctx.dom.controlsContainer.innerHTML = "";
                    const msg = document.createElement("div");
                    msg.style.cssText = "padding:12px;color:#f80;font-size:13px;text-align:center;line-height:1.4;";
                    msg.textContent = "Visual Frame Selector controls are not supported with the current renderer. Please disable Nodes 2.0 in ComfyUI settings, or use the native video controls above.";
                    ctx.dom.controlsContainer.appendChild(msg);
                    node.setSize(node.computeSize());
                    return;
                }

                // Take over the video element
                hookVideoElement(ctx, video);

                // Override video-preview's computeSize to scale with the current
                // node width rather than the video's intrinsic dimensions.
                // Without this, ComfyUI's drag resize uses computeSize() as a floor,
                // locking the node at whatever size it was saved at.
                widget.computeSize = function (width) {
                    const w = width || node.size[0];
                    const aspectRatio = (video.videoHeight && video.videoWidth)
                        ? video.videoHeight / video.videoWidth
                        : 9 / 16;
                    return [w, Math.round(w * aspectRatio)];
                };

                // Re-run widget ordering now that video-preview exists
                reorderWidgets();

                // If video already has metadata loaded, restore state
                if (video.duration && ctx.state.video.loaded) {
                    updateState(ctx, {
                        selection: {
                            currentFrame: ctx.state.selection.currentFrame,
                            startFrame: ctx.state.selection.startFrame,
                            endFrame: ctx.state.selection.endFrame,
                        }
                    }, "programmatic");
                }

                console.log("[VisualFrameSelector] Hooked into native video element");
            });

            // ── Video file selector callback ──
            if (widgets.video) {
                let initialLoadDone = false;
                const origVideoCallback = widgets.video.callback;

                widgets.video.callback = async function (value) {
                    // Call ComfyUI's native callback first — this triggers the
                    // video-preview widget to load the new source via /api/view.
                    if (origVideoCallback) {
                        origVideoCallback.apply(this, arguments);
                    }
                    // Re-find the video element — ComfyUI may have replaced it
                    rehookVideo(ctx);
                    initialLoadDone = true;
                    await loadVideo(ctx, value);
                };

                // Deferred initial load for workflow restore.
                // Set state.video.filename so loadVideo treats this as a
                // restore (isNewVideo=false) and defers reading widget values
                // until loadedmetadata, by which time ComfyUI will have
                // finished restoring all widget values.
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        if (!initialLoadDone && widgets.video.value) {
                            initialLoadDone = true;
                            state.video.filename = widgets.video.value;
                            rehookVideo(ctx);
                            loadVideo(ctx, widgets.video.value);
                        }
                    });
                });
            }

            // ── Cleanup on node removal ──
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                abortController.abort();
                if (origOnRemoved) origOnRemoved.call(this);
            };

            node._vfsCtx = ctx;
            return result;
        };
    },
});
