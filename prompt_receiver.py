"""Receive prompts pushed from another app (e.g. Local AI Prompt Generator) into a node in the open workflow.

The app POSTs {"channel", "text", "queue", "source"} to /prompt_receiver/send. The server can't see the
browser's graph, so it broadcasts the text over the websocket and each open tab answers with an ack
saying how many Prompt Receiver nodes on that channel it updated. The route waits briefly for those
acks, then answers the app with the total. When queue is true, exactly one tab (the one focused most
recently) is told to queue the workflow.
"""

import asyncio
import hashlib
import logging
import time
import uuid

from aiohttp import web

from server import PromptServer

ROUTE = "/prompt_receiver"
DEFAULT_CHANNEL = "main"
ACK_TIMEOUT = 1.0     # seconds the send route waits for open tabs to answer
MAX_TEXT = 200_000    # characters; prompts are a few thousand at most

log = logging.getLogger("PromptReceiver")


def norm_channel(channel):
    channel = (channel or "").strip() if isinstance(channel, str) else ""
    return (channel or DEFAULT_CHANNEL).lower()


_last = {}      # channel -> {"text", "source", "time"}
_pending = {}   # send id -> _Send


class _Send:
    def __init__(self, expected):
        self.expected = expected
        self.acks = []            # (nodes, focused_at)
        self.all_in = asyncio.Event()
        self.decided = asyncio.Event()
        self.queue_ack = None     # index into acks of the tab that should queue


def parse_send(body):
    """Validate the app's JSON. Returns (fields, error)."""
    if not isinstance(body, dict):
        return None, "body must be a JSON object"
    text = body.get("text")
    if not isinstance(text, str):
        return None, "text must be a string"
    if len(text) > MAX_TEXT:
        return None, f"text is longer than {MAX_TEXT} characters"
    channel = body.get("channel", DEFAULT_CHANNEL)
    if channel is not None and not isinstance(channel, str):
        return None, "channel must be a string"
    queue = body.get("queue", False)
    if not isinstance(queue, bool):
        return None, "queue must be true or false"
    source = body.get("source") or ""
    if not isinstance(source, str):
        return None, "source must be a string"
    return {"channel": norm_channel(channel), "text": text, "queue": queue, "source": source[:200]}, None


def _socket_count():
    try:
        return len(PromptServer.instance.sockets)
    except Exception:
        return 1


async def deliver(fields, timeout=ACK_TIMEOUT):
    """Broadcast to open tabs, wait for their acks, pick the tab that queues. Returns (nodes, queued)."""
    _last[fields["channel"]] = {"text": fields["text"], "source": fields["source"], "time": time.time()}
    expected = _socket_count()
    if expected == 0:
        return 0, False
    send_id = uuid.uuid4().hex
    state = _pending[send_id] = _Send(expected)
    try:
        PromptServer.instance.send_sync("prompt_receiver.text", {**fields, "id": send_id})
        try:
            await asyncio.wait_for(state.all_in.wait(), timeout)
        except asyncio.TimeoutError:
            pass  # e.g. an API client that never answers; count what we have
        if fields["queue"]:
            candidates = [i for i, (n, _) in enumerate(state.acks) if n > 0]
            if candidates:
                state.queue_ack = max(candidates, key=lambda i: state.acks[i][1])
        state.decided.set()
        return sum(n for n, _ in state.acks), state.queue_ack is not None
    finally:
        _pending.pop(send_id, None)


async def ack(send_id, nodes, focused_at):
    """Record one tab's answer. Waits for the decision, then says whether this tab should queue."""
    state = _pending.get(send_id)
    if state is None:
        return False  # too late; the app already has its answer
    index = len(state.acks)
    state.acks.append((max(0, int(nodes)), float(focused_at)))
    if len(state.acks) >= state.expected:
        state.all_in.set()
    await state.decided.wait()
    return state.queue_ack == index


# ---------------------------------------------------------------- routes

routes = PromptServer.instance.routes


@routes.post(ROUTE + "/send")
async def _send(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "body must be JSON"}, status=400)
    fields, error = parse_send(body)
    if error:
        return web.json_response({"ok": False, "error": error}, status=400)
    nodes, queued = await deliver(fields)
    log.info("[Prompt Receiver] %d chars from %s on channel '%s' -> %d node(s)%s",
             len(fields["text"]), fields["source"] or "unknown app", fields["channel"], nodes, ", queued" if queued else "")
    return web.json_response({"ok": True, "nodes": nodes, "queued": queued})


@routes.post(ROUTE + "/ack")
async def _ack(request):
    try:
        body = await request.json()
        send_id = str(body["id"])
        nodes = int(body.get("count", 0))
        focused_at = float(body.get("focused_at", 0))
    except Exception:
        return web.json_response({"error": "expected {id, count, focused_at}"}, status=400)
    return web.json_response({"queue": await ack(send_id, nodes, focused_at)})


@routes.get(ROUTE + "/last")
async def _last_text(request):
    channel = norm_channel(request.rel_url.query.get("channel"))
    item = _last.get(channel)
    if item is None:
        return web.json_response({"channel": channel, "text": None})
    return web.json_response({"channel": channel, **item})


# ---------------------------------------------------------------- node

class PromptReceiver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "channel": ("STRING", {"default": DEFAULT_CHANNEL, "tooltip": "The app sends to a channel. Every Prompt Receiver on that channel gets the text. Use different channels (e.g. main, face_detail) to aim at one node."}),
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "The received prompt. You can edit it before you queue."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "output"
    CATEGORY = "text"
    DESCRIPTION = "Shows prompts sent from Local AI Prompt Generator (Send to ComfyUI). Edit if needed, then queue."

    @classmethod
    def IS_CHANGED(s, channel=None, text="", **kwargs):
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

    def output(self, channel, text):
        return (text,)


NODE_CLASS_MAPPINGS = {"PromptReceiver": PromptReceiver}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptReceiver": "Prompt Receiver"}
