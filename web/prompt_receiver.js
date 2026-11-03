import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ROUTE = "/prompt_receiver";
const NODE = "PromptReceiver";
const FLASH_MS = 1500;

const CSS = `
.pr-row { display: flex; gap: 4px; align-items: center; font: 12px sans-serif; color: var(--input-text, #ddd); }
.pr-status { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; opacity: .8; }
.pr-row button { background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #555); border-radius: 4px; padding: 2px 6px; font: inherit; }
.pr-row button:hover { filter: brightness(1.3); cursor: pointer; }
`;

const channelOf = (v) => (typeof v === "string" && v.trim() ? v.trim() : "main").toLowerCase();
const widget = (node, name) => node.widgets?.find((w) => w.name === name);

// Only one tab should queue: the server picks the tab that was focused most recently.
let focusedAt = document.hasFocus() ? Date.now() : 0;
window.addEventListener("focus", () => { focusedAt = Date.now(); });

/** Every Prompt Receiver in the open workflow, including ones inside subgraphs. */
function receivers() {
  const found = new Set();
  const seen = new Set();
  const walk = (graph) => {
    if (!graph || seen.has(graph)) return;
    seen.add(graph);
    for (const n of graph._nodes ?? graph.nodes ?? []) {
      if (n.comfyClass === NODE || n.type === NODE) found.add(n);
      if (n.subgraph) walk(n.subgraph);
    }
    for (const sg of graph.subgraphs?.values?.() ?? []) walk(sg);
  };
  walk(app.graph);
  return [...found];
}

function setText(node, text, source) {
  const w = widget(node, "text");
  if (!w) return false;
  w.value = text;
  w.callback?.(text);
  const when = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  node.prStatus?.(`Received ${when}${source ? ` from ${source}` : ""}`);
  node.prFlashUntil = Date.now() + FLASH_MS;
  setTimeout(() => app.graph?.setDirtyCanvas(true, true), FLASH_MS + 50);
  return true;
}

async function onText({ id, channel, text, source, queue }) {
  let count = 0;
  for (const node of receivers()) {
    if (channelOf(widget(node, "channel")?.value) === channel && setText(node, text, source)) count++;
  }
  app.graph?.setDirtyCanvas(true, true);
  let shouldQueue = false;
  try {
    const resp = await api.fetchApi(ROUTE + "/ack", {
      method: "POST",
      body: JSON.stringify({ id, count, focused_at: focusedAt }),
    });
    shouldQueue = (await resp.json()).queue === true;
  } catch {}
  if (queue && shouldQueue && count > 0) await app.queuePrompt(0);
}

function addStatus(node) {
  const status = Object.assign(document.createElement("span"), { className: "pr-status", textContent: "Waiting for a prompt…" });
  const pull = Object.assign(document.createElement("button"), {
    textContent: "Pull last",
    title: "Load the last prompt sent to this channel since ComfyUI started (e.g. one sent while this tab was closed).",
  });
  pull.addEventListener("pointerdown", (e) => e.stopPropagation());
  const row = document.createElement("div");
  row.className = "pr-row";
  row.append(status, pull);

  node.prStatus = (msg) => { status.textContent = msg; status.title = msg; };
  pull.onclick = async () => {
    const channel = channelOf(widget(node, "channel")?.value);
    try {
      const data = await (await api.fetchApi(`${ROUTE}/last?channel=${encodeURIComponent(channel)}`)).json();
      if (typeof data.text !== "string") return node.prStatus(`Nothing sent to '${channel}' yet`);
      setText(node, data.text, data.source);
      app.graph?.setDirtyCanvas(true, true);
    } catch (e) {
      node.prStatus(`Couldn't reach ComfyUI: ${e}`);
    }
  };

  node.addDOMWidget("receiver_status", "prompt_receiver_status", row, {
    serialize: false,
    getValue: () => "",
    setValue: () => {},
    getMinHeight: () => 26,
    getMaxHeight: () => 26,
  });
}

app.registerExtension({
  name: "Talisar.PromptReceiver",

  setup() {
    document.head.append(Object.assign(document.createElement("style"), { textContent: CSS }));
    api.addEventListener("prompt_receiver.text", (e) => onText(e.detail));
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      addStatus(this);
      return r;
    };
    // Brief green outline when a new prompt arrives (drawn, not saved with the workflow).
    const draw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = draw?.apply(this, arguments);
      if (this.prFlashUntil && Date.now() < this.prFlashUntil && !this.flags?.collapsed) {
        const title = globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
        ctx.save();
        ctx.strokeStyle = "#3c3";
        ctx.lineWidth = 3;
        ctx.globalAlpha = Math.min(1, (this.prFlashUntil - Date.now()) / 500);
        ctx.beginPath();
        const box = [-3, -title - 3, this.size[0] + 6, this.size[1] + title + 6];
        if (ctx.roundRect) ctx.roundRect(...box, 8);
        else ctx.rect(...box);
        ctx.stroke();
        ctx.restore();
        this.setDirtyCanvas?.(true, false);
      }
      return r;
    };
  },
});
