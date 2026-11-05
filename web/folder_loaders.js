import { app } from "../../scripts/app.js";

const NODES = ["LoadTextByIndex", "LoadImageByIndex"];
// Changing any of these picks a different file list, so the index goes back to 1.
const SOURCE_WIDGETS = ["folder", "sort", "extensions"];

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function controlMode() {
  try {
    return app.extensionManager?.setting?.get("Comfy.WidgetControlMode")
      ?? app.ui?.settings?.getSettingValue?.("Comfy.WidgetControlMode")
      ?? "after";
  } catch {
    return "after";
  }
}

// Sets index to 1 so the next queued run loads the first file. With "control before generate",
// ComfyUI bumps the index before it queues; resetPending makes that next bump keep 1 instead.
function resetIndex(node) {
  const index = widget(node, "index");
  if (!index) return;
  index.value = 1;
  node.talisarResetPending = true;
  node.setDirtyCanvas?.(true, true);
}

function setStatus(node, text) {
  const status = widget(node, "last loaded");
  if (!status) return;
  status.value = text;
  node.setDirtyCanvas?.(true, true);
}

function rememberSourceValues(node) {
  for (const name of SOURCE_WIDGETS) {
    const w = widget(node, name);
    if (w) w.talisarLast = w.value;
  }
}

app.registerExtension({
  name: "Talisar.FolderLoaders",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;

    // Loading a saved workflow sets widget values without callbacks; remember them so only
    // a real edit afterwards counts as a change. The saved index is kept.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      rememberSourceValues(this);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const used = message?.index?.[0];
      const total = message?.total?.[0];
      if (used == null || total == null) return;
      setStatus(this, `${used} / ${total}: ${message?.name?.[0] ?? ""}`);
      if (widget(this, "at_end")?.value === "stop and reset" && used >= total) {
        resetIndex(this);
      }
    };
  },

  nodeCreated(node) {
    if (!NODES.includes(node.comfyClass)) return;

    // ComfyUI starts every control_after_generate widget on "randomize". These nodes are meant to
    // step through a folder, so new copies start on "increment". Saved workflows keep their own value.
    const control = widget(node, "control_after_generate");
    if (control) {
      control.value = "increment";
      const beforeQueued = control.beforeQueued;
      control.beforeQueued = function () {
        const pending = node.talisarResetPending;
        node.talisarResetPending = false;
        const index = widget(node, "index");
        const kept = index?.value;
        const r = beforeQueued?.apply(this, arguments);
        if (pending && index && controlMode() === "before") index.value = kept;
        return r;
      };
    }

    for (const name of SOURCE_WIDGETS) {
      const w = widget(node, name);
      if (!w) continue;
      w.talisarLast = w.value;
      const callback = w.callback;
      w.callback = function (value) {
        const r = callback?.apply(this, arguments);
        const now = value ?? w.value;
        if (now !== w.talisarLast) {
          w.talisarLast = now;
          resetIndex(node);
          setStatus(node, "");
        }
        return r;
      };
    }

    const button = node.addWidget("button", "reset index to 1", null, () => resetIndex(node));
    button.serialize = false;

    const status = node.addWidget("text", "last loaded", "", () => {});
    status.serialize = false;
    try {
      status.disabled = true;
    } catch {}
  },
});
