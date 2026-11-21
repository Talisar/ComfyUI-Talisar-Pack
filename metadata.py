"""Graph tracing and A1111-style "parameters" string building.

Pure Python (no ComfyUI imports) so it can be tested standalone. Works on the
API-format prompt graph ComfyUI hands to output nodes via the hidden PROMPT
input: {node_id: {"class_type": str, "inputs": {name: value | [src_id, out_idx]}}}.
"""

import json
import os
import re
from collections import deque

# ComfyUI sampler / scheduler names -> A1111 names (what Civitai understands).
SAMPLER_NAMES = {
    "euler": "Euler",
    "euler_cfg_pp": "Euler CFG++",
    "euler_ancestral": "Euler a",
    "euler_ancestral_cfg_pp": "Euler a CFG++",
    "heun": "Heun",
    "heunpp2": "Heun++ 2",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_2s_ancestral_cfg_pp": "DPM++ 2S a CFG++",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_cfg_pp": "DPM++ 2M CFG++",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "DPM++ 3M SDE",
    "dpmpp_3m_sde_gpu": "DPM++ 3M SDE",
    "ddpm": "DDPM",
    "lcm": "LCM",
    "ipndm": "iPNDM",
    "ipndm_v": "iPNDM_v",
    "deis": "DEIS",
    "res_multistep": "Res Multistep",
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": "UniPC",
}

SCHEDULER_NAMES = {
    "normal": "Automatic",
    "karras": "Karras",
    "exponential": "Exponential",
    "sgm_uniform": "SGM Uniform",
    "simple": "Simple",
    "ddim_uniform": "DDIM",
    "beta": "Beta",
    "linear_quadratic": "Linear Quadratic",
    "kl_optimal": "KL Optimal",
}

SAMPLER_CLASSES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}

# Input names that hold the prompt text on text-encoder nodes, best first.
TEXT_KEYS = ("text", "t5xxl", "text_g", "prompt", "positive", "clip_l", "text_l", "string")

# Input names that carry a scalar a "value provider" node passes through.
VALUE_KEYS = ("value", "seed", "noise_seed", "text", "string", "int", "float", "number",
              "source", "STRING", "INT", "FLOAT", "anything", "any", "input")

NON_COND_INPUTS = {"clip", "model", "vae", "image", "images", "mask", "latent", "latent_image",
                   "samples", "control_net", "style_model", "clip_vision_output"}


def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)


class Runtime:
    """Values recorded while the prompt actually executed (see hooks.py)."""

    def __init__(self, prompt_id=None, texts=None, samples=None, list_index=None):
        self.prompt_id = prompt_id
        self.list_index = list_index or 0  # which item of a list run the save node is on
        self.texts = texts or {}      # (prompt_id, node_id, list_index) -> [str, ...]
        self.samples = samples or {}  # (prompt_id, node_id, list_index) -> {seed, steps, ...}

    def _get(self, store, node_id):
        """Value for this list item; like ComfyUI, a shorter list repeats its last item."""
        node_id = str(node_id)
        found = store.get((self.prompt_id, node_id, self.list_index))
        if found is not None:
            return found
        indexes = [k[2] for k in store if k[0] == self.prompt_id and k[1] == node_id]
        below = [i for i in indexes if i <= self.list_index]
        if not indexes:
            return None
        return store[(self.prompt_id, node_id, max(below) if below else min(indexes))]

    def text(self, node_id):
        found = self._get(self.texts, node_id)
        return max(found, key=len) if found else None

    def sample(self, node_id):
        return self._get(self.samples, node_id)


def pixaroma_loras(state):
    """[(name, model_strength)] for the switched-on rows of a LoRA Loader Pixaroma state."""
    try:
        obj = json.loads(state) if isinstance(state, str) else state
        rows = obj.get("loras") or []
    except (ValueError, AttributeError):
        return []
    return [(e["name"], e.get("sm", e.get("strength", 1.0))) for e in rows
            if isinstance(e, dict) and isinstance(e.get("name"), str) and e.get("on", True)
            and not e["name"].startswith("(")]


class GraphReader:
    def __init__(self, prompt, runtime=None):
        self.prompt = {str(k): v for k, v in (prompt or {}).items()}
        self.rt = runtime or Runtime()

    def node(self, node_id):
        return self.prompt.get(str(node_id))

    # ---- scalar resolution -------------------------------------------------

    def resolve(self, value, depth=0):
        """Follow a link to the scalar it represents; None if it can't be known statically."""
        if not is_link(value):
            return value
        if depth > 40:
            return None
        src_id, out_idx = str(value[0]), value[1]
        node = self.node(src_id)
        if node is None:
            return None
        ct = node.get("class_type", "")
        ins = node.get("inputs", {})

        # Text that went through a text encoder at runtime (covers LLM prompt nodes etc.)
        if ct in ("TextGenerate",) or "LLM" in ct or "Ollama" in ct or "Gemini" in ct:
            return None

        if "switch" in ins and ("on_true" in ins or "on_false" in ins):
            s = self.resolve(ins["switch"], depth + 1)
            if s is None:
                return None
            return self.resolve(ins.get("on_true" if s else "on_false"), depth + 1)

        if ct == "StringConcatenate" or ("string_a" in ins and "string_b" in ins):
            a = self.resolve(ins.get("string_a", ""), depth + 1)
            b = self.resolve(ins.get("string_b", ""), depth + 1)
            d = self.resolve(ins.get("delimiter", ""), depth + 1) or ""
            if a is None or b is None:
                return None
            return f"{a}{d}{b}"

        # 2D value providers (width/height sliders etc.)
        if out_idx == 1 and "Yi" in ins:
            return self.resolve(ins["Yi"], depth + 1)
        if out_idx == 0 and "Xi" in ins:
            return self.resolve(ins["Xi"], depth + 1)

        for k in VALUE_KEYS:
            if k in ins:
                return self.resolve(ins[k], depth + 1)
        return None

    # ---- graph walking -----------------------------------------------------

    def bfs(self, start_value, match, follow=None, max_nodes=500):
        """BFS upstream from a link. Returns node ids where match(id, node) is truthy (in order)."""
        if not is_link(start_value):
            return []
        seen, out = set(), []
        q = deque([str(start_value[0])])
        while q and len(seen) < max_nodes:
            nid = q.popleft()
            if nid in seen:
                continue
            seen.add(nid)
            node = self.node(nid)
            if node is None:
                continue
            m = match(nid, node)
            if m:
                out.append(nid)
                if m != "continue":
                    continue
            for name, v in node.get("inputs", {}).items():
                if is_link(v) and (follow is None or follow(name, node)):
                    q.append(str(v[0]))
        return out

    @staticmethod
    def is_sampler(node):
        ct = node.get("class_type", "")
        ins = node.get("inputs", {})
        if ct in SAMPLER_CLASSES:
            return True
        if "positive" in ins and "negative" in ins and ("latent_image" in ins or "latent" in ins):
            return True
        if "guider" in ins and "sigmas" in ins:
            return True
        return ("seed" in ins or "noise_seed" in ins) and "steps" in ins and "model" in ins

    def find_sampler(self, image_link):
        found = self.bfs(image_link, lambda nid, n: self.is_sampler(n))
        return found[0] if found else None

    # ---- extraction --------------------------------------------------------

    def sampler_params(self, sid):
        ins = self.node(sid)["inputs"]
        r = self.resolve
        p = {
            "seed": r(ins.get("seed", ins.get("noise_seed"))),
            "steps": r(ins.get("steps")),
            "cfg": r(ins.get("cfg")),
            "sampler_name": r(ins.get("sampler_name")),
            "scheduler": r(ins.get("scheduler")),
            "denoise": r(ins.get("denoise")),
            "positive": ins.get("positive"),
            "negative": ins.get("negative"),
            "model": ins.get("model"),
        }

        # SamplerCustom(Advanced): parameters live on the noise/guider/sampler/sigmas nodes
        for key in ("noise", "guider", "sampler", "sigmas"):
            link = ins.get(key)
            node = self.node(link[0]) if is_link(link) else None
            if node is None:
                continue
            sub = node.get("inputs", {})
            if key == "noise" and p["seed"] is None:
                p["seed"] = r(sub.get("noise_seed", sub.get("seed")))
            elif key == "guider":
                if p["cfg"] is None:
                    p["cfg"] = r(sub.get("cfg"))
                p["positive"] = p["positive"] or sub.get("positive") or sub.get("conditioning")
                p["negative"] = p["negative"] or sub.get("negative")
                p["model"] = p["model"] or sub.get("model")
            elif key == "sampler" and p["sampler_name"] is None:
                p["sampler_name"] = r(sub.get("sampler_name"))
            elif key == "sigmas":
                for k in ("scheduler", "steps", "denoise"):
                    if p[k] is None:
                        p[k] = r(sub.get(k))
                p["model"] = p["model"] or sub.get("model")

        rt = self.rt.sample(sid)
        if rt:
            for k, v in rt.items():
                if v is not None:
                    p[k] = v

        # Flux-style guidance applied on the conditioning
        p["guidance"] = None
        for gid in self.bfs(p["positive"], lambda nid, n: "guidance" in n.get("inputs", {}) and "conditioning" in n.get("inputs", {})):
            p["guidance"] = r(self.node(gid)["inputs"]["guidance"])
            break
        return p

    def encoder_text(self, nid, node):
        rt = self.rt.text(nid)
        if rt is not None:
            return rt
        ins = node.get("inputs", {})
        for k in TEXT_KEYS:
            if k in ins:
                v = self.resolve(ins[k])
                if isinstance(v, str):
                    return v
        return None

    def conditioning_text(self, cond_link):
        texts = []

        def match(nid, node):
            ct = node.get("class_type", "")
            ins = node.get("inputs", {})
            if "ZeroOut" in ct:
                return True  # zeroed conditioning carries no prompt; stop this branch
            if "clip" in ins and any(k in ins for k in TEXT_KEYS):
                t = self.encoder_text(nid, node)
                if t and t.strip() and t not in texts:
                    texts.append(t)
                return True
            return False

        # Walk upstream link by link (not node by node): nodes such as ControlNetApplyAdvanced,
        # InpaintModelConditioning or Qwen/Wan edit helpers take both "positive" and
        # "negative" and output them in that order, so only the input matching the output
        # we arrived through belongs to this side.
        if not is_link(cond_link):
            return ""
        seen = set()
        q = deque([(str(cond_link[0]), cond_link[1])])
        while q and len(seen) < 500:
            nid, out_idx = q.popleft()
            if (nid, out_idx) in seen:
                continue
            seen.add((nid, out_idx))
            node = self.node(nid)
            if node is None or match(nid, node):
                continue
            for name in self.cond_inputs(node, out_idx):
                v = node["inputs"][name]
                q.append((str(v[0]), v[1]))
        return "\n".join(t.strip() for t in texts)

    @staticmethod
    def cond_inputs(node, out_idx):
        """Input names to follow upstream when tracing conditioning out of output out_idx."""
        ins = node.get("inputs", {})
        links = [k for k, v in ins.items() if is_link(v) and k not in NON_COND_INPUTS]
        if "positive" in ins and "negative" in ins:
            side = {0: "positive", 1: "negative"}.get(out_idx)
            return [side] if side and is_link(ins.get(side)) else []
        cond = [k for k in links if "cond" in k.lower() or k in ("positive", "negative")]
        return cond or links

    def model_chain(self, model_link):
        """Return (model_name, model_folder, [(lora_name, strength), ...]) walking up the model input."""
        loras, model_name, folder = [], None, None
        link, guard = model_link, 0
        while is_link(link) and guard < 100:
            guard += 1
            node = self.node(link[0])
            if node is None:
                break
            ins = node.get("inputs", {})
            r = self.resolve

            node_loras = []
            if "lora_name" in ins:
                name = r(ins["lora_name"])
                strength = r(ins.get("strength_model", ins.get("strength", 1.0)))
                node_loras.append((name, strength))
            if "LoraLoaderState" in ins:  # LoRA Loader Pixaroma keeps its rows in a JSON blob
                node_loras += pixaroma_loras(r(ins["LoraLoaderState"]))
            for k, v in ins.items():
                if isinstance(v, dict) and "lora" in v and v.get("on", True):
                    node_loras.append((v.get("lora"), v.get("strength", 1.0)))
                m = re.fullmatch(r"lora_name_(\d+)", k)
                if m:
                    i = m.group(1)
                    strength = ins.get(f"model_str_{i}", ins.get(f"lora_wt_{i}", 1.0))
                    node_loras.append((r(v), r(strength)))
            loras = node_loras + loras  # walking upstream, so earlier loaders go first

            for key, fld in (("ckpt_name", "checkpoints"), ("unet_name", "diffusion_models"), ("model_name", "diffusion_models")):
                if key in ins:
                    model_name, folder = r(ins[key]), fld
                    break
            if model_name is not None:
                break

            nxt = ins.get("model")
            if not is_link(nxt):
                links = [v for v in ins.values() if is_link(v)]
                nxt = links[0] if len(links) == 1 else None
            link = nxt

        cleaned, seen = [], set()
        for name, strength in loras:
            if not isinstance(name, str) or name in ("None", "") or name in seen:
                continue
            try:
                if float(strength) == 0:
                    continue
            except (TypeError, ValueError):
                pass
            seen.add(name)
            cleaned.append((name, strength))
        return model_name, folder, cleaned


def short_name(path):
    return os.path.splitext(os.path.basename(str(path).replace("\\", "/")))[0]


def a1111_quote(text):
    text = str(text)
    if "," not in text and "\n" not in text and ":" not in text and '"' not in text:
        return text
    return json.dumps(text, ensure_ascii=False)


def fmt_num(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def extract(prompt, image_link, runtime=None):
    """Pull everything Civitai cares about out of the graph."""
    g = GraphReader(prompt, runtime)
    sid = g.find_sampler(image_link)
    info = {"sampler_node": sid, "positive": "", "negative": "", "loras": [], "model": None, "model_folder": None}
    if sid is None:
        return info
    p = g.sampler_params(sid)
    info.update({k: p[k] for k in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "guidance")})
    info["positive"] = g.conditioning_text(p["positive"])
    info["negative"] = g.conditioning_text(p["negative"])
    if info["negative"] and info["negative"].strip() == info["positive"].strip():
        # Negative wired to the same conditioning as the positive (common in CFG 1 edit
        # workflows where the negative is ignored anyway): not a real negative prompt.
        info["negative"] = ""
    info["model"], info["model_folder"], info["loras"] = g.model_chain(p["model"])
    return info


def build_parameters(info, width, height, model_hash=None, lora_hashes=None, lora_tags=True, extra=""):
    lora_hashes = lora_hashes or {}
    positive = info.get("positive") or ""
    if lora_tags and info.get("loras"):
        tags = " ".join(f"<lora:{short_name(n)}:{fmt_num(s)}>" for n, s in info["loras"])
        positive = f"{positive}\n{tags}" if positive else tags

    lines = [positive]
    if info.get("negative"):
        lines.append(f"Negative prompt: {info['negative']}")

    fields = []

    def add(key, value):
        if value is not None and value != "":
            fields.append(f"{key}: {a1111_quote(fmt_num(value))}")

    add("Steps", info.get("steps"))
    sampler = info.get("sampler_name")
    if sampler is not None:
        add("Sampler", SAMPLER_NAMES.get(sampler, sampler))
    scheduler = info.get("scheduler")
    if scheduler is not None:
        add("Schedule type", SCHEDULER_NAMES.get(scheduler, scheduler))
    add("CFG scale", info.get("cfg"))
    add("Distilled CFG Scale", info.get("guidance"))
    add("Seed", info.get("seed"))
    add("Size", f"{width}x{height}")
    add("Model hash", model_hash)
    if info.get("model"):
        add("Model", short_name(info["model"]))
    denoise = info.get("denoise")
    if denoise is not None and float(denoise) < 1.0:
        add("Denoising strength", denoise)

    named = {short_name(n): lora_hashes.get(n) for n, _ in info.get("loras", [])}
    named = {k: v for k, v in named.items() if v}
    if named:
        fields.append('Lora hashes: "' + ", ".join(f"{k}: {v}" for k, v in named.items()) + '"')

    hashes = {}
    if model_hash:
        hashes["model"] = model_hash
    for k, v in named.items():
        hashes[f"lora:{k}"] = v
    if hashes:
        fields.append(f"Hashes: {json.dumps(hashes, ensure_ascii=False)}")

    if extra and extra.strip():
        fields.append(extra.strip().strip(","))
    fields.append("Version: ComfyUI")

    lines.append(", ".join(fields))
    return "\n".join(lines)


# A1111 field, e.g. 'Steps: 20' or 'Hashes: {"model": "abc"}' or 'Lora hashes: "a: 1, b: 2"'.
_FIELD_RE = re.compile(r'\s*([\w][\w \-/()+.]*?):\s*("(?:\\.|[^"\\])*"|\{.*?\}(?=\s*(?:,|$))|[^,]*)\s*(?:,|$)')
_SIZE_RE = re.compile(r"(^|,\s*)Size:\s*(\d+)x(\d+)")

# A1111 field -> info key, for filename keys when the metadata comes from a source image.
_SOURCE_FIELDS = {"Seed": "seed", "Steps": "steps", "CFG scale": "cfg", "Sampler": "sampler_name",
                  "Schedule type": "scheduler", "Denoising strength": "denoise", "Model": "model"}


def _fields_line(lines):
    """Index of the 'Steps: ..., Size: ...' line (the last line with a Size field), or None."""
    for i in range(len(lines) - 1, -1, -1):
        if _SIZE_RE.search(lines[i]):
            return i
    return None


def parse_source_fields(text):
    """Pull seed/steps/cfg/... out of an existing A1111 parameters string."""
    lines = (text or "").split("\n")
    i = _fields_line(lines)
    if i is None:
        return {}
    info = {}
    for key, value in _FIELD_RE.findall(lines[i]):
        name = _SOURCE_FIELDS.get(key.strip())
        if not name or not value:
            continue
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except ValueError:
                value = value.strip('"')
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except (TypeError, ValueError):
                pass
        info[name] = value
    return info


def rescale_parameters(text, width, height, extra=""):
    """Reuse a source image's parameters for an upscaled copy.

    Size becomes the new size; if it is bigger than the source, 'Hires upscale: <factor>'
    is added (multiplied into an existing one). Extra fields go before 'Version:'.
    """
    lines = text.strip("\n").split("\n")
    i = _fields_line(lines)
    if i is None:
        lines.append(f"Size: {width}x{height}")
        i = len(lines) - 1
    line = lines[i]
    m = _SIZE_RE.search(line)
    old_w = int(m.group(2))
    line = line[:m.start()] + f"{m.group(1)}Size: {width}x{height}" + line[m.end():]

    factor = width / old_w if old_w else 1.0
    if factor > 1.005:
        hires = re.search(r"(^|,\s*)Hires upscale:\s*([\d.]+)", line)
        if hires:
            try:
                factor *= float(hires.group(2))
            except ValueError:
                pass
            line = line[:hires.start()] + f"{hires.group(1)}Hires upscale: {round(factor, 2):g}" + line[hires.end():]
        else:
            size_end = _SIZE_RE.search(line).end()
            line = line[:size_end] + f", Hires upscale: {round(factor, 2):g}" + line[size_end:]

    extra = (extra or "").strip().strip(",").strip()
    if extra and extra not in line:
        version = re.search(r",\s*Version:", line)
        if version:
            line = line[:version.start()] + f", {extra}" + line[version.start():]
        else:
            line = f"{line}, {extra}"

    lines[i] = line
    return "\n".join(lines)


# ---- splitting a parameters string back into generation settings -------------

_STEPS_RE = re.compile(r"(^|,\s*)Steps:\s*\d")
_LORA_TAG_RE = re.compile(r"\s*<(?:lora|lyco):[^>]*>")

# A1111 names -> ComfyUI names. Built from the maps above; the first ComfyUI name wins
# for A1111 names shared by several (e.g. "DPM++ SDE" -> dpmpp_sde, not dpmpp_sde_gpu).
SAMPLER_FROM_A1111 = {}
for _c, _a in SAMPLER_NAMES.items():
    SAMPLER_FROM_A1111.setdefault(_a.lower(), _c)
SCHEDULER_FROM_A1111 = {}
for _c, _a in SCHEDULER_NAMES.items():
    SCHEDULER_FROM_A1111.setdefault(_a.lower(), _c)
SCHEDULER_FROM_A1111.update({"uniform": "sgm_uniform", "align your steps": "normal"})


def _settings_line(lines):
    """Index of the 'Steps: ..., Size: ...' line, or None."""
    i = _fields_line(lines)
    if i is not None:
        return i
    for j in range(len(lines) - 1, -1, -1):
        if _STEPS_RE.search(lines[j]):
            return j
    return None


def _field_value(value):
    if value.startswith('"'):
        try:
            return json.loads(value)
        except ValueError:
            return value.strip('"')
    return value.strip()


def to_comfy_sampler(name, samplers=None):
    """A1111 or ComfyUI sampler name -> (ComfyUI sampler, scheduler implied by the name or None).

    Old A1111 images put the scheduler in the sampler name ("DPM++ 2M Karras").
    """
    if not name:
        return None, None
    name = str(name).strip()
    if samplers and name in samplers:
        return name, None
    low = name.lower()
    if low in SAMPLER_FROM_A1111:
        return SAMPLER_FROM_A1111[low], None
    for suffix, sched in ((" karras", "karras"), (" exponential", "exponential")):
        if low.endswith(suffix) and low[: -len(suffix)] in SAMPLER_FROM_A1111:
            return SAMPLER_FROM_A1111[low[: -len(suffix)]], sched
    guess = low.replace("++", "pp").replace(" ", "_").replace("-", "_")
    if guess == "euler_a":
        guess = "euler_ancestral"
    if not samplers or guess in samplers:
        return guess, None
    return None, None


def to_comfy_scheduler(name, schedulers=None):
    if not name:
        return None
    name = str(name).strip()
    if schedulers and name in schedulers:
        return name
    low = name.lower()
    if low in SCHEDULER_FROM_A1111:
        return SCHEDULER_FROM_A1111[low]
    guess = low.replace(" ", "_")
    return guess if not schedulers or guess in schedulers else None


def split_parameters(text, samplers=None, schedulers=None):
    """Turn an A1111 'parameters' string back into ComfyUI settings.

    Returns a dict with positive, negative, loras ([(name, strength)]), seed, steps, cfg,
    sampler_name, scheduler, width, height, denoise, guidance, model, fields (all raw
    fields). A key is None when the text doesn't have it.
    """
    out = {"positive": "", "negative": "", "loras": [], "seed": None, "steps": None, "cfg": None,
           "sampler_name": None, "scheduler": None, "width": None, "height": None,
           "denoise": None, "guidance": None, "model": None, "fields": {}}
    lines = (text or "").strip("\n").split("\n")
    if not any(line.strip() for line in lines):
        return out

    i = _settings_line(lines)
    prompt_lines = lines if i is None else lines[:i]
    fields = {}
    if i is not None:
        for key, value in _FIELD_RE.findall(lines[i]):
            if key.strip() and value:
                fields[key.strip()] = _field_value(value)
    out["fields"] = fields

    neg = next((j for j, line in enumerate(prompt_lines) if line.startswith("Negative prompt:")), None)
    pos_lines = prompt_lines if neg is None else prompt_lines[:neg]
    positive = "\n".join(pos_lines).strip()
    if neg is not None:
        neg_lines = [prompt_lines[neg][len("Negative prompt:"):]] + prompt_lines[neg + 1:]
        out["negative"] = "\n".join(neg_lines).strip()

    for tag in re.findall(r"<(?:lora|lyco):([^>]*)>", positive):
        parts = tag.split(":")
        strength = 1.0
        if len(parts) > 1:
            try:
                strength = float(parts[1])
            except ValueError:
                pass
        out["loras"].append((parts[0].strip(), strength))
    out["positive"] = positive

    def num(key, cast):
        try:
            return cast(float(fields[key])) if cast is int else cast(fields[key])
        except (KeyError, TypeError, ValueError):
            return None

    out["seed"] = num("Seed", int)
    out["steps"] = num("Steps", int)
    out["cfg"] = num("CFG scale", float)
    out["denoise"] = num("Denoising strength", float)
    out["guidance"] = num("Distilled CFG Scale", float)
    out["model"] = fields.get("Model") or None

    sampler, implied = to_comfy_sampler(fields.get("Sampler"), samplers)
    out["sampler_name"] = sampler
    out["scheduler"] = to_comfy_scheduler(fields.get("Schedule type"), schedulers) or implied

    size = re.match(r"\s*(\d+)\s*x\s*(\d+)", str(fields.get("Size", "")))
    if size:
        w, h = int(size.group(1)), int(size.group(2))
        hires = num("Hires upscale", float)
        # Our save node writes the saved (upscaled) size plus the factor; A1111 writes the
        # first-pass size. Give the first-pass size either way.
        if hires and hires > 1 and str(fields.get("Version", "")).startswith("ComfyUI"):
            w, h = _round8(w / hires), _round8(h / hires)
        out["width"], out["height"] = w, h
    return out


def _round8(v):
    return max(8, int(round(v / 8.0)) * 8)


def strip_lora_tags(text):
    return _LORA_TAG_RE.sub("", text or "").strip()
