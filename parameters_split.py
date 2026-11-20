"""Split Image Parameters: an A1111 'parameters' string -> separate outputs to wire into
KSampler, CLIP Text Encode, Empty Latent Image and Load Checkpoint."""

import logging

import comfy.samplers
import folder_paths

from .metadata import short_name, split_parameters, strip_lora_tags as _strip_lora_tags

log = logging.getLogger(__name__)

SAMPLERS = comfy.samplers.KSampler.SAMPLERS
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS


class ComboOutput(list):
    """A combo output type (the options list) that matches any combo input.

    ComfyUI checks a link by comparing the output's list with the input's list. A plain
    list goes stale when e.g. a checkpoint is added without a restart, and the run fails
    with 'received_type mismatch'. The real value is checked by the receiving node.
    """

    def __eq__(self, other):
        return isinstance(other, list) or other == "COMBO"

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

# Used when the text doesn't have a value (logged as a warning).
DEFAULTS = {"seed": 0, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal",
            "denoise": 1.0, "width": 1024, "height": 1024, "guidance": 3.5}


def match_checkpoint(model):
    """The checkpoint file whose name (without folder/extension) matches the Model field."""
    if not model:
        return None
    want = short_name(model).lower()
    for name in folder_paths.get_filename_list("checkpoints"):
        if short_name(name).lower() == want:
            return name
    return None


class SplitImageParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parameters": ("STRING", {"forceInput": True, "tooltip": "The 'parameters' output of Load Image (with Parameters) or Load Image by Index, or any A1111-style generation data text."}),
                "strip_lora_tags": ("BOOLEAN", {"default": True, "tooltip": "Remove <lora:name:strength> tags from the positive prompt. CLIP Text Encode doesn't load LoRAs from tags; they'd only add noise to the prompt. The tags are still on the 'loras' output."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", ComboOutput(SAMPLERS), ComboOutput(SCHEDULERS), "FLOAT",
                    "INT", "INT", "FLOAT", "STRING", ComboOutput(folder_paths.get_filename_list("checkpoints")), "STRING")
    RETURN_NAMES = ("positive", "negative", "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise",
                    "width", "height", "guidance", "model_name", "ckpt_name", "loras")
    OUTPUT_TOOLTIPS = (
        "Positive prompt.",
        "Negative prompt (empty if the image has none).",
        "Seed.",
        "Steps.",
        "CFG scale.",
        "Sampler, as a ComfyUI sampler name (A1111 names like 'DPM++ 2M' are converted).",
        "Scheduler, as a ComfyUI scheduler name ('Karras' -> karras, 'Automatic' -> normal).",
        "Denoise (1.0 unless the image was img2img).",
        "Width of the first pass (an upscale's 'Hires upscale' factor is taken off).",
        "Height of the first pass.",
        "Flux guidance ('Distilled CFG Scale'), for a FluxGuidance node.",
        "Model name as written in the image, e.g. 'myModel_v2'.",
        "Matching checkpoint file for Load Checkpoint. If none matches, the model name is passed on and the loader will say it can't find it.",
        "LoRA tags from the prompt, e.g. '<lora:VeraU:0.8>', one per line.",
    )
    FUNCTION = "split"
    CATEGORY = "image"
    DESCRIPTION = "Splits an image's A1111 generation data (from Load Image (with Parameters)) into separate outputs: prompts, seed, steps, CFG, sampler, scheduler, size, model. Wire them into KSampler / CLIP Text Encode to reuse a previous image's settings."

    def split(self, parameters, strip_lora_tags=True):
        p = split_parameters(parameters, SAMPLERS, SCHEDULERS)
        if not (parameters or "").strip():
            log.warning("[Split Image Parameters] parameters is empty (the image has no A1111 metadata); using defaults.")

        missing = []
        for key, default in DEFAULTS.items():
            if p[key] is None:
                p[key] = default
                if key != "guidance" and key != "denoise":
                    missing.append(key)
        if missing and (parameters or "").strip():
            log.warning("[Split Image Parameters] not in the image's parameters, using defaults: %s", ", ".join(missing))

        model = p["model"] or ""
        ckpt = match_checkpoint(model)
        if ckpt is None and model:
            log.warning("[Split Image Parameters] no checkpoint named '%s' found; ckpt_name gets the raw name.", model)

        positive = _strip_lora_tags(p["positive"]) if strip_lora_tags else p["positive"]
        loras = "\n".join(f"<lora:{n}:{s:g}>" for n, s in p["loras"])
        return (positive, p["negative"], p["seed"], p["steps"], p["cfg"], p["sampler_name"], p["scheduler"],
                p["denoise"], p["width"], p["height"], p["guidance"], model, ckpt or model, loras)


NODE_CLASS_MAPPINGS = {"SplitImageParameters": SplitImageParameters}
NODE_DISPLAY_NAME_MAPPINGS = {"SplitImageParameters": "Split Image Parameters"}
