# Qwen21 No-Offset Editing (ComfyUI Custom Nodes)

> **About this project**: a standalone ComfyUI custom-node package, improved from [xingyuezhiyuan/Comfyui-txtnode](https://github.com/xingyuezhiyuan/Comfyui-txtnode). It keeps only the three nodes needed by the no-offset editing pipeline, and adds the **Auto Align to Reference** node together with a **pixel-perfect (1:1) no-offset editing pipeline**.

**English** | [简体中文](README_CN.md)

---

## Why "No-Offset" Editing

When fully-regenerative editing models such as Qwen Image Edit edit an image, the output content is **globally shifted by 1–2 latent cells** (≈16–32 px at 1024 resolution) relative to the original. This is inherent model behavior (attention layers) and cannot be prevented at the input stage. Consequences:

- Cropping the output back to the original size → misaligned content;
- Correcting the shift with edge-replication padding → visible "copy streaks" along image borders.

This plugin provides a mechanistic fix — **align-then-crop (crop mode)**: instead of shifting the image, it locates the shift via FFT cross-correlation and directly slices a window that is **pixel-aligned** with the original content. Every pixel inside the window is real decoded content, eliminating copy streaks and misalignment at the root.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/u12138ujkigh/Qwen21-NoOffset-Editing.git
```

Restart ComfyUI. Nodes appear under the `txtnode` category.

> ⚠️ **Compatibility warning**: node IDs are identical to Comfyui-txtnode (`ResizeAndPadNode` / `RemovePadFromImageNode` / `AutoAlignToReferenceNode`). **Do not install both packages** (or any txtnode variant) at the same time — node definitions will conflict.

## Nodes

### 1. Resize and Pad Image (enhanced)

Scales the image proportionally and pads it onto a canvas, outputting `output_image` plus padding metadata `image_info` (6-tuple: left/top/right/bottom padding, **canvas width**, **canvas height**; the old single-side 5-tuple is still accepted downstream).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_size` | INT | 1024 | Side / long-side cap; **0 = auto** (1:1 mode: long side rounded up to a multiple of 32; aspect modes: no rescale) |
| `resolution_multiple` | INT | 8 | Snaps target_size to a multiple of this; 0 = disable |
| `upscale_method` | COMBO | lanczos | lanczos / bicubic / area / nearest |
| `resize_and_pad` | BOOLEAN | True | Off = bypass (pass-through) |
| `background_color` | STRING | #000000 | Pad color; **`transparent`** = transparent padding (RGBA output) |
| `scale_mode` | COMBO | fill_target | `fill_target` = scale to fit exactly; **`no_upscale`** = shrink only, never enlarge |
| **`pad_mode`** | COMBO | 1:1 (square) | **Canvas mode**: `1:1（方形画布）` / `16 的倍数` / `32 的倍数` |

**Canvas modes (new)**

| Mode | Canvas | Padding side | Notes |
|---|---|---|---|
| `1:1（方形画布）` | Square, side = `target_size` or long side rounded up to a multiple of 32 | centered | **Legacy behaviour**, pixel-identical to before |
| `16 的倍数` | Aspect preserved, each axis rounded up to a multiple of 16 | **right/bottom only** | Smallest area; see the hard constraint below |
| `32 的倍数` | Same, multiple of 32 | **right/bottom only** | **Recommended**: matches the core encoder's rounding |

Benefits of the two aspect modes:
1. **Faster** — no more squaring; a portrait image saves 30–45% of latent tokens (e.g. 1200×1746 goes from 1760² to 1216×1760);
2. **Content origin lands on the latent grid** — padding only right/bottom keeps content pinned at `(0,0)`, whereas centered padding offsets the origin by up to half a cell (e.g. 1200×1746 in a 1760 square gives `pad_left=280`, and 280 % 16 = 8).

> ⚠️ **Hard constraint: canvas width/height must be a multiple of 32.** With `resolution = 0`, the core
> `TextEncodeQwenImage21` computes the latent size as `round(dim / 32) * 32`, so a canvas width of 1200
> becomes 1216 — mismatching the canvas, and "Auto Align to Reference" then reports inconsistent sizes.
> Therefore the **`16 的倍数` mode is unusable with the core encoder** (the node prints an explicit warning),
> unless you switch to a 16-aligned text encoder. The `32 的倍数` mode has no such problem and already
> captures 30.9% of the speed-up (the 16 mode would be 31.8% — only 0.9 points more).

Background: measured on `qwen_image_2.1_vae_bf16`, `downscale_ratio = 16` and `latent_channels = 64`,
i.e. **16 px = one latent cell**; a dimension that is not a multiple of 16 is **truncated** by the VAE
(`1200×1746 → latent 75×109 → decodes to 1200×1744`, losing 2 rows at the bottom).

**Enhancements over the original plugin** (key to the no-offset pipeline):

- `scale_mode = no_upscale` + `target_size = 0`: **zero-resample** — content enters the canvas pixel-for-pixel, enabling pixel-exact comparison after editing;
- `background_color = transparent`: transparent padding, so the model does not treat black bars as "image content" and paint artifacts into them;
- `pad_mode`: selectable canvas shape / multiple, as above.

### 2. Remove Pad from Image

Crops the padding area according to `image_info` metadata to restore the original aspect ratio. A general-purpose companion node; the recommended no-offset pipeline (crop mode) does **not** need it.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_image` | IMAGE | - | Padded-domain image |
| `image_info` | IMAGE_INFO | - | From "Resize and Pad Image" |
| `remove_pad` | BOOLEAN | True | Off = bypass |
| `latent_scale` | FLOAT | 0.0 | Optional, latent-space scale factor (for exact matching) |

### 3. Auto Align to Reference (new in this project)

Estimates the integer-pixel shift of the edited output relative to the reference original within `max_shift`, using **FFT circular cross-correlation**, then corrects it in one of two ways:

- **Shift mode** (`crop_to_reference = False`, legacy behavior): translates the whole image back into alignment; out-of-bounds areas are filled by replicating the nearest edge pixels (small shifts produce copy streaks at borders);
- **Crop mode** (`crop_to_reference = True`, **recommended**): no translation — directly slices the window corresponding to the aligned reference content area. Every pixel inside is real content, **zero replicated fill**. Place this node in the padded domain: connect `reference_image` to the output of "Resize and Pad Image" and `image_info` to its metadata.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | - | Edited output (same domain/size as reference) |
| `reference_image` | IMAGE | - | Reference original image |
| `max_shift` | INT | 64 | Maximum search shift (px), 0–512 |
| `crop_to_reference` | BOOLEAN | False | **Crop-mode switch** |
| `min_confidence` | FLOAT | 0.0 | Overlap-region NCC confidence threshold (0–1); below it the shift is considered unreliable and the image is left untouched; 0 = always correct |
| `image_info` | IMAGE_INFO (optional) | - | Content-area geometry (required by crop mode, from "Resize and Pad Image") |

**Outputs**: `image` (corrected), `shift_dx` / `shift_dy` (detected shift in px).

## The No-Offset Editing Pipeline (recommended usage)

```
LoadImage ─▶ Resize and Pad ─▶ VAE Encode ─▶ KSampler ─▶ VAE Decode ─▶ Auto Align to Reference ─▶ Save
                  │ (no_upscale                                       │        (crop_to_reference=True)
                  │  +transparent)                                    │
                  ├──────────── image_info ───────────────────────────┤
                  └── output_image ──▶ (reference_image) ─────────────┘
```

1. **Zero-resample padding**: Resize and Pad with `scale_mode = no_upscale`, `target_size = 0`, `background_color = transparent` — content enters the canvas 1:1, padding transparent. Using `pad_mode = 32 的倍数` (aspect preserved, padding only right/bottom) costs 30–45% fewer latent tokens than `1:1（方形画布）` and keeps the content origin exactly on the latent grid;
2. **Edit normally**: VAE Encode → KSampler (Qwen Image Edit) → VAE Decode. The model works in the padded domain and will repaint the padded area too (normal — crop mode removes it);
3. **Align-then-crop**: Auto Align to Reference with `crop_to_reference = True`, `reference_image` from the padded original, `image_info` from its metadata — FFT finds the 16–32 px shift introduced by the model and slices the window back to a 1:1 match with the original;
4. **Output**: size = original content size, pixels 1:1, no copy streaks, no misalignment.

> 💡 **Large-image protection**: for very large inputs, insert a "Scale Image to Suitable Size" node (e.g. 1.5 MP) before "Resize and Pad Image" to lower the editing resolution — it only affects the editing domain; the crop-mode output size is still governed by `image_info`, so alignment is unaffected.

**Ready-to-use workflow**: [`workflow/Qwen21_无偏移编辑_像素1比1.json`](workflow/Qwen21_无偏移编辑_像素1比1.json) — drag it into ComfyUI (requires a local Qwen Image 2.1 model).

## Credits & Attribution

- This project is an improved distribution of [xingyuezhiyuan/Comfyui-txtnode](https://github.com/xingyuezhiyuan/Comfyui-txtnode):
  - **From the original plugin**: the base design of the "Resize and Pad Image" and "Remove Pad from Image" nodes (proportional scaling, center padding, `image_info`-based cropping);
  - **Added/enhanced by this project**: the "Auto Align to Reference" node (entirely new); the `no_upscale` zero-resample scale mode, `transparent` padding and `target_size = 0` auto size of "Resize and Pad Image"; plus the complete no-offset editing pipeline and the ready-to-use workflow.
- Thanks to the original author [@xingyuezhiyuan](https://github.com/xingyuezhiyuan).
- The original project ships no open-source license; this repository is published for study and reference only, and all rights of the original work remain with its author. If you are the original author and wish to change how this repository is published, please open an issue.
