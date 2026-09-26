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
| **`pad_mode`** | COMBO | 1:1 (square) | **Canvas mode**: `1:1（方形画布）` / `16 的倍数` / `32 的倍数` / `32 的倍数（四周留边）` / `1:1（四周留边）` / `32 的倍数（四边自定义）` |
| **`edge_margin`** | INT | 64 | Margin per side, used by the two `四周留边` modes (0–512, **step 8**). **The left/top room is exactly the value you set** — the content is placed flush against the left/top edge and the ceil-to-32 slack all lands on the right/bottom, so this value is also the maximum drift the aligner can correct on the left/top axes |
| **`margin_left` / `margin_top` / `margin_right` / `margin_bottom`** | INT | 0 | Per-side margins (px, step 8), used by `32 的倍数（四边自定义）` only. **Pad only the sides your model actually drifts towards** — the same canvas area buys a larger margin on that side |

**Canvas modes**

| Mode | Canvas | Padding side | Notes |
|---|---|---|---|
| `1:1（方形画布）` | Square, side = `target_size` or long side rounded up to a multiple of 32 | centered | **Legacy behaviour**, pixel-identical to before. ⚠️ Cannot correct drift on the axis where the content side equals the canvas side — horizontal for a typical landscape, **vertical for a typical portrait** (983×1280 → 1280×1280, `pad_top = 0`). A 983×1280 portrait gets 148 px of horizontal room yet still comes out **11 px high** — see below |
| `16 的倍数` | Aspect preserved, each axis rounded up to a multiple of 16 | **right/bottom only** | Smallest area; unusable with the core encoder, see the hard constraint below |
| `32 的倍数` | Same, multiple of 32 | **right/bottom only** | Smaller area; **zero left/top room**, so left/up drift cannot be corrected |
| `32 的倍数（四周留边）` | Aspect preserved, `edge_margin` on all four sides, then rounded up to a multiple of 32 | **left/top = your value, right/bottom = your value + slack** | **Recommended for the no-offset pipeline**: content origin stays on the latent grid *and* every direction has room to correct drift |
| `1:1（四周留边）` | Square, side = long side rounded up to a multiple of 32 **plus 2 × `edge_margin`** | **all four sides ≥ your value** | Same drift protection while keeping a (larger) square canvas: 983×1280 → 1344² vs 1056×1344 for the aspect mode |
| `32 的倍数（四边自定义）` | Aspect preserved, canvas = `ceil32(w + left + right) × ceil32(h + top + bottom)`, content placed at `(left, top)` | **left/top = your value; right/bottom = your value + slack** | **Cheapest**: specify each side separately and pad only where the model actually drifts. E.g. 964×1280 with left 64 / top 32 / right 0 / bottom 0 → canvas 1056×**1312**, which is **2.4% smaller** than "32 on all four sides" (1056×1344) yet gives **64** px of left room; measured zero-shift NCC = **1.0000** |

> 🔎 **The margin granularity is 8 px, not a multiple of 32.**
> The implementation is literally the two steps you'd expect — "grow the content by the margin you
> typed, then round the canvas up to a multiple of 32" — so the canvas is always a multiple of 32
> (required by the core encoder) while **the margin itself is free**
> (older builds silently snapped 40 down to 32, throwing away 8 px of room).
>
> Evidence: a real-VAE encode→decode round trip (no sampling) over margins 1–64 at every phase
> shows a content drift of exactly **(0,0)** and a constant content MAE of 0.74/255
> (`_gate/verify_margin_granularity.py`; full-resolution re-run over the multiples of 8:
> NCC 0.99986, drift all zero). So an origin that is *not* on the 16/32 grid does **not** make the
> VAE drift by itself. The latent spatial compression is 16, i.e. 8 px = half a latent cell.

> ✅ **Use `32 的倍数（四周留边）` with a margin of 64.** The margin *is* the maximum drift the aligner
> can correct on that axis: **the left/top room equals exactly the value you set**, so a margin of 32 can
> only correct 32 px (the right/bottom sides get a little extra from the ceil-to-32 slack — never rely on
> it). A fine-tuned Qwen 2.1 shifting **42 px left** on a 964×1280 image leaves a 10 px gap at a margin
> of 32 (measured residual **10 px left**) and drops to **0** at 64. Use 96 for models that drift further.
> If you specifically need a square canvas, use `1:1（四周留边）` with the same margin.
> **To save area**, use `32 的倍数（四边自定义）` when the drift direction is fixed (here: always left):
> pad only left 64 / top 32 and leave right/bottom at 0 — the canvas becomes 1056×**1312**, **2.4% smaller**
> than "32 on all four sides" (1056×1344), yet the left room is 64; measured zero-shift NCC = **1.0000**.
> The trade-off is zero correction room on the right/bottom — pad those too if your model drifts that way.
> The bundled workflow already ships with `32 的倍数（四周留边）` + 64.

Benefits of the aspect modes:
1. **Faster** — no more squaring; a landscape image saves ~37% of latent tokens (2752×1536 goes from 2752² to 2880×1664), a portrait one 30–45% (1200×1746 goes from 1760² to 1216×1760);
2. **Content origin is unambiguous** — with `right/bottom only` the content stays pinned at `(0,0)`; the `四周留边` modes place it at `(margin, margin)`. Margins are snapped to 8 px, and a real-VAE round trip shows no measurable drift at any phase (see above), so an off-grid origin is fine.

> ⚠️ **Why `32 的倍数（四周留边）` is the recommended canvas mode.** Crop mode corrects drift by
> starting its window at `origin + shift`. With right/bottom-only padding `origin = (0,0)`, so a left/up
> drift makes that start negative, it gets clamped to 0 and **the correction is silently swallowed** —
> the output stays shifted by exactly the model's own drift. The legacy `1:1` mode has the same problem
> on the horizontal axis whenever the content width equals the canvas width (typical landscape:
> `pad_left = pad_right = 0`, so `x0` can only ever be 0, whatever the drift direction). A fine-tuned
> model with an inherent left drift therefore comes out *consistently shifted left*, while a stock model
> (drift ≈ 0) looks fine. Set `edge_margin ≥ max drift` — the margin is a *capacity*, not a flag:
> having room on a side does not mean having *enough* room.
> The Auto Align node now **prints an explicit warning naming the shortfall and the exact value to set**
> whenever it detects clamping.
>
> 📏 **Measured run A** (983×1280 portrait, fine-tuned Qwen 2.1): the model shifts the content
> **24 px left and 11 px up**. With `32 的倍数` the canvas is 992×1280 ⇒ room L/T/R/B = 0/0/9/0, so both
> corrections are clamped away and the output keeps the full shift — the measured residual is exactly
> **−24 / −11**. With `edge_margin = 32` (canvas 1056×1344) the residual drops to **0 / 0**, and with
> `1:1（四周留边）` 32 (canvas 1344×1344) also **0 / 0**. Note that `max_shift = 64` was never the limit:
> 24 px sits well inside it. The missing *margin*, not the search range, was the problem.
>
> 📏 **Measured run B — margin present but too small** (964×1280, same fine-tuned model): it shifts the
> content **42 px left**. With `32 的倍数（四周留边）` and `edge_margin = 32` the canvas is 1056×1344 with
> room L/T/R/B = **32/32/60/32** — the left room is exactly the 32 you typed. 42 > 32, so a **10 px gap**
> cannot be taken from outside the canvas and the output comes out **10 px left** (measured residual
> `10`, matching 42 − 32 exactly). At `edge_margin = 64` (canvas 1120×1408, left room 64) the residual is
> **0**. Drift varies with model *and* content (24 → 42 here), so leave headroom rather than sizing the
> margin to the drift you last saw.
>
> ⚠️ **"Round the long side up to a multiple of 32, then square it" is *not* the same as leaving a
> margin.** When the long side is already a multiple of 32 (e.g. 768 for a 768×715 landscape) those two
> steps reproduce the legacy `1:1` canvas **pixel for pixel**, with zero extra room — the content ends up
> flush against the canvas edge and no drift can be corrected in that axis. Room only appears when you
> add `2 × edge_margin` on top, which is what the two `四周留边` modes do.

> ⚠️ **Hard constraint: canvas width/height must be a multiple of 32.** With `resolution = 0`, the core
> `TextEncodeQwenImage21` computes the latent size as `round(dim / 32) * 32`, so a canvas width of 1200
> becomes 1216 — mismatching the canvas, and "Auto Align to Reference" then reports inconsistent sizes.
> Therefore the **`16 的倍数` mode is unusable with the core encoder** (the node prints an explicit warning),
> unless you switch to a 16-aligned text encoder. The `32 的倍数` and `32 的倍数（四周留边）` modes have no
> such problem and already capture ~31% of the speed-up (the 16 mode would be 31.8% — only 0.9 points more).

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
- **Crop mode** (`crop_to_reference = True`, **recommended**): no translation — directly slices the window corresponding to the aligned reference content area. Every pixel inside is real content, **zero replicated fill**. Place this node in the padded domain: connect `reference_image` to the output of "Resize and Pad Image" and `image_info` to its metadata. The window starts at `origin + shift`, so that side of the canvas must have at least `|shift|` px of padding — otherwise the start falls outside the canvas, is clamped to 0 and **the correction is silently lost**. When that happens the node now prints a warning naming the affected direction(s) and the `edge_margin` to set, instead of failing quietly.

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

1. **Zero-resample padding**: Resize and Pad with `scale_mode = no_upscale`, `target_size = 0`, `background_color = transparent` — content enters the canvas 1:1, padding transparent. **Use a 64 px canvas margin**: `pad_mode = 32 的倍数（四周留边）` with `edge_margin = 64` — or `1:1（四周留边）` when you need a square canvas: these keep the content origin on the latent grid **and** leave room on all four sides for the aligner to crop into (**left/top room = the value you set**, so 32 can only correct 32 px), while still saving 30–45% of the latent tokens of the legacy square canvas. Do **not** use `32 的倍数` (right/bottom only) or `1:1（方形画布）` for drift-prone models — both have zero room on at least one axis, so the correction gets clamped away and the output stays shifted;
2. **Edit normally**: VAE Encode → KSampler (Qwen Image Edit) → VAE Decode. The model works in the padded domain and will repaint the padded area too (normal — crop mode removes it);
3. **Align-then-crop**: Auto Align to Reference with `crop_to_reference = True`, `reference_image` from the padded original, `image_info` from its metadata — FFT finds the 16–32 px shift introduced by the model and slices the window back to a 1:1 match with the original;
4. **Output**: size = original content size, pixels 1:1, no copy streaks, no misalignment.

> 💡 **Large-image protection**: for very large inputs, insert a "Scale Image to Suitable Size" node (e.g. 1.5 MP) before "Resize and Pad Image" to lower the editing resolution — it only affects the editing domain; the crop-mode output size is still governed by `image_info`, so alignment is unaffected.

**Ready-to-use workflow**: [`workflow/Qwen21_无偏移编辑_像素1比1.json`](workflow/Qwen21_无偏移编辑_像素1比1.json) — drag it into ComfyUI (requires a local Qwen Image 2.1 model). The canvas is already set to the recommended **`32 的倍数（四周留边）` with a 64 px margin**, nothing to tune.

## Credits & Attribution

- This project is an improved distribution of [xingyuezhiyuan/Comfyui-txtnode](https://github.com/xingyuezhiyuan/Comfyui-txtnode):
  - **From the original plugin**: the base design of the "Resize and Pad Image" and "Remove Pad from Image" nodes (proportional scaling, center padding, `image_info`-based cropping);
  - **Added/enhanced by this project**: the "Auto Align to Reference" node (entirely new); the `no_upscale` zero-resample scale mode, `transparent` padding and `target_size = 0` auto size of "Resize and Pad Image"; plus the complete no-offset editing pipeline and the ready-to-use workflow.
- Thanks to the original author [@xingyuezhiyuan](https://github.com/xingyuezhiyuan).
- The original project ships no open-source license; this repository is published for study and reference only, and all rights of the original work remain with its author. If you are the original author and wish to change how this repository is published, please open an issue.
