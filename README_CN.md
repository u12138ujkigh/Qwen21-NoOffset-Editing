# Qwen21 无偏移编辑节点 (Qwen21 No-Offset Editing)

> **关于本项目**：独立发行的 ComfyUI 自定义节点包，改进自 [xingyuezhiyuan/Comfyui-txtnode](https://github.com/xingyuezhiyuan/Comfyui-txtnode)。只保留无偏移编辑管线所需的三个节点，并新增了 **「自动对齐到参考图」** 节点与 **像素 1:1 无偏移编辑管线**。

[English](README.md) | **简体中文**

---

## 为什么需要无偏移编辑

Qwen Image Edit 等「全重绘」模型编辑图片时，输出内容会相对原图**整体平移 1~2 个 latent 格**（1024 分辨率下约 16~32 像素）。这是模型 attention 层的固有行为，无法从输入端预防，直接后果：

- 把输出直接裁回原尺寸 → 内容位置错位；
- 用边缘复制填充校正位移 → 画面边缘出现「复制条纹」。

本插件提供机制性解法——**对齐即裁剪（crop 模式）**：不平移图像，而是用 FFT 互相关找出位移后，直接从解码图中切出与原图内容**逐像素对齐**的窗口。窗口内全部是真实像素，从根源上消除复制条纹与错位。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/u12138ujkigh/Qwen21-NoOffset-Editing.git
```

重启 ComfyUI，节点位于 `txtnode` 分类下。

> ⚠️ **兼容性警告**：本包节点 ID 与 Comfyui-txtnode 完全相同（`ResizeAndPadNode` / `RemovePadFromImageNode` / `AutoAlignToReferenceNode`）。**不要与 Comfyui-txtnode（或其任何变体）同时安装**，否则节点定义会冲突。

## 节点介绍

### 1. 调整图像尺寸填充（Resize and Pad Image）— 增强版

将图像等比缩放并居中填充到方形画布，输出 `output_image` 与填充元数据 `image_info`（五元组：左/上/右/下补边宽度、画布尺寸）。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `target_size` | INT | 1024 | 画布边长；**0 = 自动**（长边向上取整到 32 的倍数） |
| `resolution_multiple` | INT | 8 | 将 target_size 吸附到该值的倍数；0 = 不吸附 |
| `upscale_method` | COMBO | lanczos | lanczos / bicubic / area / nearest |
| `resize_and_pad` | BOOLEAN | True | 关闭时旁路（原样输出） |
| `background_color` | STRING | #000000 | 补边颜色；**`transparent`** = 透明补边（输出 RGBA） |
| `scale_mode` | COMBO | fill_target | `fill_target` = 等比缩放至恰好放入画布；**`no_upscale`** = 只缩不放大 |

**相对原插件的增强**（无偏移管线的关键）：

- `scale_mode = no_upscale` + `target_size = 0`：内容**零重采样**方形化——内容像素 1:1 进入 padded 画布，编辑后可与原图逐像素比对；
- `background_color = transparent`：透明补边，避免黑色补边被模型当成「画面内容」重绘出杂物。

### 2. 移除图像填充（Remove Pad from Image）

按 `image_info` 元数据裁掉补边，恢复原始宽高比。与「调整图像尺寸填充」配套的通用节点；在推荐的无偏移管线（crop 模式）中**不需要**它。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `input_image` | IMAGE | - | padded 域图像 |
| `image_info` | IMAGE_INFO | - | 来自「调整图像尺寸填充」 |
| `remove_pad` | BOOLEAN | True | 关闭时旁路 |
| `latent_scale` | FLOAT | 0.0 | 可选，latent 空间缩放系数（精确匹配用） |

### 3. 自动对齐到参考图（Auto Align to Reference）— 本项目新增

在 `max_shift` 范围内用 **FFT 循环互相关**估计编辑输出相对参考原图的整像素位移，再按所选模式校正：

- **平移模式**（`crop_to_reference = False`，旧行为）：整图平移回对齐位置，出界处用边缘最近像素复制填充（小位移下边缘会产生复制条纹）；
- **裁剪模式**（`crop_to_reference = True`，**推荐**）：不平移，直接从源图切出「对齐后对应参考内容区」的窗口——窗口内全部是真实像素，**零复制填充**。此时节点应放在 padded 域：`reference_image` 接「调整图像尺寸填充」的输出，`image_info` 接其元数据。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `image` | IMAGE | - | 编辑输出图（与参考图同域同尺寸） |
| `reference_image` | IMAGE | - | 参考原图 |
| `max_shift` | INT | 64 | 最大搜索位移（px），0~512 |
| `crop_to_reference` | BOOLEAN | False | **裁剪模式开关** |
| `min_confidence` | FLOAT | 0.0 | 重叠区 NCC 置信度阈值（0~1），低于则视为位移不可信、保持不动；0 = 总是校正 |
| `image_info` | IMAGE_INFO (可选) | - | 内容区几何（裁剪模式必需，来自「调整图像尺寸填充」） |

**输出**：`image`（校正后图像）、`shift_dx` / `shift_dy`（检测到的位移，px）。

## 无偏移编辑管线（推荐用法）

```
加载图像 ─▶ 调整图像尺寸填充 ─▶ VAE编码 ─▶ KSampler ─▶ VAE解码 ─▶ 自动对齐到参考图 ─▶ 保存图像
                 │ (no_upscale                                            │        (crop_to_reference=True)
                 │  +transparent)                                        │
                 ├────────────── image_info ─────────────────────────────┤
                 └── output_image ──▶ (reference_image) ─────────────────┘
```

1. **零重采样方形化**：调整图像尺寸填充，`scale_mode = no_upscale`、`target_size = 0`、`background_color = transparent`——内容像素 1:1 进入方形画布，补边透明；
2. **正常编辑**：VAE 编码 → KSampler（Qwen Image Edit）→ VAE 解码。模型在 padded 域创作，补边区也会被重绘（属正常，裁剪模式会把它裁掉）；
3. **对齐即裁剪**：自动对齐到参考图，`crop_to_reference = True`，`reference_image` 接 padded 域原图、`image_info` 接元数据——FFT 找出模型引入的 16~32px 位移，直接切回与原图 1:1 的内容窗口；
4. **输出**：尺寸 = 原图内容区尺寸，像素 1:1，无复制条纹、无错位。

> 💡 **大图保护**：图片太大时可在「调整图像尺寸填充」前加「图像缩放到合适尺寸」（如 1.5MP）来降低编辑分辨率——只影响编辑域，裁剪模式的输出尺寸仍由 `image_info` 决定，与对齐不冲突。

**开箱即用工作流**：[`workflow/Qwen21_无偏移编辑_像素1比1.json`](workflow/Qwen21_无偏移编辑_像素1比1.json)，拖入 ComfyUI 即可使用（需本地具备 Qwen Image 2.1 模型）。

## 来源与致谢

- 本项目改进自 [xingyuezhiyuan/Comfyui-txtnode](https://github.com/xingyuezhiyuan/Comfyui-txtnode)：
  - **来自原插件**：「调整图像尺寸填充」「移除图像填充」两个节点的基础设计（等比缩放、居中填充、`image_info` 元数据裁剪）；
  - **本项目新增/增强**：「自动对齐到参考图」节点（全新）；「调整图像尺寸填充」的 `no_upscale` 零重采样缩放模式、`transparent` 透明填充、`target_size = 0` 自动尺寸；以及完整的无偏移编辑管线与开箱即用工作流。
- 感谢原作者 [@xingyuezhiyuan](https://github.com/xingyuezhiyuan)。
- 原项目未附带开源许可证；本项目以公开仓库形式发布，仅用于学习交流，版权归原作者所有。如你是原作者并希望调整本仓库的发布方式，请提 issue 联系。
