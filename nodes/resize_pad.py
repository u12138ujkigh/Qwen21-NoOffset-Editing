import torch
import numpy as np
from PIL import Image
from typing import List
from comfy_api.latest import io


# ============================================================
# 类型转换工具函数
# ============================================================

def tensor_to_pil(tensors) -> List[Image.Image]:
    """将 ComfyUI IMAGE 张量 [B,H,W,C] 转换为 PIL Image 列表"""
    if isinstance(tensors, np.ndarray):
        arr = tensors
    else:
        arr = tensors.detach().cpu().numpy()
    imgs = []
    for tensor in arr:
        img = (np.clip(tensor, 0.0, 1.0) * 255.0).astype(np.uint8)
        imgs.append(Image.fromarray(img))
    return imgs


def pil_to_tensor(pil_images: List[Image.Image]) -> torch.Tensor:
    """将 PIL Image 列表转换回 ComfyUI IMAGE 张量 [B,H,W,C]"""
    return torch.stack(
        [
            torch.from_numpy(
                np.array(pil_image).astype(np.float32) / 255.0
            )
            for pil_image in pil_images
        ]
    )


# 自定义 IMAGE_INFO 类型，用于在两节点间传递填充元数据
ImageInfo = io.Custom("IMAGE_INFO")


# ============================================================
# ResizeAndPadNode — 调整尺寸并填充
# ============================================================

# 与 ComfyUI_LayerStyle 的「按宽高比缩放 V2」相同的约定：填这些关键字（或留空）
# 表示补边透明，节点输出 RGBA（4 通道），补边 alpha=0，上游带来的 alpha 原样保留。
TRANSPARENT_KEYWORDS = {"transparent", "none", "alpha", "clear"}


def _wants_transparent(background_color) -> bool:
    """背景色请求透明时返回 True（不区分大小写，空串也算）。"""
    if background_color is None:
        return True
    return str(background_color).strip().lower() in TRANSPARENT_KEYWORDS or str(background_color).strip() == ""


def _content_size(orig_w: int, orig_h: int, target_size: int, scale_mode: str):
    """等比缩放后的内容尺寸。

    target_size <= 0 表示不缩放（配合 no_upscale 实现内容零重采样）。
    no_upscale 只缩不放大；fill_target 可放大到刚好放入 target_size 的方形。
    返回 (new_w, new_h, ratio)。
    """
    if target_size is None or target_size <= 0:
        return orig_w, orig_h, 1.0
    ratio = min(target_size / orig_w, target_size / orig_h)
    if scale_mode == "no_upscale":
        ratio = min(ratio, 1.0)          # 只缩不放大
        return int(orig_w * ratio + 0.5), int(orig_h * ratio + 0.5), ratio
    return int(orig_w * ratio), int(orig_h * ratio), ratio


def parse_image_info(image_info):
    """解析 image_info，返回 (left, top, right, bottom, canvas_w, canvas_h)。

    兼容两种格式：
      旧 5 元组 (left, top, right, bottom, canvas)      -> canvas_w = canvas_h = canvas
      新 6 元组 (left, top, right, bottom, canvas_w, canvas_h)
    解析不出来时返回 None。
    """
    info = image_info
    if isinstance(info, list) and len(info) > 0 and isinstance(info[0], (list, tuple)):
        info = info[0]
    if not isinstance(info, (tuple, list)) or len(info) < 5:
        return None
    try:
        left, top, right, bottom = (int(info[0]), int(info[1]), int(info[2]), int(info[3]))
        if len(info) >= 6:
            canvas_w, canvas_h = int(info[4]), int(info[5])
        else:
            canvas_w = canvas_h = int(info[4])   # 旧格式：单一边长（方形）
    except (TypeError, ValueError):
        return None
    return left, top, right, bottom, canvas_w, canvas_h


class ResizeAndPadNode(io.ComfyNode):
    """将图像等比缩放并居中填充到正方形画布，同时记录填充元数据供后续裁剪使用"""

    UPSCALE_METHODS = ["lanczos", "bicubic", "area", "nearest"]

    # 画布模式：
    #   0) 1:1 —— 正方形画布（旧行为，边长 = target_size 或长边向上取整到 32 的倍数）
    #   1) 16 的倍数 —— 保持比例，宽高各自向上取整到 16 的倍数，补边只在右/下
    #   2) 32 的倍数 —— 同上，但取 32 的倍数（与核心 TextEncodeQwenImage21 的取整一致）
    #
    # 关于「16」的硬约束：核心 TextEncodeQwenImage21 在 resolution=0 时按
    # round(dim/32)*32 计算 latent 尺寸，所以**画布宽高必须是 32 的倍数**，
    # 否则 latent 会和画布对不上（例如画布宽 1200 会被它算成 1216）。
    # 因此「16 的倍数」档位在核心编码节点下不可用，除非换成 16 对齐的文本编码。
    # 节点会在这种情况下打印明确警告。
    PAD_MODES = ["1:1（方形画布）", "16 的倍数", "32 的倍数"]
    PAD_SQUARE, PAD_16, PAD_32 = PAD_MODES

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResizeAndPadNode",
            display_name="调整图像尺寸填充",
            category="txtnode",
            inputs=[
                io.Image.Input("input_image", display_name="输入图像"),
                # 0 = 自动：按输入图长边向上取整到 32 的倍数（配合 no_upscale
                # 实现内容零缩放的方形化，Qwen 编辑防像素偏移）
                io.Int.Input("target_size", default=1024, min=0, max=8192, step=1,
                             display_name="画布边长 / 长边上限",
                             tooltip="0 = 自动（1:1 档取长边向上对齐 32 的倍数；比例档不缩放）"),
                io.Int.Input("resolution_multiple", default=8, min=0, max=128, step=8,
                             display_name="边长吸附倍数",
                             tooltip="把「画布边长」吸附到该值的倍数；0 = 不吸附"),
                io.Combo.Input("upscale_method", options=cls.UPSCALE_METHODS,
                               display_name="缩放算法"),
                io.Boolean.Input("resize_and_pad", default=True,
                                 display_name="启用尺寸填充",
                                 tooltip="关闭则原样输出（旁路）"),
                # 填十六进制色（如 #FF0000）则补边涂该色、输出 RGB；
                # 填 transparent（或清空）则补边透明、输出 RGBA，
                # 上游「按宽高比缩放 V2」选 transparent 时其 alpha 会原样保留。
                io.String.Input("background_color", default="#000000",
                                display_name="补边颜色",
                                tooltip="填 transparent（或清空）= 透明补边，输出 RGBA"),
                # fill_target：旧行为，内容等比缩放至恰好放入 target 画布；
                # no_upscale：内容只缩不放大（超过 target 才缩），配合 target 取
                # 长边向上取整的 32 倍数可实现内容零重采样的方形化（Qwen 编辑防偏移）。
                io.Combo.Input("scale_mode", options=["fill_target", "no_upscale"], default="fill_target",
                               display_name="缩放模式",
                               tooltip="fill_target = 缩放至恰好放入画布；no_upscale = 只缩不放大（零重采样）"),
                # 画布模式（新增，追加在末尾以兼容旧工作流）：
                #   1:1（方形画布）—— 旧行为，与之前完全一致
                #   16 的倍数 / 32 的倍数 —— 保持比例，只在右/下补边，
                #     内容原点固定在 (0,0)，严格落在 latent 网格上；
                #     画布面积更小 ⇒ 采样更快（竖图可省 30%~45%）。
                #   注意：核心 TextEncodeQwenImage21 按 32 取整，故 16 档位
                #   需要 16 对齐的文本编码节点，否则会尺寸不一致。
                io.Combo.Input("pad_mode", options=cls.PAD_MODES, default=cls.PAD_SQUARE,
                               display_name="画布模式",
                               tooltip="1:1 = 旧的正方形画布；16/32 的倍数 = 保持比例只补右/下，更省时间"
                                       "（推荐 32：与核心文本编码节点的取整一致）"),
            ],
            outputs=[
                io.Image.Output("output_image", display_name="输出图像"),
                ImageInfo.Output("image_info", display_name="填充元数据"),
            ],
        )

    @classmethod
    def execute(cls, input_image, target_size, resolution_multiple, upscale_method, resize_and_pad,
                background_color="#000000", scale_mode="fill_target", pad_mode=None):
        # pad_mode 缺省（旧工作流没这个控件）时按「1:1 方形画布」，行为与之前完全一致
        pad_mode = pad_mode or cls.PAD_SQUARE
        if pad_mode not in cls.PAD_MODES:
            pad_mode = cls.PAD_SQUARE

        # bypass 模式：直接返回原图，image_info 中 canvas=1 防止下游除零
        if not resize_and_pad:
            return io.NodeOutput(input_image, (0, 0, 0, 0, 1, 1))

        # 将 target_size 吸附到 resolution_multiple 的最近倍数（为 0 时不修正，按原值使用）
        # 注意：只有 target_size > 0 时才吸附；否则 (target_size=0, multiple>0) 会把
        # 自动模式的最大值顶成 multiple，导致自动模式失效。
        if resolution_multiple > 0 and target_size > 0:
            remainder = target_size % resolution_multiple
            if remainder != 0:
                if remainder >= resolution_multiple / 2:
                    target_size = target_size + (resolution_multiple - remainder)
                else:
                    target_size = target_size - remainder
            target_size = max(target_size, resolution_multiple)

        transparent_pad = _wants_transparent(background_color)

        pil_images = tensor_to_pil(input_image)
        if not pil_images:
            return io.NodeOutput(input_image, (0, 0, 0, 0, 1, 1))

        # 重采样算法映射（area 映射到 PIL 的 BOX 滤波器）
        resampling_filter = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "area": Image.Resampling.BOX,
            "nearest": Image.Resampling.NEAREST,
        }[upscale_method]

        aspect_mode = pad_mode != cls.PAD_SQUARE
        multiple = 32 if pad_mode == cls.PAD_32 else 16

        # ---- 画布尺寸：只按第一张图决定（假设批次内尺寸一致）----
        first_w, first_h = pil_images[0].size
        if aspect_mode:
            cw, ch, _ = _content_size(first_w, first_h, target_size, scale_mode)
            canvas_w = -(-cw // multiple) * multiple
            canvas_h = -(-ch // multiple) * multiple
            if multiple == 16 and (canvas_w % 32 or canvas_h % 32):
                print(
                    "[调整图像尺寸填充] 警告：画布 %dx%d 不是 32 的倍数，"
                    "核心 TextEncodeQwenImage21（resolution=0）会把它取整成 %dx%d，"
                    "与画布不一致，下游「自动对齐到参考图」会报尺寸不一致。"
                    "请改用「32 的倍数」档位，或换用 16 对齐的文本编码节点。"
                    % (canvas_w, canvas_h,
                       round(canvas_w / 32) * 32, round(canvas_h / 32) * 32))
        else:
            if target_size <= 0:
                target_size = -(-max(first_w, first_h) // 32) * 32
            canvas_w = canvas_h = target_size

        processed_pil_images = []
        image_info_out = None

        for pil_image in pil_images:
            orig_width, orig_height = pil_image.size
            new_width, new_height, _ = _content_size(orig_width, orig_height, target_size, scale_mode)

            # 内容不许超出画布（批次内尺寸不一致时的兜底，比静默裁掉可读）
            if new_width > canvas_w or new_height > canvas_h:
                raise ValueError(
                    "[调整图像尺寸填充] 内容 %dx%d 超出画布 %dx%d：批次内图像尺寸不一致，"
                    "请把 target_size 设成不小于最长边的值，或逐张处理。"
                    % (new_width, new_height, canvas_w, canvas_h)
                )

            if scale_mode == "no_upscale" and new_width == orig_width and new_height == orig_height:
                resized_image = pil_image  # 零重采样：内容像素原样
            else:
                resized_image = pil_image.resize((new_width, new_height), resample=resampling_filter)

            if aspect_mode:
                # 只在右/下补边：内容原点固定 (0,0)，严格落在 latent 网格上
                pad_left = pad_top = 0
            else:
                # 旧行为：居中
                pad_left = (canvas_w - new_width) // 2
                pad_top = (canvas_h - new_height) // 2

            # 创建画布：填色走 RGB（与旧版行为一致），
            # transparent 走 RGBA —— 补边 alpha=0，图像区保留上游 alpha（无则 255）。
            if transparent_pad:
                padded_image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                padded_image.paste(resized_image.convert("RGBA"), (pad_left, pad_top))
            else:
                padded_image = Image.new("RGB", (canvas_w, canvas_h),
                                         color=str(background_color).strip() or "#000000")
                padded_image.paste(resized_image.convert("RGB"), (pad_left, pad_top))
            processed_pil_images.append(padded_image)

            # 仅从第一张图记录 image_info（假设批次内所有图像尺寸一致）
            if image_info_out is None:
                pad_right = canvas_w - new_width - pad_left
                pad_bottom = canvas_h - new_height - pad_top
                # 6 元组：末尾两位是画布宽高（旧版是单一边长的 5 元组，下游已兼容两种）
                image_info_out = (pad_left, pad_top, pad_right, pad_bottom, canvas_w, canvas_h)

        return io.NodeOutput(pil_to_tensor(processed_pil_images), image_info_out)


# ============================================================
# RemovePadFromImageNode — 移除图像填充
# ============================================================

class RemovePadFromImageNode(io.ComfyNode):
    """根据 image_info 元数据裁剪填充区域，恢复图像原始宽高比"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RemovePadFromImageNode",
            display_name="移除图像填充",
            category="txtnode",
            inputs=[
                io.Image.Input("input_image", display_name="输入图像"),
                ImageInfo.Input("image_info", display_name="填充元数据"),
                io.Boolean.Input("remove_pad", default=True,
                                 display_name="启用移除填充",
                                 tooltip="关闭则原样输出（旁路）"),
                io.Float.Input("latent_scale", default=0.0, optional=True,
                               display_name="已知缩放比（可选）",
                               tooltip="输出图经过已知倍率放大时可填这里；留 0 则按画布自动推算"),
            ],
            outputs=[
                io.Image.Output("output_image", display_name="输出图像"),
            ],
        )

    @classmethod
    def execute(cls, input_image, image_info, remove_pad, latent_scale=0.0):
        # bypass 模式
        if not remove_pad:
            return io.NodeOutput(input_image)

        # 安全提取 image_info（兼容 5 元组旧格式 / 6 元组新格式 / tuple 或 list 包装）
        parsed = parse_image_info(image_info)
        if parsed is None:
            print(f"[RemovePadFromImageNode] 无效的 image_info: {image_info}，旁路返回原图")
            return io.NodeOutput(input_image)
        left, top, right, bottom, canvas_w, canvas_h = parsed

        # 零填充检测（bypass 模式产生的 (0,0,0,0,1,1)）
        if left == 0 and top == 0 and right == 0 and bottom == 0:
            return io.NodeOutput(input_image)

        pil_images = tensor_to_pil(input_image)
        cropped_images = []

        for pil_image in pil_images:
            final_width, final_height = pil_image.size
            # 逐轴算缩放：输出图相对「画布」被整体放大/缩小时（如后面接过放大器），
            # 填充坐标也要按同比例放大。方形画布时两轴相等，与旧行为一致。
            scale_x = final_width / float(canvas_w) if canvas_w else 1.0
            scale_y = final_height / float(canvas_h) if canvas_h else 1.0

            # 若有 latent_scale 且在 10% 容差内匹配，优先使用精确值（两轴同用）
            if latent_scale is not None and latent_scale > 0.0:
                tolerance = 0.1
                if abs(scale_x - float(latent_scale)) <= tolerance * scale_x:
                    scale_x = float(latent_scale)
                if abs(scale_y - float(latent_scale)) <= tolerance * scale_y:
                    scale_y = float(latent_scale)

            # 缩放填充坐标并裁剪（对坐标做边界保护，避免任意尺寸下裁剪框越界或反转报错）
            crop_left = max(0, int(left * scale_x))
            crop_top = max(0, int(top * scale_y))
            crop_right = min(final_width, final_width - int(right * scale_x))
            crop_bottom = min(final_height, final_height - int(bottom * scale_y))

            # 裁剪框无效（宽或高非正）时保留原图
            if crop_right - crop_left <= 0 or crop_bottom - crop_top <= 0:
                cropped_images.append(pil_image)
                continue

            cropped_images.append(pil_image.crop((crop_left, crop_top, crop_right, crop_bottom)))

        return io.NodeOutput(pil_to_tensor(cropped_images))
