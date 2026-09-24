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


class ResizeAndPadNode(io.ComfyNode):
    """将图像等比缩放并居中填充到正方形画布，同时记录填充元数据供后续裁剪使用"""

    UPSCALE_METHODS = ["lanczos", "bicubic", "area", "nearest"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResizeAndPadNode",
            display_name="调整图像尺寸填充",
            category="txtnode",
            inputs=[
                io.Image.Input("input_image"),
                # 0 = 自动：按输入图长边向上取整到 32 的倍数（配合 no_upscale
                # 实现内容零缩放的方形化，Qwen 编辑防像素偏移）
                io.Int.Input("target_size", default=1024, min=0, max=8192, step=1,
                             tooltip="0 = auto: square canvas side = input long side rounded UP to a multiple of 32"),
                io.Int.Input("resolution_multiple", default=8, min=0, max=128, step=8),
                io.Combo.Input("upscale_method", options=cls.UPSCALE_METHODS),
                io.Boolean.Input("resize_and_pad", default=True),
                # 填十六进制色（如 #FF0000）则补边涂该色、输出 RGB；
                # 填 transparent（或清空）则补边透明、输出 RGBA，
                # 上游「按宽高比缩放 V2」选 transparent 时其 alpha 会原样保留。
                io.String.Input("background_color", default="#000000"),
                # fill_target：旧行为，内容等比缩放至恰好放入 target 画布；
                # no_upscale：内容只缩不放大（超过 target 才缩），配合 target 取
                # 长边向上取整的 32 倍数可实现内容零重采样的方形化（Qwen 编辑防偏移）。
                io.Combo.Input("scale_mode", options=["fill_target", "no_upscale"], default="fill_target"),
            ],
            outputs=[
                io.Image.Output("output_image"),
                ImageInfo.Output("image_info"),
            ],
        )

    @classmethod
    def execute(cls, input_image, target_size, resolution_multiple, upscale_method, resize_and_pad, background_color="#000000", scale_mode="fill_target"):
        # bypass 模式：直接返回原图，image_info 中 original_size=1 防止下游除零
        if not resize_and_pad:
            image_info_out = (0, 0, 0, 0, 1)
            return io.NodeOutput(input_image, image_info_out)

        # 将 target_size 吸附到 resolution_multiple 的最近倍数（为 0 时不修正，按原值使用）
        if resolution_multiple > 0:
            remainder = target_size % resolution_multiple
            if remainder != 0:
                if remainder >= resolution_multiple / 2:
                    target_size = target_size + (resolution_multiple - remainder)
                else:
                    target_size = target_size - remainder
            target_size = max(target_size, resolution_multiple)

        transparent_pad = _wants_transparent(background_color)

        pil_images = tensor_to_pil(input_image)
        processed_pil_images = []
        image_info_out = None

        # target_size <= 0：自动模式——按第一张图长边向上取整到 32 的倍数
        if target_size <= 0 and pil_images:
            first_w, first_h = pil_images[0].size
            target_size = -(-max(first_w, first_h) // 32) * 32

        # 重采样算法映射（area 映射到 PIL 的 BOX 滤波器）
        resampling_filter = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "area": Image.Resampling.BOX,
            "nearest": Image.Resampling.NEAREST,
        }[upscale_method]

        for pil_image in pil_images:
            orig_width, orig_height = pil_image.size

            # 等比缩放：取宽高较小比例，确保完整放入正方形。
            # no_upscale 模式下内容小于画布时保持原尺寸（只缩不放大），
            # 保证内容像素零重采样（Qwen 编辑防像素偏移的关键）。
            ratio = min(target_size / orig_width, target_size / orig_height)
            if scale_mode == "no_upscale":
                # 只缩不放大；四舍五入避免浮点截断少 1px
                ratio = min(ratio, 1.0)
                new_width = int(orig_width * ratio + 0.5)
                new_height = int(orig_height * ratio + 0.5)
            else:
                new_width = int(orig_width * ratio)
                new_height = int(orig_height * ratio)

            if scale_mode == "no_upscale" and new_width == orig_width and new_height == orig_height:
                resized_image = pil_image  # 零重采样：内容像素原样
            else:
                resized_image = pil_image.resize((new_width, new_height), resample=resampling_filter)

            pad_left = (target_size - new_width) // 2
            pad_top = (target_size - new_height) // 2

            # 创建画布并居中粘贴：填色走 RGB（与旧版行为一致），
            # transparent 走 RGBA —— 补边 alpha=0，图像区保留上游 alpha（无则 255）。
            if transparent_pad:
                padded_image = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
                padded_image.paste(resized_image.convert("RGBA"), (pad_left, pad_top))
            else:
                padded_image = Image.new("RGB", (target_size, target_size),
                                         color=str(background_color).strip() or "#000000")
                padded_image.paste(resized_image.convert("RGB"), (pad_left, pad_top))
            processed_pil_images.append(padded_image)

            # 仅从第一张图记录 image_info（假设批次内所有图像尺寸一致）
            if image_info_out is None:
                pad_right = target_size - new_width - pad_left
                pad_bottom = target_size - new_height - pad_top
                image_info_out = (pad_left, pad_top, pad_right, pad_bottom, target_size)

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
                io.Image.Input("input_image"),
                ImageInfo.Input("image_info"),
                io.Boolean.Input("remove_pad", default=True),
                io.Float.Input("latent_scale", default=0.0, optional=True),
            ],
            outputs=[
                io.Image.Output("output_image"),
            ],
        )

    @classmethod
    def execute(cls, input_image, image_info, remove_pad, latent_scale=0.0):
        # bypass 模式
        if not remove_pad:
            return io.NodeOutput(input_image)

        # 安全提取 image_info（兼容 tuple 或 list 格式）
        image_info_tuple = image_info
        if isinstance(image_info, list) and len(image_info) > 0:
            image_info_tuple = image_info[0]

        if not isinstance(image_info_tuple, (tuple, list)) or len(image_info_tuple) < 5:
            print(f"[RemovePadFromImageNode] 无效的 image_info: {image_info}，旁路返回原图")
            return io.NodeOutput(input_image)

        left, top, right, bottom, original_size = image_info_tuple[:5]

        # 零填充检测（bypass 模式产生的 (0,0,0,0,1)）
        if left == 0 and top == 0 and right == 0 and bottom == 0:
            return io.NodeOutput(input_image)

        pil_images = tensor_to_pil(input_image)
        cropped_images = []

        for pil_image in pil_images:
            final_width, final_height = pil_image.size
            scale_from_image = final_width / float(original_size)
            scale_factor = scale_from_image

            # 若有 latent_scale 且在 10% 容差内匹配，优先使用精确值
            if latent_scale is not None and latent_scale > 0.0:
                tolerance = 0.1
                diff = abs(scale_from_image - float(latent_scale))
                if diff <= tolerance * scale_from_image:
                    scale_factor = float(latent_scale)

            # 缩放填充坐标并裁剪（对坐标做边界保护，避免任意尺寸下裁剪框越界或反转报错）
            crop_left = max(0, int(left * scale_factor))
            crop_top = max(0, int(top * scale_factor))
            crop_right = min(final_width, final_width - int(right * scale_factor))
            crop_bottom = min(final_height, final_height - int(bottom * scale_factor))

            # 裁剪框无效（宽或高非正）时保留原图
            if crop_right - crop_left <= 0 or crop_bottom - crop_top <= 0:
                cropped_images.append(pil_image)
                continue

            cropped_images.append(pil_image.crop((crop_left, crop_top, crop_right, crop_bottom)))

        return io.NodeOutput(pil_to_tensor(cropped_images))
