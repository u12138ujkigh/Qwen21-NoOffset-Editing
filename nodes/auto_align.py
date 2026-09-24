# -*- coding: utf-8 -*-
"""AutoAlignToReferenceNode — 输出图相对参考原图的整像素位移自动校正。

背景：Qwen Image Edit 等全重绘模型编辑后，内容会整体平移 1~2 个 latent 格
（实测 16~32px），这是模型 attention 层的固有行为，任何输入端的填充/对齐
管线都无法预防。本节点在 max_shift 范围内用 FFT 互相关估计位移，再校正：

- crop_to_reference=False（默认，旧行为）：整图平移回对齐位置，
  出界处用边缘最近像素复制填充（小位移下边缘会产生复制条纹）。
- crop_to_reference=True（推荐，配合 image_info）：不平移，直接从源图
  切出「对齐后对应参考内容区」的窗口 —— 窗口内全部是真实像素，
  无任何复制填充。image_info 提供内容区几何（来自调整图像尺寸填充），
  输出尺寸 = 原图域尺寸；此时本节点应放在「移除图像填充」之前
  （padded 域），reference_image 接「调整图像尺寸填充」的输出。
"""
import numpy as np
import torch
from comfy_api.latest import io

try:
    from .resize_pad import ImageInfo
except ImportError:  # 兼容不同包导入环境
    from nodes.resize_pad import ImageInfo


def _to_gray(t):
    """[B,H,W,C] float 0..1 → [B,H,W] 灰度 float32。"""
    arr = t.detach().cpu().numpy()
    if arr.shape[-1] >= 3:
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return (arr[..., :3].astype(np.float32) @ w).astype(np.float32)
    return arr[..., 0].astype(np.float32)


def _norm(a):
    a = a - a.mean()
    s = a.std()
    return a / s if s > 1e-6 else a


def _overlap_ncc(ref, img, dy, dx):
    """在 (dy,dx) 位移下的两图重叠区计算标准 NCC（-1..1）。

    FFT 循环互相关的原始峰值含周期环绕项、且上界非 1，不能直接当置信度；
    峰位置确定后只在重叠区各自 z-score 再点积，才是可比的相似度。
    """
    H, W = ref.shape
    if dy >= 0:
        ry0, ry1, iy0, iy1 = 0, H - dy, dy, H
    else:
        ry0, ry1, iy0, iy1 = -dy, H, 0, H + dy
    if dx >= 0:
        rx0, rx1, ix0, ix1 = 0, W - dx, dx, W
    else:
        rx0, rx1, ix0, ix1 = -dx, W, 0, W + dx
    a = ref[ry0:ry1, rx0:rx1]
    b = img[iy0:iy1, ix0:ix1]
    if a.size == 0 or b.size == 0:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return -1.0
    return float((a * b).mean() / (sa * sb))


def _estimate_shift(ref, img, max_shift):
    """估计 img 相对 ref 的内容位移 (dx, dy)（内容向右下移动为正）。

    返回 (dx, dy, score)。约定：img[y, x] ≈ ref[y - dy, x - dx]。
    score 为峰位移处重叠区的标准 NCC（-1..1），供 min_confidence 阈值判断。
    """
    fa = np.fft.rfft2(_norm(ref))
    fb = np.fft.rfft2(_norm(img))
    cc = np.fft.irfft2(fb * np.conj(fa), s=ref.shape)
    H, W = ref.shape
    best = (0, 0, -2.0)
    for dy in range(-max_shift, max_shift + 1):
        row = cc[dy % H]
        for dx in range(-max_shift, max_shift + 1):
            c = float(row[dx % W])
            if c > best[2]:
                best = (dx, dy, c)
    dx, dy, _peak = best
    # cc[py, px] 峰 => img[y, x] ≈ ref[y - py, x - px]，即 img 是 ref 平移 (px,py) 的结果
    # 按"向右下为正"约定：位移 (dx, dy) = (px, py)（峰坐标本身），校正 _shift(arr, dx, dy) 即搬回
    return dx, dy, _overlap_ncc(ref, img, dy, dx)


def _shift(arr, dx, dy):
    """out[y,x] = arr[y+dy, x+dx]（出界取边缘最近像素）。"""
    ys = np.clip(np.arange(arr.shape[0]) + dy, 0, arr.shape[0] - 1)
    xs = np.clip(np.arange(arr.shape[1]) + dx, 0, arr.shape[1] - 1)
    return arr[ys][:, xs]


def _crop_aligned(arr, dx, dy, win_h, win_w, origin):
    """从 arr 切出对齐窗口：out[a,b] = arr[oy+a+dy, ox+b+dx]。

    (oy, ox) 为参考内容区在 arr 坐标系中的左上角，(win_h, win_w) 为窗口尺寸。
    窗口起点 clamp 到图像内 —— 无边缘复制填充（极少量越界时窗口整体
    平移 clamp，牺牲亚窗口精度换取干净边缘）。
    """
    H, W = arr.shape[:2]
    oy, ox = origin
    y0 = int(np.clip(oy + dy, 0, max(0, H - win_h)))
    x0 = int(np.clip(ox + dx, 0, max(0, W - win_w)))
    return arr[y0 : y0 + win_h, x0 : x0 + win_w]


class AutoAlignToReferenceNode(io.ComfyNode):
    """把编辑输出图的内容自动平移回与参考原图对齐的位置"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AutoAlignToReferenceNode",
            display_name="自动对齐到参考图",
            category="txtnode",
            inputs=[
                io.Image.Input("image"),
                io.Image.Input("reference_image"),
                io.Int.Input("max_shift", default=64, min=0, max=512, step=8),
                # True：不平移，直接切对齐窗口（零复制边缘）；配合 image_info 使用，
                # 本节点放在「移除图像填充」之前（padded 域）
                io.Boolean.Input("crop_to_reference", default=False),
                # FFT 峰相关度低于该值时认为位移不可信、保持原图不动（0 = 总是校正）
                io.Float.Input("min_confidence", default=0.0, min=0.0, max=1.0, step=0.05),
                # 调整图像尺寸填充的元数据：crop 模式下提供内容区几何与输出尺寸
                ImageInfo.Input("image_info", optional=True),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Int.Output("shift_dx"),
                io.Int.Output("shift_dy"),
            ],
        )

    @classmethod
    def execute(cls, image, reference_image, max_shift=64,
                crop_to_reference=False, min_confidence=0.0, image_info=None):
        ref_frames = _to_gray(reference_image)
        ref = _norm(ref_frames[0])
        img_frames = _to_gray(image)
        arr = image.detach().cpu().numpy()

        if ref.shape != img_frames[0].shape:
            raise ValueError(
                f"自动对齐到参考图：image {img_frames[0].shape[1]}x{img_frames[0].shape[0]} 与 "
                f"reference_image {ref.shape[1]}x{ref.shape[0]} 尺寸不一致。"
                "两路必须同域：crop 模式下 reference_image 接「调整图像尺寸填充」的 "
                "output_image（padded 域）、image 接同域的解码输出；"
                "若参考图来自裁剪后的原图域，请关闭 crop_to_reference。"
            )

        # crop 模式解析内容区几何：(left, top, right, bottom, canvas_size)
        origin = None
        win = None
        if crop_to_reference and image_info is not None:
            info = image_info
            if isinstance(info, (list, tuple)) and info and isinstance(info[0], (list, tuple)):
                info = info[0]
            left, top, right, bottom, canvas = [int(v) for v in info[:5]]
            if canvas > 1 and right >= 0 and bottom >= 0:
                win_h = canvas - top - bottom
                win_w = canvas - left - right
                if 0 < win_h <= arr.shape[1] and 0 < win_w <= arr.shape[2]:
                    origin = (top, left)
                    win = (win_h, win_w)

        outs = []
        dx_out = dy_out = 0
        for i in range(arr.shape[0]):
            dx, dy, score = _estimate_shift(ref, _norm(img_frames[i]), max_shift)
            if score < min_confidence:
                dx = dy = 0
            dx_out, dy_out = dx, dy
            if crop_to_reference and win is not None:
                outs.append(_crop_aligned(arr[i], dx, dy, win[0], win[1], origin))
            else:
                outs.append(_shift(arr[i], dx, dy))
        result = torch.from_numpy(np.stack(outs).astype(np.float32))
        return io.NodeOutput(result, dx_out, dy_out)
