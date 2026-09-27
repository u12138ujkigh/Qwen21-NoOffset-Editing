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

⚠ 往左/上漂移的**硬前提**：crop 模式的窗口起点 = 内容原点 + 位移。若模型把内容
往左/上推，起点是负数，会被 clamp 回 0 —— 校正量被静默吃掉，表现为「输出整体
往左/上偏，且怎么调都没用」。所以此时画布在左/上必须有留边：
「调整图像尺寸填充」请选「32 的倍数（四周留边）」或「1:1（四周留边）」档位
（留边 ≥ 模型漂移量）；也可以选「32 的倍数（四边自定义）」，只给会漂的方向留边 ——
同样的画布面积能拿到更大的单边余量。纯右/下补边的档位、以及贴边居中的旧
「1:1（方形画布）」档位只在模型往右/下漂移时才有校正能力。
命中这种情况时本节点会打印明确的修复指引（含具体该填多少），不再静默通过。
"""
import numpy as np
import torch
from comfy_api.latest import io

try:
    from .resize_pad import ImageInfo, parse_image_info as _parse_image_info
except ImportError:  # 兼容不同包导入环境
    from nodes.resize_pad import ImageInfo, parse_image_info as _parse_image_info


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


def _ceil32(v):
    """向上取整到 32 的倍数（「调整图像尺寸填充」target_size=0 的规则）。"""
    return -(-int(v) // 32) * 32


def _round32(v):
    """四舍五入到 32 的倍数（原生 TextEncodeQwenImage21 的规则）。"""
    return int(round(int(v) / 32)) * 32


def _diagnose_domain_mismatch(iw, ih, rw, rh, crop_to_reference, info=None):
    """两路尺寸不一致时，追加一段「可能的错位来源」提示，便于用户自查接线。

    info = 「调整图像尺寸填充」的填充元数据（已解析为六元组），有它才能做最要命的那条
    判定：**解码输出其实是「未填充的原图」域** —— 用户把 TextEncodeQwenImage21 的
    image_1 接成了原图（不是填充后的画布）。此时补边只送进了本节点，
    模型从头到尾没见过补边，latent/解码都在原图尺寸域里，必然对不上。
    """
    tips = []
    # 情形零：解码输出 = 原图内容区按 32 四舍五入（TextEncodeQwenImage21 建 latent 的规则），
    # 而参考图 = 填充后的画布 —— 接线错误的铁证，优先给出（其余泛泛提示都不如这条准）
    if info is not None:
        left, top, right, bottom, cw, ch = info
        win_w, win_h = cw - left - right, ch - top - bottom
        if (cw, ch) == (rw, rh) and win_w > 0 and win_h > 0:
            cand = {(_round32(win_w), _round32(win_h)), (_round32(win_h), _round32(win_w))}
            if (iw, ih) in cand:
                tips.append(
                    f"解码输出 {iw}x{ih} 恰好等于「内容区 {win_w}x{win_h} 按 32 四舍五入」的尺寸，"
                    f"而参考图 {rw}x{rh} 是填充后的画布 —— 说明「TextEncodeQwenImage21」的 image_1 "
                    "接的是**未填充的原图**：它按原图尺寸建 latent，模型整条去噪链路没见过补边，"
                    "补边只送到了本节点的 reference_image，两路必然不同域。\n"
                    "  修复：把「TextEncodeQwenImage21」的 image_1 改接"
                    "「调整图像尺寸填充」的「输出图像」（同一个 padded 域）。"
                )
    il, rl = max(iw, ih), max(rw, rh)
    same_min = abs(min(iw, ih) - min(rw, rh)) <= 32
    # 情形一：两路是同一张图，只是 32 对齐的取整规则不同（长边差 ≤ 32）
    # —— 这是最常见的「假错位」，两路其实同图，只是一个做了 32 上取整 / 四舍五入
    if not tips and same_min and 0 < abs(il - rl) <= 32:
        lo, hi = min(il, rl), max(il, rl)
        rule = None
        if _ceil32(lo) == hi:
            rule = "按 32 向上取整（「调整图像尺寸填充」target_size=0）"
        elif _round32(lo) == hi:
            rule = "四舍五入到 32 倍数（原生 TextEncodeQwenImage21）"
        if rule:
            tips.append(
                f"较大的一侧长边 {hi} 恰好等于另一侧长边 {lo} {rule}的结果 —— "
                "两路很可能是同一张图，只是走了不同的 32 对齐路径。"
                "请让 reference_image 与 image 走同一个尺寸域。"
            )
        else:
            tips.append(
                f"两路长边只差 {abs(il - rl)}（{il} vs {rl}），像是同一张图分别做了 32 倍数对齐，"
                "请检查两路是否走了不同的缩放/填充节点。"
            )
    # 情形二：两路长边差异较大，说明来自不同的输入图
    elif not tips and abs(il - rl) > 32:
        tips.append(
            f"两路长边差距较大（image {il} vs reference {rl}），"
            "通常意味着它们来自不同的输入图 —— "
            "请确认 reference_image 接的是与 image 同尺寸域的那一路。"
        )
    # 情形三：多图参考场景（batch 内尺寸不齐）
    if crop_to_reference:
        tips.append("当前 crop_to_reference=True：若参考图来自裁剪后的原图域，请关闭该选项。")
    if not tips:
        return ""
    return "\n【诊断】" + "\n· ".join([""] + tips)


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

    返回 (窗口, 是否精确)。「精确」= 窗口起点没有被 clamp 掉所需位移。
    clamp 一旦发生，说明该方向**没有留边**，校正量会静默丢失
    （表现就是「模型往左漂 16px，输出就往左偏 16px」），调用方必须据此告警。
    """
    H, W = arr.shape[:2]
    oy, ox = origin
    need_y, need_x = oy + dy, ox + dx
    y0 = int(np.clip(need_y, 0, max(0, H - win_h)))
    x0 = int(np.clip(need_x, 0, max(0, W - win_w)))
    exact = (y0 == need_y and x0 == need_x)
    return arr[y0 : y0 + win_h, x0 : x0 + win_w], exact


def _warn_insufficient_margin(dx, dy, margins=None):
    """crop 模式因留边不足而无法校正位移时，给出可直接照做的修复指引。

    margins = (left, top, right, bottom)：内容区到画布四边的实际余量（来自 image_info）。
    有它才能算出**缺口**和**该填多少** —— 只说「设为不小于 N」用户仍可能填回 32（N<64 时
    最近的下取整恰好就是 32），于是再跑一次还是偏。所以这里必须给到具体数值。
    """
    dirs = []
    if dx < 0:
        dirs.append("左")
    elif dx > 0:
        dirs.append("右")
    if dy < 0:
        dirs.append("上")
    elif dy > 0:
        dirs.append("下")
    need = max(abs(dx), abs(dy))
    suggest = -(-need // 8) * 8    # 8 的倍数，且 ≥ need（留边粒度 8px）
    # 四边自定义档位的控件名：用于给出「只给该方向留边」的具体填法
    _CTRL = {"左": "「左边留边」", "右": "「右边留边」",
             "上": "「上边留边」", "下": "「下边留边」"}
    _names = "、".join(_CTRL[d] for d in dirs if d in _CTRL)

    avail = None
    if margins is not None:
        left, top, right, bottom = margins
        cand = []
        if dx < 0:
            cand.append(left)
        elif dx > 0:
            cand.append(right)
        if dy < 0:
            cand.append(top)
        elif dy > 0:
            cand.append(bottom)
        if cand:
            avail = min(cand)

    if avail is None:
        head = (
            "[自动对齐到参考图] 校正量不足：模型把内容向 %s 推了 %dpx，"
            "但画布在该方向没有余量（画布外取不到像素），这部分偏移只能留在输出里。"
            % ("/".join(dirs) or "?", need)
        )
    else:
        head = (
            "[自动对齐到参考图] 校正量不足：模型把内容向 %s 推了 %dpx，"
            "但画布在该方向只留了 %dpx 余量，差 %dpx 取不到（画布外没有像素），"
            "这 %dpx 只能留在输出里 —— 输出仍会向 %s 偏 %dpx。"
            % ("/".join(dirs) or "?", need, avail, need - avail, need - avail,
               "/".join(dirs) or "?", need - avail)
        )
    print(
        head + "\n"
        "  修复（二选一）：\n"
        "   a) 四边要同样余量：画布模式选「32 的倍数（四周留边）」或「1:1（四周留边）」，"
        "把「四周留边」设为不小于 %d（建议正好 %d）——留边就是该方向能校正的最大像素数。\n"
        "   b) 只想给会漂的方向留边、省画布面积：画布模式选「32 的倍数（四边自定义）」，"
        "把 %s 填成不小于 %d（建议正好 %d），其余方向可以留 0。"
        % (need, suggest, _names or "对应方向", need, suggest)
    )


class AutoAlignToReferenceNode(io.ComfyNode):
    """把编辑输出图的内容自动平移回与参考原图对齐的位置"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AutoAlignToReferenceNode",
            display_name="自动对齐到参考图",
            category="txtnode",
            inputs=[
                io.Image.Input("image", display_name="编辑输出"),
                io.Image.Input("reference_image", display_name="参考原图"),
                io.Int.Input("max_shift", default=64, min=0, max=512, step=8,
                             display_name="最大位移（像素）",
                             tooltip="在此范围内搜索位移；模型固有位移约 16~32px，64 足够"),
                # True：不平移，直接切对齐窗口（零复制边缘）；配合 image_info 使用，
                # 本节点放在「移除图像填充」之前（padded 域）
                io.Boolean.Input("crop_to_reference", default=False,
                                 display_name="裁剪到参考域",
                                 tooltip="开启后不再平移，直接切出对齐窗口（零复制边缘），输出回到原图域"),
                # FFT 峰相关度低于该值时认为位移不可信、保持原图不动（0 = 总是校正）
                io.Float.Input("min_confidence", default=0.0, min=0.0, max=1.0, step=0.05,
                               display_name="位移可信度下限",
                               tooltip="FFT 峰相关度低于该值时认为位移不可信、保持不动；0 = 总是校正"),
                # 调整图像尺寸填充的元数据：crop 模式下提供内容区几何与输出尺寸
                ImageInfo.Input("image_info", optional=True, display_name="填充元数据"),
            ],
            outputs=[
                io.Image.Output("image", display_name="对齐后图像"),
                io.Int.Output("shift_dx", display_name="水平位移"),
                io.Int.Output("shift_dy", display_name="垂直位移"),
            ],
        )

    @classmethod
    def execute(cls, image, reference_image, max_shift=64,
                crop_to_reference=False, min_confidence=0.0, image_info=None):
        ref_frames = _to_gray(reference_image)
        ref = _norm(ref_frames[0])
        img_frames = _to_gray(image)
        arr = image.detach().cpu().numpy()

        # 填充元数据先解析：尺寸不一致时也要用它判定「两路各自在哪个域」
        info = _parse_image_info(image_info) if image_info is not None else None

        if ref.shape != img_frames[0].shape:
            ih, iw = img_frames[0].shape
            rh, rw = ref.shape
            # 常见的「尺寸域错位」诊断：看两路是否只是 32 倍数取整的差异
            hint = _diagnose_domain_mismatch(iw, ih, rw, rh, crop_to_reference, info)
            raise ValueError(
                f"自动对齐到参考图：image {iw}x{ih} 与 "
                f"reference_image {rw}x{rh} 尺寸不一致。"
                "两路必须同域：crop 模式下 reference_image 接「调整图像尺寸填充」的 "
                "output_image（padded 域）、image 接同域的解码输出；"
                "若参考图来自裁剪后的原图域，请关闭 crop_to_reference。"
                + hint
            )

        # crop 模式解析内容区几何：(left, top, right, bottom, canvas_w, canvas_h)
        # 兼容旧 5 元组 (…, canvas) —— 那种情况两轴都取该值（方形画布）。
        origin = None
        win = None
        margins = None
        if crop_to_reference and image_info is not None:
            parsed = info   # 上面已解析（尺寸校验也要用，故不重复解析）
            if parsed is not None:
                left, top, right, bottom, canvas_w, canvas_h = parsed
                if canvas_w > 1 and canvas_h > 1 and right >= 0 and bottom >= 0:
                    win_h = canvas_h - top - bottom
                    win_w = canvas_w - left - right
                    if 0 < win_h <= arr.shape[1] and 0 < win_w <= arr.shape[2]:
                        origin = (top, left)
                        win = (win_h, win_w)
                        # 四边实际余量：留边就是「自动对齐」在该方向能校正的最大像素数，
                        # 告警要用它算缺口（只报「偏移了多少」用户不知道该把留边填多大）。
                        margins = (left, top, right, bottom)

        if crop_to_reference and win is None:
            print(
                "[自动对齐到参考图] 已开启「裁剪到参考域」但没有可用的填充元数据，"
                "将退化为整图平移校正（边缘会出现复制条纹）。"
                "请把「填充元数据」接到「调整图像尺寸填充」的 image_info 输出。"
            )

        # 位移估计只在「内容窗口」内做：参考图里补边区是空的（透明 ⇒ RGB 全 0），
        # 而模型会把那块也画上内容 —— 拿整张画布做互相关会被这块「假内容」拉低峰的
        # 可信度、甚至带偏峰位。裁到内容窗后两边都是真内容，估计才干净。
        if win is not None:
            oy, ox = origin
            ref_probe = ref[oy:oy + win[0], ox:ox + win[1]]
        else:
            ref_probe = ref

        outs = []
        dx_out = dy_out = 0
        for i in range(arr.shape[0]):
            frame = _norm(img_frames[i])
            if win is not None:
                oy, ox = origin
                frame_probe = frame[oy:oy + win[0], ox:ox + win[1]]
            else:
                frame_probe = frame
            dx, dy, score = _estimate_shift(ref_probe, frame_probe, max_shift)
            if score < min_confidence:
                dx = dy = 0
            dx_out, dy_out = dx, dy
            if crop_to_reference and win is not None:
                crop, exact = _crop_aligned(arr[i], dx, dy, win[0], win[1], origin)
                if not exact:
                    _warn_insufficient_margin(dx, dy, margins)
                outs.append(crop)
            else:
                outs.append(_shift(arr[i], dx, dy))
        result = torch.from_numpy(np.stack(outs).astype(np.float32))
        return io.NodeOutput(result, dx_out, dy_out)
