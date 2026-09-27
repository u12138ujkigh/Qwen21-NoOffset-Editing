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

# 补边填充方式（下拉五选一）：标签即取值 —— ComfyUI 把下拉选中的字符串原样作为输入传给节点。
#
# 前三项是纯色/透明；后两项把补边做成「内容的延续」，这是让 Qwen 把它当画面重绘的关键：
# 实测（红潮模型、种子固定、画布 1056×1312、中性提示词）——
#   纯白 → 解码图左侧残留 3px 白带，模型只做了轻微洗白，色带仍在；
#   纯黑 → 模型**逐像素照抄**，解码图左侧 54px 纯黑带原样保留；
#   透明 → 同上，照抄 54px（alpha=0 在 RGB 里即黑边）；
#   镜像 → 解码图左侧**零平坦带**（std 43），模型当画面重绘，裁剪后 0px 残留，内容保真 NCC 0.9916；
#   边缘延展 → 同样零残留（NCC 0.9845）。
# 即：只有「镜像 / 边缘延展」能真正消掉裁剪后的色带，故默认给「镜像」。
PAD_WHITE = "纯白 (#FFFFFF)"
PAD_BLACK = "纯黑 (#000000)"
PAD_TRANSPARENT = "透明 (alpha=0)"
PAD_MIRROR = "镜像 (reflect)｜推荐"
PAD_EDGE = "边缘延展 (edge)"
PAD_COLORS = [PAD_WHITE, PAD_BLACK, PAD_TRANSPARENT, PAD_MIRROR, PAD_EDGE]

# 需实际「生成像素」的补边方式（按 _resolve_background 返回的 fill_mode 索引）→ numpy.pad 模式名
PAD_NUMPY_MODES = {"reflect": "reflect", "edge": "edge"}

# 旧工作流里手填过的透明关键字（与 ComfyUI_LayerStyle 的「按宽高比缩放 V2」同一约定）。
TRANSPARENT_KEYWORDS = {"transparent", "none", "alpha", "clear", ""}

# 解析结果：fill_mode ∈ {"color", "transparent", "reflect", "edge"}
FILL_COLOR, FILL_TRANSPARENT = "color", "transparent"


def _resolve_background(background_color):
    """把「补边颜色」归一化成 (fill_mode, hex_color)。

    三种来源都要认：控件从「自由字符串」改成下拉后，老工作流里存的 "#FFFFFF" /
    "transparent" 仍会原样送进来（节点已在 validate_inputs 里为它豁免了白名单校验），
    所以这里必须兼容历史写法，否则老工作流的行为会静默改变。
      * 新版下拉：纯白 / 纯黑 / 透明 / 镜像 / 边缘延展；
      * 旧版手填的十六进制色（#FFFFFF、#abc…）—— 原样使用；
      * 旧版透明关键字（transparent / none / alpha / clear，或留空）—— 视为透明。
    返回 (mode, color)：
      ("transparent", None) —— 画布走 RGBA、补边 alpha=0，上游 alpha 原样保留；
      ("reflect"/"edge", None) —— 补边由内容镜像/边缘延展生成（RGB）；
      ("color", "#RRGGBB") —— 纯色补边（RGB）。
    """
    s = "" if background_color is None else str(background_color).strip()
    low = s.lower()
    if low in TRANSPARENT_KEYWORDS or s.startswith("透明"):
        return FILL_TRANSPARENT, None
    if s.startswith("纯白") or low in ("white", "#fff", "#ffffff"):
        return FILL_COLOR, "#FFFFFF"
    if s.startswith("纯黑") or low in ("black", "#000", "#000000"):
        return FILL_COLOR, "#000000"
    if s.startswith("镜像") or low in ("reflect", "mirror", "mirrored"):
        return "reflect", None
    if s.startswith("边缘延展") or low in ("edge", "replicate", "clamp"):
        return "edge", None
    return FILL_COLOR, s


def _build_canvas(resized_image, canvas_w, canvas_h, pad_left, pad_top,
                  fill_mode, pad_color):
    """按补边方式把内容放到画布上。

    * color       —— 纯色底 + 粘贴（RGB）
    * transparent —— 透明底 + 粘贴（RGBA，补边 alpha=0，保留上游 alpha）
    * reflect/edge—— 用 numpy.pad 让补边直接由内容生成（镜像 / 边缘像素延展），
                     模型因此不会把它当作「纯色边框」照抄，而是当画面重绘 ——
                     裁剪后不留色带。
    """
    if fill_mode in PAD_NUMPY_MODES:
        arr = np.asarray(resized_image.convert("RGB"))
        pad_bottom = canvas_h - pad_top - arr.shape[0]
        pad_right = canvas_w - pad_left - arr.shape[1]
        if min(pad_left, pad_top, pad_right, pad_bottom) < 0:
            raise ValueError(
                "[调整图像尺寸填充] 画布 %dx%d 装不下内容 %dx%d（左%d 上%d 右%d 下%d）"
                % (canvas_w, canvas_h, arr.shape[1], arr.shape[0],
                   pad_left, pad_top, pad_right, pad_bottom))
        padded = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=PAD_NUMPY_MODES[fill_mode],
        )
        return Image.fromarray(padded)

    if fill_mode == FILL_TRANSPARENT:
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        canvas.paste(resized_image.convert("RGBA"), (pad_left, pad_top))
        return canvas

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=pad_color)
    canvas.paste(resized_image.convert("RGB"), (pad_left, pad_top))
    return canvas


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
    #   3) 32 的倍数（四周留边）—— 先四边各留 edge_margin 像素，再把画布按 32 向上取整；
    #      内容贴在 (margin, margin)，给「自动对齐到参考图」的 crop 模式留出各方向裁剪余量。
    #   4) 1:1（四周留边）—— 在档位 0 的正方形画布基础上，四边各再留 edge_margin 像素：
    #      先把长边（画布边长）补齐到 32 的倍数，再加 2*margin 做成 1:1（内容居中）。
    #   5) 32 的倍数（四边自定义）—— 四边分别留 margin_left/top/right/bottom 后再按 32 向上
    #      取整，内容贴在 (left, top)。只给「模型真正会漂的方向」留边，比四边均留**面积更小、
    #      余量更大**：例（964×1280，实测左漂 42px）取 左64/上32/右0/下0 → 画布 1056×1312、
    #      左余量 64；而「四边各 32」是 1056×1344、左余量只有 32。
    #      （实测见 _gate/verify_custom_margin.py：前者零位 NCC 1.0000，后者 0.9021 残留约 10px。）
    #
    #      留边粒度 = 8px（= 半个 latent 格；latent 空间压缩比 16）。
    #      实测（真实 VAE encode→decode 往返，不采样，见 _gate/verify_margin_granularity.py）：
    #      margin 取遍 1~64 的各个相位，内容往返漂移恒为 (0,0)、内容 MAE 恒为 0.74/255 ——
    #      即**内容原点不落在 16/32 网格上也不会自己产生漂移**。所以粒度放宽到 8px，
    #      让余量可以精确分配到真正需要的一侧。（早期版本强制吸附到 32 的倍数，
    #      会把用户填的 40 默默吃成 32，白白丢掉 8px 余量。）
    #
    #      ⚠ 档位 3/4/5 才是「防偏移」的关键：crop 模式的对齐窗口起点 = 内容原点 + 位移。
    #      纯右/下补边（档位 1/2）把内容原点钉在 (0,0)，窗口起点一为负就被 clamp 回 0 ——
    #      **校正量被静默吃掉**。档位 0（1:1）在横图上同样贴边：画布宽 == 内容长边
    #      ⇒ pad_left = pad_right = 0，横向一格余量都没有。只有画布在「会漂的方向」上
    #      留下 ≥ 漂移量的余量（档位 3/4/5）才校正得回来。
    #      余量不必四边均等：档位 5 可以只给一个方向留边，用同样的画布面积买到更大的余量。
    #
    #      ⚠ 别把「1:1（四周留边）」误当成「长边多补一格 32」：长边本身已是 32 的倍数时，
    #      「补齐到 32 的倍数 + 做成 1:1」这两步得到的画布与档位 0 **逐像素相同**
    #      （768×715 → 768×768，横向仍是 0 余量），必须靠 edge_margin 额外留边才有余量。
    #
    # 关于「16」的硬约束：核心 TextEncodeQwenImage21 在 resolution=0 时按
    # round(dim/32)*32 计算 latent 尺寸，所以**画布宽高必须是 32 的倍数**，
    # 否则 latent 会和画布对不上（例如画布宽 1200 会被它算成 1216）。
    # 因此「16 的倍数」档位在核心编码节点下不可用，除非换成 16 对齐的文本编码。
    # 节点会在这种情况下打印明确警告。
    PAD_MODES = ["1:1（方形画布）", "16 的倍数", "32 的倍数", "32 的倍数（四周留边）",
                 "1:1（四周留边）", "32 的倍数（四边自定义）"]
    PAD_SQUARE, PAD_16, PAD_32, PAD_MARGIN, PAD_SQUARE_MARGIN, PAD_CUSTOM = PAD_MODES

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
                # 补边填充方式（下拉五选一）。语义见 _resolve_background。
                # 控件虽是下拉，节点仍接受历史上手填的任意十六进制色与 transparent 关键字
                # —— validate_inputs 已为该输入豁免了下拉白名单校验，老工作流无需迁移。
                io.Combo.Input("background_color", options=PAD_COLORS, default=PAD_MIRROR,
                               display_name="补边填充",
                               tooltip="⚠ 模型把「纯色边框」当画面内容照抄，裁剪后必留色带"
                                       "（实测纯黑/透明的带完整保留、纯白只被轻微洗白）。"
                                       "推荐「镜像」或「边缘延展」：补边由内容自身生成，"
                                       "模型会把它当画面重绘，裁剪后零残留（实测内容保真 NCC 0.99）。"
                                       "纯白/纯黑 = 补边填 #FFFFFF/#000000；透明 = 输出 RGBA、补边 alpha=0"),
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
                                       "（推荐 32：与核心文本编码节点的取整一致）；"
                                       "⚠「32 的倍数」与贴边的「1:1」在左/上零余量，"
                                       "微调模型往左漂时校正量会被夹掉，输出就保持原漂移；"
                                       "防偏移请用「32 的倍数（四周留边）」或「1:1（四周留边）」；"
                                       "只想给会漂的方向留边、省画布面积，用「32 的倍数（四边自定义）」"),
                # 四周留边（仅「32 的倍数（四周留边）」档位生效）。
                # 必须是 32 的倍数：内容原点 = margin，仍要落在 latent 网格上
                # （16px = 1 格），否则本节点费劲保持的「内容原点对齐」就白做了。
                #
                # ⚠ 关键语义：内容贴「左/上」放置（pad_left = pad_top = margin），
                #   ceil32 产生的零头全部落在右/下。所以 —— 左/上余量**恰好等于**你填的值，
                #   填 32 就只能向左/上校正 32px；右/下会稍多（margin + 零头）。
                #   实测某微调模型在 964×1280 上向左漂 42px：填 32 时缺口 10px 无法校正，
                #   输出就保持左偏 10px；填 64 才够。默认因此给 64。
                io.Int.Input("edge_margin", default=64, min=0, max=512, step=8,
                             display_name="四周留边（像素）",
                             tooltip="「32 的倍数（四周留边）」与「1:1（四周留边）」两档生效。"
                                     "留边 = 「自动对齐」在该方向能校正的最大像素数："
                                     "内容贴左/上放置，故左/上余量恰好等于你填的值（填 32 只能修 32px），"
                                     "右/下会稍多。粒度 8px；建议 64（覆盖常见 16~48px 漂移），"
                                     "原版模型 32 够，微调版漂移更大时用 96"),
                # 四边自定义留边（仅「32 的倍数（四边自定义）」档位生效）：
                # 只给「模型真正会漂的方向」留边，比四边均留面积更小、余量更大。
                # 例（964×1280）：左 64 / 上 32 / 右 0 / 下 0 → 画布 1056×1312、左余量 64
                #（「四边各 32」是 1056×1344、左余量只有 32）。
                io.Int.Input("margin_left", default=0, min=0, max=512, step=8,
                             display_name="左边留边",
                             tooltip="仅「32 的倍数（四边自定义）」生效。左余量恰好等于该值；"
                                     "向左漂移多少就填多少（粒度 8px）"),
                io.Int.Input("margin_top", default=0, min=0, max=512, step=8,
                             display_name="上边留边",
                             tooltip="仅「32 的倍数（四边自定义）」生效。上余量恰好等于该值"),
                io.Int.Input("margin_right", default=0, min=0, max=512, step=8,
                             display_name="右边留边",
                             tooltip="仅「32 的倍数（四边自定义）」生效。"
                                     "右余量 = 该值 + 画布 32 取整的零头（通常比该值略大）"),
                io.Int.Input("margin_bottom", default=0, min=0, max=512, step=8,
                             display_name="下边留边",
                             tooltip="仅「32 的倍数（四边自定义）」生效。"
                                     "下余量 = 该值 + 画布 32 取整的零头（通常比该值略大）"),
            ],
            outputs=[
                io.Image.Output("output_image", display_name="输出图像"),
                ImageInfo.Output("image_info", display_name="填充元数据"),
            ],
        )

    @classmethod
    def validate_inputs(cls, background_color=None):
        """把「补边填充」纳入自定义校验，从而跳过 ComfyUI 对该下拉取值的白名单检查。

        这是**刻意的兼容措施**：该控件历史上是自由字符串，老工作流里存着 "#FFFFFF" /
        "transparent" 这类值。若不豁免，ComfyUI 会在校验阶段以 value_not_in_list 拒绝
        整个工作流（execution.py 的 `if x not in validate_function_inputs and not
        validate_has_kwargs:` 决定：被本签名列出的输入跳过 min/max/白名单三类默认校验）。
        豁免后由 _resolve_background 归一化历史写法（含纯白/纯黑/透明/镜像/边缘延展五个
        新档位），行为与旧版逐字一致。
        只影响这一个输入，其余输入的默认校验照旧。返回值必须严格为 True 才算通过。
        """
        return True

    @classmethod
    def execute(cls, input_image, target_size, resolution_multiple, upscale_method, resize_and_pad,
                background_color=PAD_MIRROR, scale_mode="fill_target", pad_mode=None,
                edge_margin=64, margin_left=0, margin_top=0, margin_right=0, margin_bottom=0):
        # pad_mode 缺省（旧工作流没这个控件）时按「1:1 方形画布」，行为与之前完全一致
        pad_mode = pad_mode or cls.PAD_SQUARE
        if pad_mode not in cls.PAD_MODES:
            pad_mode = cls.PAD_SQUARE

        # 留边粒度 = 8px（= 半个 latent 格）。实测（真实 VAE encode→decode 往返，
        # 见 _gate/verify_margin_granularity.py）margin 取 1~64 的各个相位，内容往返漂移
        # 恒为 (0,0) —— 内容原点不落在 16/32 网格上**不会**让 VAE 自己产生漂移。
        # 故不再吸附到 32 的倍数：那个吸附会把用户填的 40 默默吃成 32，白丢 8px 余量。
        def _snap8(raw):
            v = max(0, int(raw or 0))
            return v, int(round(v / 8.0)) * 8

        margin = 0
        if pad_mode in (cls.PAD_MARGIN, cls.PAD_SQUARE_MARGIN):
            raw_margin, margin = _snap8(edge_margin)
            if margin != raw_margin:
                print(
                    "[调整图像尺寸填充] 四周留边 %d 已吸附为 %d（粒度为 8px）"
                    % (raw_margin, margin)
                )

        # 四边自定义留边（仅档位 5 生效）。内容贴在 (left, top)，
        # 右/下再叠加画布 32 取整产生的零头 —— 所以左/上余量 = 所填值，右/下会略多。
        m_left = m_top = m_right = m_bottom = 0
        if pad_mode == cls.PAD_CUSTOM:
            raw4 = [int(margin_left or 0), int(margin_top or 0),
                    int(margin_right or 0), int(margin_bottom or 0)]
            snap4 = [max(0, int(round(v / 8.0)) * 8) for v in raw4]
            if snap4 != raw4:
                print(
                    "[调整图像尺寸填充] 四边留边 %s 已吸附为 %s（粒度为 8px）"
                    % (raw4, snap4)
                )
            m_left, m_top, m_right, m_bottom = snap4

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

        fill_mode, pad_color = _resolve_background(background_color)
        # 日志用的可读名（顺带把「纯色会被照抄」这个坑提示出来）
        if fill_mode == FILL_COLOR:
            fill_label = "纯色 %s（模型会照抄 ⇒ 裁剪后可能留色带）" % pad_color
        elif fill_mode == FILL_TRANSPARENT:
            fill_label = "透明 alpha=0（模型会照抄 ⇒ 裁剪后可能留透明带）"
        else:
            fill_label = {"reflect": "镜像（模型当画面重绘）",
                          "edge": "边缘延展（模型当画面重绘）"}[fill_mode]

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

        aspect_mode = pad_mode not in (cls.PAD_SQUARE, cls.PAD_SQUARE_MARGIN)
        multiple = 32 if pad_mode in (cls.PAD_32, cls.PAD_MARGIN,
                                      cls.PAD_SQUARE_MARGIN, cls.PAD_CUSTOM) else 16

        # ---- 画布尺寸：只按第一张图决定（假设批次内尺寸一致）----
        first_w, first_h = pil_images[0].size
        if aspect_mode:
            cw, ch, _ = _content_size(first_w, first_h, target_size, scale_mode)
            # 就是你直觉的那两步：先把内容按留边撑开，再把画布向上取整到 32 的倍数。
            # 档位 3/4：四边各 margin；档位 5：四边分别 m_left/m_top/m_right/m_bottom；
            # 档位 1/2：margin=0 ⇒ 等价于只补右/下。
            if pad_mode == cls.PAD_CUSTOM:
                canvas_w = -(-(cw + m_left + m_right) // multiple) * multiple
                canvas_h = -(-(ch + m_top + m_bottom) // multiple) * multiple
            else:
                canvas_w = -(-(cw + 2 * margin) // multiple) * multiple
                canvas_h = -(-(ch + 2 * margin) // multiple) * multiple
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
            if pad_mode == cls.PAD_SQUARE_MARGIN:
                # 先把长边（= 画布边长）补齐到 32 的倍数，再四边各留 margin，最后做成 1:1；
                # 结果仍是 32 的倍数画布，但横向/纵向都出现 ≥ margin 的可裁剪余量 ——
                # 这是让 「自动对齐到参考图」能修掉左右/上下漂移的前提。
                canvas_side = -(-(target_size + 2 * margin) // 32) * 32
                print(
                    "[调整图像尺寸填充] 1:1（四周留边）：画布 %d×%d（内容 %d×%d，"
                    "四边余量 ≥ %dpx）" % (canvas_side, canvas_side, first_w, first_h, margin)
                )
            else:
                canvas_side = target_size
            canvas_w = canvas_h = canvas_side

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
                # 档位 5：按四边各自的值放（左/上余量 = 所填值）；
                # 档位 3：四边同值，内容贴左/上；
                # 档位 1/2：margin=0 ⇒ 只在右/下补边，内容原点固定 (0,0)。
                # 无论哪档，ceil32 的零头恒落在右/下 ⇒ 右/下余量 ≥ 所填值。
                if pad_mode == cls.PAD_CUSTOM:
                    pad_left, pad_top = m_left, m_top
                else:
                    pad_left = pad_top = margin
            else:
                # 档位 0 与 4：居中（档位 4 的画布已含 2*margin，故四边余量 ≥ margin）
                pad_left = (canvas_w - new_width) // 2
                pad_top = (canvas_h - new_height) // 2

            # 创建画布：填色走 RGB（与旧版行为一致），transparent 走 RGBA（补边 alpha=0），
            # reflect/edge 由内容自身生成补边（详见 _build_canvas）。
            padded_image = _build_canvas(resized_image, canvas_w, canvas_h,
                                         pad_left, pad_top, fill_mode, pad_color)
            processed_pil_images.append(padded_image)

            # 仅从第一张图记录 image_info（假设批次内所有图像尺寸一致）
            if image_info_out is None:
                pad_right = canvas_w - new_width - pad_left
                pad_bottom = canvas_h - new_height - pad_top
                # 6 元组：末尾两位是画布宽高（旧版是单一边长的 5 元组，下游已兼容两种）
                image_info_out = (pad_left, pad_top, pad_right, pad_bottom, canvas_w, canvas_h)
                # 把「四边余量 = 自动对齐能校正的最大漂移量」明确打出来：
                # 留边档位是内容贴左/上放置 + ceil32 零头全落右/下，所以左/上余量 = 你填的
                # edge_margin，而右/下会多出一截。不打印的话用户会误以为四周都一样宽。
                print(
                    "[调整图像尺寸填充] 画布 %dx%d，内容 %dx%d，四边余量 左%d 右%d 上%d 下%d，"
                    "补边 %s"
                    "（余量 =「自动对齐到参考图」在各方向能校正的最大漂移量；"
                    "漂移超出该方向余量时，超出的部分取不到画布外像素，会留在输出里）"
                    % (canvas_w, canvas_h, new_width, new_height,
                       pad_left, pad_right, pad_top, pad_bottom, fill_label)
                )

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
