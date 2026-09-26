/**
 * Qwen21 No-Offset Editing —— 「调整图像尺寸填充」控件联动。
 *
 * 画布模式决定哪些留边控件真正在生效：
 *   32 的倍数（四周留边） / 1:1（四周留边） → 只有「四周留边」生效
 *   32 的倍数（四边自定义）                  → 只有四个「X 边留边」生效
 *   其余档位，以及关闭「启用尺寸填充」时     → 留边控件全部不生效
 *
 * 不生效的控件会被置灰并锁定（改不动），生效的那个保持可编辑 —— 它就是你改数值的地方，
 * 这样既不会误改，也一眼能看出「现在该调哪个」。
 *
 * 依赖：ComfyUI 前端把 widget.disabled 计入 computedDisabled（本机实测 1.52.x 生效）。
 *       DOM 类名 .qno-widget-locked 是第二层保险（canvas 绘制的老控件拿不到 element 时，
 *       就只剩 disabled 一层）。
 */
import { app } from "../../scripts/app.js";

const NODE_ID = "ResizeAndPadNode";

const MODE_MARGIN = "32 的倍数（四周留边）";
const MODE_SQUARE_MARGIN = "1:1（四周留边）";
const MODE_CUSTOM = "32 的倍数（四边自定义）";

const EDGE_WIDGET = "edge_margin";
const SIDE_WIDGETS = ["margin_left", "margin_top", "margin_right", "margin_bottom"];

const TIP_EDGE_OFF =
    "当前画布模式不使用「四周留边」。\n" +
    "它只在「32 的倍数（四周留边）」和「1:1（四周留边）」两档生效 —— 切到那两档即可修改。";
const TIP_SIDE_OFF =
    "当前画布模式不使用四边留边。\n" +
    "它只在「32 的倍数（四边自定义）」档生效 —— 切到该档即可逐边修改。";
const TIP_BYPASS =
    "「启用尺寸填充」已关闭，本节点当前旁路（原样输出），留边参数不生效。";

let styleInjected = false;

function injectStyle() {
    if (styleInjected) return;
    styleInjected = true;
    const el = document.createElement("style");
    el.textContent =
        ".qno-widget-locked{opacity:.4 !important;filter:grayscale(1) !important;}" +
        ".qno-widget-locked *{cursor:not-allowed !important;}";
    document.head.appendChild(el);
}

function findWidget(node, name) {
    return node && node.widgets ? node.widgets.find((w) => w.name === name) : undefined;
}

function domOf(widget) {
    if (!widget) return null;
    for (const c of [widget.element, widget.inputEl, widget.domElement]) {
        if (c && c.classList && typeof c.classList.toggle === "function") return c;
    }
    return null;
}

/** 置灰 + 锁定一个控件；返回是否发生了变化（用于决定要不要重绘画布）。 */
function setLocked(widget, locked, tip) {
    if (!widget) return false;
    let changed = false;

    if (widget.__qnoLocked !== locked) {
        widget.__qnoLocked = locked;
        // 第一层：ComfyUI 自己的禁用位（节点控件是 canvas 绘制的，这是唯一带视觉效果的层）
        widget.disabled = locked;
        // 第二层：万一某个控件是 DOM 渲染的，顺手把原生 disabled 和类名也打上
        const el = domOf(widget);
        if (el) {
            el.classList.toggle("qno-widget-locked", locked);
            if ("disabled" in el) el.disabled = locked;
        }
        changed = true;
    }

    // tooltip 与锁定状态同步。注意锁定原因本身也会变（例如从「档位不用」变成「整节点旁路」），
    // 所以锁着的时候也要跟着刷新，否则会留着上一个原因。
    if (locked) {
        if (widget.__qnoTip === undefined) widget.__qnoTip = widget.tooltip;
        if (widget.tooltip !== tip) {
            widget.tooltip = tip;
            changed = true;
        }
    } else if (widget.__qnoTip !== undefined) {
        widget.tooltip = widget.__qnoTip;
        widget.__qnoTip = undefined;
        changed = true;
    }
    return changed;
}

function applyMode(node) {
    if (!node || !node.widgets) return;
    const modeWidget = findWidget(node, "pad_mode");
    const bypass = findWidget(node, "resize_and_pad")?.value === false;
    const mode = modeWidget ? modeWidget.value : undefined;

    const edgeOn = !bypass && (mode === MODE_MARGIN || mode === MODE_SQUARE_MARGIN);
    const sideOn = !bypass && mode === MODE_CUSTOM;

    let changed = setLocked(findWidget(node, EDGE_WIDGET), !edgeOn, bypass ? TIP_BYPASS : TIP_EDGE_OFF);
    for (const name of SIDE_WIDGETS) {
        changed = setLocked(findWidget(node, name), !sideOn, bypass ? TIP_BYPASS : TIP_SIDE_OFF) || changed;
    }
    if (changed) {
        try { node.setDirtyCanvas?.(true, true); } catch (e) { /* 忽略 */ }
    }
}

function bindNode(node) {
    if (!node) return;
    injectStyle();
    if (!node.__qnoBound) {
        node.__qnoBound = true;
        for (const name of ["pad_mode", "resize_and_pad"]) {
            const w = findWidget(node, name);
            if (!w || w.__qnoHooked) continue;
            const orig = w.callback;
            w.callback = function (value, ...rest) {
                const out = orig ? orig.apply(this, [value, ...rest]) : undefined;
                try { applyMode(node); } catch (e) { console.error("[Qwen21NoOffset]", e); }
                return out;
            };
            w.__qnoHooked = true;
        }
    }
    applyMode(node);
}

app.registerExtension({
    name: "Qwen21NoOffset.ResizeAndPadWidgets",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== NODE_ID) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            try { bindNode(this); } catch (e) { console.error("[Qwen21NoOffset] onNodeCreated", e); }
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            try { bindNode(this); } catch (e) { console.error("[Qwen21NoOffset] onConfigure", e); }
            return r;
        };
    },

    // 兜底：combo 的取值路径在不同前端版本里未必都走 widget.callback。
    // applyMode 带幂等守卫，没变化时直接返回，轮询代价极低。
    setup() {
        setInterval(() => {
            const nodes = app?.graph?._nodes;
            if (!Array.isArray(nodes)) return;
            for (const n of nodes) {
                if (n && n.type === NODE_ID) applyMode(n);
            }
        }, 400);
    },
});
