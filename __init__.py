# -*- coding: utf-8 -*-
"""Qwen21 No-Offset Editing — Qwen Image 2.1 无偏移编辑节点包。

改进自 xingyuezhiyuan/Comfyui-txtnode，仅保留无偏移编辑管线所需的三个节点：
调整图像尺寸填充（增强版）、移除图像填充、自动对齐到参考图（本项目新增）。
"""
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes import ResizeAndPadNode, RemovePadFromImageNode, AutoAlignToReferenceNode

# 前端扩展：让「调整图像尺寸填充」的留边控件跟着「画布模式」启用/置灰。
WEB_DIRECTORY = "./web"


class NoOffsetEditingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ResizeAndPadNode,
            RemovePadFromImageNode,
            AutoAlignToReferenceNode,
        ]


async def comfy_entrypoint() -> NoOffsetEditingExtension:
    return NoOffsetEditingExtension()


__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
