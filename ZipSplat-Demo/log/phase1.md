ZipSplat-Demo 开发日志（Day 1）
日期： 2026-07-20
一、今日目标
完成 ZipSplat 项目的整体规划与第一阶段开发准备工作，明确项目定位、技术路线、开发规范以及工程架构，为后续开发打下基础。

二、项目定位
项目名称：ZipSplat Object Reconstruction Demo项目目标：构建一个基于 ZipSplat 的本地 3D 重建 Demo，实现用户上传多张物体照片后，通过 ZipSplat 自动生成 3D Gaussian Splatting 模型，并支持导出 PLY 文件及后续网页展示。当前版本定位：
科研 Demo
Windows 本地运行
单用户
不涉及训练，仅进行模型推理（Inference）未来规划：
Gradio Web Demo
AI 图片补充模块（可选）
Three.js 在线查看
FastAPI 接口
Docker 部署
支持更多 3D 重建模型

三、开发环境确认
系统：Windows 11开发工具：VS Code硬件配置：GPU：NVIDIA RTX 4050 Laptop GPU（6GB VRAM）CPU：Intel Core i7-14650内存：16GB软件环境：
Python 已安装
Git 已安装
NVIDIA 驱动正常
CUDA 环境正常

四、项目开发路线确定
采用敏捷开发方式，将项目划分为多个 Milestone。
Milestone	内容
0	开发环境搭建
1	跑通官方 ZipSplat Demo
2	封装 ZipSplat 推理模块
3	实现本地图片输入
4	导出 PLY 与自动生成旋转视频
5	开发 Gradio Web Demo
6+	增加 AI 图片补充、Three.js、FastAPI 等扩展功能

五、最终工程架构设计
采用官方源码与业务代码分离的方式。
工作区结构：
AI-Workspace/├── ZipSplat/ （官方源码）
└── ZipSplat-Demo/ （自主开发项目）
设计原则：
官方源码保持不修改。
所有业务逻辑放入 ZipSplat-Demo。
通过封装调用官方 API。
后续官方更新时无需修改业务代码。

六、ZipSplat-Demo 目录规划
ZipSplat-Demo/├── app/│ 
├── main.py│ 
├── config.py│ 
└── utils.py│
├── backend/│ 
├── image_loader.py│ 
├── zipsplat_engine.py│ 
├── renderer.py│ 
└── exporter.py│
├── data/│ 
├── input/│ 
├── output/│ 
└── cache/│
├── models/
├── ui/│ 
└── gradio_app.py│
├── docs/
├── requirements.txt
└── README.md

七、开发规范确定
项目遵循以下原则：
官方代码不修改。
所有功能采用模块化设计。
Controller 与算法解耦。
每完成一个 Milestone 进行一次验收。
每一步均保持项目可运行。
整体架构：
UI↓Controller↓Business Logic↓ZipSplat Engine↓Official ZipSplat

八、输入输出流程设计
第一阶段采用文件夹输入方式。
流程：
用户拍摄 5~10 张物体图片↓data/input/↓ZipSplat 推理↓生成 Gaussian Scene↓scene.ply↓自动生成旋转视频↓后续升级为网页上传

九、官方项目管理策略
决定采用方案 A：官方仓库与 Demo 工程完全分离。
原因：
易维护
易升级
避免修改官方源码
符合软件工程规范

十、源码学习计划
采用**"逆向工程式学习"**方法。
第一阶段：
理解项目结构↓跑通官方 Demo↓理解输入输出流程↓封装推理接口↓构建自己的业务层
重点关注四个模块：
模型（Model）
推理（Inference）
输入（Input）
输出（Output）

十一、当前完成情况
已完成：
✓ 项目定位
✓ 技术路线
✓ 开发规划
✓ 工程架构设计
✓ Windows 环境确认
✓ 官方 ZipSplat 下载
✓ 工作区设计
✓ 项目目录规划
✓ 后续开发路线确定
当前项目整体完成度：约15%

十二、下一步计划（Day 2）
Milestone 0：
检查官方源码目录结构。
安装项目依赖。
验证 PyTorch + CUDA。
跑通官方 ZipSplat Demo。
分析官方推理流程。
开始设计 zipsplat_engine.py 封装层。

今日总结
今天没有急于编写代码，而是优先完成了整个项目的架构设计和开发规划。确定了以"官方源码 + 自主业务工程"分离的开发模式，采用模块化、可扩展的软件工程思想组织项目，为后续实现 Gradio Web Demo、AI 图片补充和多模型扩展预留了良好的架构基础。相比直接运行 GitHub Demo，本项目更注重工程化、可维护性和后续扩展能力，目标不仅是完成一个科研演示，更是构建一个可持续迭代的 3D AI 重建平台。
以后我们每天都会维护两份文档：
DEV_LOG.md：记录每天的开发过程、问题和总结（开发日志）。
PROJECT_PLAN.md：记录整体架构、Milestone、模块设计和长期规划（项目规划）。
