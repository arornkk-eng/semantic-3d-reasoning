# arXiv 风格 LaTeX 源码

## 文件

`main.tex`：双栏中文论文正文。

`references.bib`：BibTeX 参考文献。

## 编译

需要完整 TeX Live 或 Overleaf，使用 XeLaTeX：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

本机当前没有安装 LaTeX 工具，因此尚未执行 PDF 编译和版面检查。

## 提交前必须修改

1. 替换作者姓名、单位和邮箱。
2. 完成相关工作检索并核对全部 BibTeX 元数据。
3. 加入正式方法图、结果图和定量实验表。
4. 将初步功能结果替换为多场景精度实验。
5. 根据目标会议或期刊模板调整文档类。

## arXiv 上传

将 `main.tex` 与 `references.bib` 一起打包上传。若后续加入图片，确保图片使用相对路径并包含在压缩包中。编译器选择 XeLaTeX。
