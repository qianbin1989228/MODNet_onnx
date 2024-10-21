# MODNet_onnx
Windows下的快速精准的抠图算法（Python实现）

​
## 1.安装环境依赖
```bash
pip install opencv-python
pip install onnxruntime
pip install onnx==1.6.1
```
其中尤其需要注意第三个依赖库onnx，如果版本太高可能会出现dll运行错误。

## 2.运行
```bash
python infer.py
```
