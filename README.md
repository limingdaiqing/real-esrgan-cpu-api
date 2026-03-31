# Suzhou Super Resolution

基于 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 的图像超分辨率 API 服务。

将 Real-ESRGAN 封装为 RESTful 接口，支持 **CPU / GPU 自动切换**，可通过 HTTP 调用实现图片超分放大，适用于需要集成超分能力的业务系统。

## 解决什么问题

- **无 GPU 也能跑**：支持纯 CPU 推理，无需 CUDA 环境，普通服务器即可部署
- **接口化调用**：无需命令行操作，业务系统通过 HTTP POST 即可完成超分
- **灵活参数**：支持自定义放大倍数、分块大小、输出格式等

## 项目结构

```
Suzhou_Super_Resolution/
├── image_super_resolution_api.py    # FastAPI 接口服务（主入口）
├── environment.yml                   # Conda 环境配置
├── .gitignore
└── Real-ESRGAN/                      # Real-ESRGAN 源码
    ├── inference_realesrgan.py       # 官方 GPU 推理脚本
    ├── inference_realesrgan_cpu.py   # CPU 推理脚本（适配无 GPU 环境）
    ├── realesrgan/                   # 核心推理库
    └── weights/                      # 模型权重目录（需手动下载）
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/limingdaiqing/Suzhou_Super_Resolution.git
cd Suzhou_Super_Resolution
```

### 2. 下载模型权重

模型文件未包含在仓库中（体积较大），需手动下载：

```bash
# 创建权重目录
mkdir -p Real-ESRGAN/weights

# 下载默认使用的 Anime 模型（18MB，推荐）
curl -L -o Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
```

其他可选模型（按需下载）：

| 模型 | 大小 | 适用场景 | 下载链接 |
|------|------|---------|---------|
| RealESRGAN_x4plus_anime_6B | 18MB | 动漫/插画图片（默认） | [下载](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) |
| RealESRGAN_x4plus | 64MB | 通用真实图片 | [下载](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| RealESRGAN_x2plus | 64MB | 通用图片（2x 放大） | [下载](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |
| realesr-general-x4v3 | 轻量 | 通用图片（轻量级） | [下载](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) |

下载后放入 `Real-ESRGAN/weights/` 目录即可。

### 3. 创建环境

使用 Conda（推荐）：

```bash
conda env create -f environment.yml
conda activate realesrgan
```

或使用 pip：

```bash
pip install fastapi uvicorn python-multipart opencv-python numpy pillow torch basicsr realesrgan
```

### 4. 启动服务

```bash
# 自动检测 GPU（有 GPU 用 GPU，没有用 CPU）
python image_super_resolution_api.py

# 开发模式（启用热更新）
python image_super_resolution_api.py --reload

# 自定义端口
python image_super_resolution_api.py --port 9000

# 强制使用 CPU
set REAL_ESRGAN_DEVICE=cpu    # Windows
export REAL_ESRGAN_DEVICE=cpu # Linux/Mac
python image_super_resolution_api.py

# 强制使用 GPU
set REAL_ESRGAN_DEVICE=cuda    # Windows
export REAL_ESRGAN_DEVICE=cuda # Linux/Mac
python image_super_resolution_api.py
```

启动成功后会输出设备信息：

```
设备: cuda | 半精度: True | CUDA可用: True
GPU: NVIDIA GeForce RTX 4090
```

## API 文档

启动后访问 `http://localhost:8000/docs` 可查看自动生成的 Swagger UI 文档。

### POST /super_resolve

图片超分辨率处理接口。

**请求参数：**

| 参数 | 类型 | 位置 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | form-data | 必填 | 需要超分的图片（支持 PNG/JPG/TIF） |
| `outscale` | float | query | 4.0 | 放大倍数 |
| `tile` | int | query | 128 | 分块大小（CPU 建议 64-128，GPU 可设更大） |
| `output_format` | string | query | auto | 输出格式：auto/png/jpg/tif |

**调用示例（curl）：**

```bash
# 基础用法：4x 放大
curl -X POST http://localhost:8000/super_resolve \
  -F "file=@input.png" \
  --output output.png

# 指定 2x 放大，输出 JPG
curl -X POST "http://localhost:8000/super_resolve?outscale=2&output_format=jpg" \
  -F "file=@input.jpg" \
  --output output.jpg
```

**调用示例（Python）：**

```python
import requests

with open("input.png", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/super_resolve",
        files={"file": f},
        params={"outscale": 4, "output_format": "png"}
    )

with open("output.png", "wb") as f:
    f.write(resp.content)
```

**调用示例（JavaScript）：**

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const resp = await fetch("http://localhost:8000/super_resolve?outscale=4", {
  method: "POST",
  body: formData,
});

const blob = await resp.blob();
```

**响应：**

- 成功：返回超分后的图片二进制流（`Content-Type: image/png` 等）
- 响应头包含 `X-Image-Format`（输出格式）和 `X-Upscale-Factor`（放大倍数）

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `REAL_ESRGAN_DEVICE` | 强制指定设备：`cpu` 或 `cuda`。不设置则自动检测 | 自动检测 |

## 性能参考

CPU 模式下使用 `tile=128` 处理一张 512x512 图片约需 10-30 秒（取决于 CPU 性能）。GPU 模式下通常在 1-3 秒内完成。

如果 CPU 模式下内存不足，可减小 `tile` 参数（如 64）；如果内存充裕，可增大 `tile`（如 256）以提升速度。

## 切换模型

默认使用 `RealESRGAN_x4plus_anime_6B`（动漫优化模型）。如需切换为通用模型，修改 `image_super_resolution_api.py` 中的以下部分：

```python
# 模型路径
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "Real-ESRGAN", "weights", "RealESRGAN_x4plus.pth"  # 改为通用模型
)

# 对应的模型结构
model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4  # 23 blocks
)
```

## 技术栈

- **Web 框架**：FastAPI + Uvicorn
- **超分模型**：Real-ESRGAN（RRDBNet 架构）
- **深度学习**：PyTorch
- **图像处理**：OpenCV

## 致谢

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 腾讯 ARC 实验室开发的图像超分辨率模型

## License

本项目仅供学习和研究使用。Real-ESRGAN 遵循其原始 [BSD-3-Clause License](Real-ESRGAN/LICENSE)。
