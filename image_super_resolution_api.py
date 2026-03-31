import os
import argparse
import torch
from fastapi import FastAPI, File, UploadFile, Response, Query
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from io import BytesIO
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# 初始化 FastAPI 应用
app = FastAPI(
    title="Real-ESRGAN 超分接口（CPU版）",
    description="支持 PNG/JPG/TIF 等格式，对齐命令行参数：RealESRGAN_x4plus_anime_6B + outscale=4 + tile=128",
    version="1.0",
)

# 允许跨域（前端/其他服务调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- 设备选择（自动检测 GPU，可通过环境变量覆盖）----------------------
# 环境变量 REAL_ESRGAN_DEVICE 可设为 "cpu" 或 "cuda" 来强制指定设备
_env_device = os.environ.get("REAL_ESRGAN_DEVICE", "").lower()
if _env_device == "cpu":
    DEVICE = "cpu"
elif _env_device == "cuda":
    DEVICE = "cuda"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HALF = DEVICE.startswith("cuda")  # GPU 模式下使用 fp16 加速

# ---------------------- 模型初始化（修改模型路径为相对路径）----------------------
# 获取项目根目录（image_super_resolution_api.py 所在的目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型权重路径：拼接项目根目录和 Real-ESRGAN/weights/ 下的模型文件
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "Real-ESRGAN", "weights", "RealESRGAN_x4plus_anime_6B.pth"
)

# 检查模型文件是否存在（避免路径错误导致启动失败）
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"模型权重文件不存在：{MODEL_PATH}\n"
        f"请确认项目目录结构为：\n"
        f"your_project/\n"
        f"├── image_super_resolution_api.py\n"
        f"└── Real-ESRGAN/\n"
        f"    └── weights/\n"
        f"        └── RealESRGAN_x4plus_anime_6B.pth"
    )

# 模型配置：和你命令行 -n RealESRGAN_x4plus_anime_6B 一致
model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
)

# 初始化超分器（对齐命令行参数：--outscale 4、-t 128）
upsampler = RealESRGANer(
    scale=4,  # 对应 --outscale 4
    model_path=MODEL_PATH,  # 使用修改后的模型路径
    model=model,
    tile=128,  # 对应 -t 128（你原来的参数）
    tile_pad=10,
    pre_pad=0,
    half=HALF,
    device=DEVICE,
    alpha_upsampler="realesrgan",
)


# ---------------------- 超分接口（核心功能，无需修改）----------------------
@app.post("/super_resolve", summary="图片超分处理")
async def super_resolve(
    file: UploadFile = File(
        ..., description="上传需要超分的图片（支持 PNG/JPG/TIF 等格式）"
    ),
    outscale: float = Query(
        4.0, description="超分放大倍数（默认 4.0，和你命令行 --outscale 一致）"
    ),
    tile: int = Query(
        128, description="分块大小（默认 128，和你命令行 -t 一致，CPU 建议 64-128）"
    ),
    output_format: str = Query(
        "auto", description="输出格式（auto=和输入一致，可选 png/jpg/tif）"
    ),
):
    try:
        # 1. 读取上传的图片二进制流，转化为 OpenCV 可处理的数组
        contents = await file.read()  # 读取二进制流
        nparr = np.frombuffer(contents, np.uint8)  # 二进制流 → numpy 数组
        img = cv2.imdecode(
            nparr, cv2.IMREAD_UNCHANGED
        )  # 解码为图片数组（支持透明通道/多通道）

        if img is None:
            return {"error": "无法解析图片，请上传有效格式（PNG/JPG/TIF 等）"}, 400

        # 2. 调用 Real-ESRGAN 超分（对齐你原来的命令行逻辑）
        # 动态调整 tile 参数（如果接口传入和初始化不同）
        upsampler.tile = tile
        restored_img, _ = upsampler.enhance(img, outscale=outscale)

        # 3. 确定输出格式（优先用户指定，否则和输入一致）
        if output_format == "auto":
            # 从文件名后缀获取输入格式
            input_suffix = file.filename.split(".")[-1].lower()
            output_format = (
                input_suffix
                if input_suffix in ["png", "jpg", "jpeg", "tif", "tiff"]
                else "png"
            )
        else:
            output_format = output_format.lower()
            if output_format not in ["png", "jpg", "jpeg", "tif", "tiff"]:
                output_format = "png"  # 非法格式默认用 PNG

        # 4. 将处理后的图片编码为二进制流（保持格式无损）
        # JPG 格式需要指定压缩质量（默认 95，避免过度压缩）
        encode_params = []
        if output_format in ["jpg", "jpeg"]:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif output_format == "png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 0=无压缩（无损）
        elif output_format in ["tif", "tiff"]:
            encode_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]  # 1=无压缩

        is_success, buffer = cv2.imencode(
            f".{output_format}", restored_img, encode_params
        )
        if not is_success:
            return {"error": "图片编码失败"}, 500

        # 5. 返回二进制流图片（前端可直接下载或显示）
        return Response(
            content=buffer.tobytes(),
            media_type=f"image/{output_format}",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.splitext(file.filename)[0]}_sr.{output_format}",
                "X-Image-Format": output_format,
                "X-Upscale-Factor": str(outscale),
            },
        )

    except Exception as e:
        return {"error": f"超分处理失败：{str(e)}"}, 500


# ---------------------- 启动服务（和你本地环境兼容）----------------------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Real-ESRGAN 超分接口服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="启用热更新（开发模式）")
    args = parser.parse_args()

    print(f"设备: {DEVICE} | 半精度: {HALF} | CUDA可用: {torch.cuda.is_available()}")
    if DEVICE.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    uvicorn.run(
        app="image_super_resolution_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
