import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_version,
    check_yaml,
    colorstr,
    file_size,
    get_default_args,
    print_args,
    url2file,
    yaml_save,
)
from utils.torch_utils import select_device, smart_inference_mode

MACOS = platform.system() == "Darwin"  # macOS environment


class iOSModel(torch.nn.Module):
    def __init__(self, model, im):
        """Initializes an iOS compatible model with normalization based on image dimensions."""
        super().__init__()
        b, c, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = model.nc  # number of classes
        if w == h:
            self.normalize = 1.0 / w
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)
            # np = model(im)[0].shape[1]  # number of points
            # self.normalize = torch.tensor([1. / w, 1. / h, 1. / w, 1. / h]).expand(np, 4)  # explicit (faster, larger)

    def forward(self, x):
        """Runs forward pass on the input tensor, returning class confidences and normalized coordinates."""
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)


def export_formats():
    """Returns a DataFrame of supported YOLOv5 model export formats and their properties."""
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlmodel", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def try_export(inner_func):
    """Decorator @try_export for YOLOv5 model export functions that logs success/failure, time taken, and file size."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Logs success/failure and execution details of model export functions wrapped with @try_export decorator."""
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            return None, None

    return outer_func




@try_export
def export_edgetpu(file, prefix=colorstr("Edge TPU:")):
    """
    Exports a YOLOv5 model to Edge TPU compatible TFLite format; requires Linux and Edge TPU compiler.

    https://coral.ai/docs/edgetpu/models-intro/
    """
    cmd = "edgetpu_compiler --version"
    help_url = "https://coral.ai/docs/edgetpu/compiler/"
    assert platform.system() == "Linux", f"export only supported on Linux. See {help_url}"
    if subprocess.run(f"{cmd} > /dev/null 2>&1", shell=True).returncode != 0:
        LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
        sudo = subprocess.run("sudo --version >/dev/null", shell=True).returncode == 0  # sudo installed on system
        for c in (
            "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
            'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
            "sudo apt-get update",
            "sudo apt-get install edgetpu-compiler",
        ):
            subprocess.run(c if sudo else c.replace("sudo ", ""), shell=True, check=True)
    ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

    LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
    f = str(file).replace(".pt", "-int8_edgetpu.tflite")  # Edge TPU model
    f_tfl = str(file).replace(".pt", "-int8.tflite")  # TFLite model

    subprocess.run(
        [
            "edgetpu_compiler",
            "-s",
            "-d",
            "-k",
            "10",
            "--out_dir",
            str(file.parent),
            f_tfl,
        ],
        check=True,
    )
    return f, None










def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 model export configurations, returning the parsed options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model.pt path(s)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640, 640], help="image (h, w)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLOv5 Detect() inplace=True")
    parser.add_argument("--keras", action="store_true", help="TF: use Keras")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF/OpenVINO INT8 quantization")
    parser.add_argument("--per-tensor", action="store_true", help="TF per-tensor quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="TF: add NMS to model")
    parser.add_argument("--agnostic-nms", action="store_true", help="TF: add agnostic NMS to model")
    parser.add_argument("--topk-per-class", type=int, default=100, help="TF.js NMS: topk per class to keep")
    parser.add_argument("--topk-all", type=int, default=100, help="TF.js NMS: topk for all classes to keep")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="TF.js NMS: IoU threshold")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="TF.js NMS: confidence threshold")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    parse_opt(opt)
