# l1bsr_sr_gui_updated.py
# PyQt5 GUI for L1BSR super-resolution (REC/RCAN only), taking separate GeoTIFFs for S2 B02,B03,B04,B08
# Saves 2x output as GeoTIFF (same CRS) or JPEG + JGW.
# Includes interactive before/after preview, scrollable pane, and optional sharpening for visual outputs.
#
# Dependencies: pyqt5, torch, safetensors, rasterio, pillow, numpy
# Optional: CUDA-capable PyTorch for GPU inference.

import os
import sys
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file as load_safetensors

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QComboBox, QLineEdit,
    QMessageBox, QFrame, QScrollArea, QFormLayout, QCheckBox, QSlider
)

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from PIL import Image, ImageFilter

# ------------------------
# Model (RCAN, 4-channel)
# ------------------------

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)
    ):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x) + x

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True))
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x) + x

class RCAN(nn.Module):
    def __init__(self, n_colors, conv=default_conv):
        super(RCAN, self).__init__()
        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 2

        modules_head = [conv(n_colors, n_feats, kernel_size)]
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

# ------------------------
# I/O & Geo helpers
# ------------------------

@dataclass
class BandData:
    path: str
    arr: np.ndarray            # HxW, uint16
    transform: Affine
    crs: any
    width: int
    height: int

def read_band(path: str) -> BandData:
    with rasterio.open(path) as src:
        arr = src.read(1)
        if arr.dtype != np.uint16:
            warnings.warn(f"{os.path.basename(path)} is {arr.dtype}, casting to uint16 for model input.")
            arr = arr.astype(np.uint16)
        return BandData(
            path=path,
            arr=arr,
            transform=src.transform,
            crs=src.crs,
            width=src.width,
            height=src.height
        )

def check_alignment(bands: Dict[str, BandData]) -> Tuple[bool, Optional[str]]:
    keys = ["B02","B03","B04","B08"]
    ref = bands[keys[0]]
    for k in keys[1:]:
        b = bands[k]
        if (b.width != ref.width) or (b.height != ref.height):
            return False, f"Size mismatch: {k}={b.width}x{b.height} vs B02={ref.width}x{ref.height}"
        if (b.transform != ref.transform):
            return False, f"GeoTransform mismatch between B02 and {k}."
        if (b.crs != ref.crs):
            return False, f"CRS mismatch between B02 and {k}."
    return True, None

def stack_bgrn(b02: BandData, b03: BandData, b04: BandData, b08: BandData) -> np.ndarray:
    h, w = b02.arr.shape
    out = np.zeros((h, w, 4), dtype=np.uint16)
    out[..., 0] = b02.arr
    out[..., 1] = b03.arr
    out[..., 2] = b04.arr
    out[..., 3] = b08.arr
    return out

def to_torch_4ch(img_bgrn_u16: np.ndarray, device: torch.device) -> torch.Tensor:
    ten = torch.from_numpy(img_bgrn_u16.astype(np.float32)).permute(2,0,1)[None]
    return ten.to(device) / 400.0

def from_torch_to_u16(sr: torch.Tensor) -> np.ndarray:
    sr_np = (sr.detach().cpu().numpy() * 400.0).astype(np.uint16)
    sr_np = np.moveaxis(sr_np[0], 0, -1)
    return sr_np

def write_geotiff(out_path: str, data_u16_hwc: np.ndarray, ref: BandData):
    h2, w2, c = data_u16_hwc.shape
    assert c == 4
    new_transform = ref.transform * Affine.scale(0.5, 0.5)
    profile = {
        "driver": "GTiff", "height": h2, "width": w2, "count": 4,
        "dtype": rasterio.uint16, "crs": ref.crs, "transform": new_transform,
        "compress": "lzw", "tiled": True, "interleave": "pixel"
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(4):
            dst.write(data_u16_hwc[..., i], i+1)

def percentile_stretch(arr: np.ndarray, p_low=2.0, p_high=98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        vmin, vmax = np.percentile(arr, [p_low, p_high])
        vmax = vmax if vmax > vmin else vmin + 1e-3
        out = np.clip((arr - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
        return out
    elif arr.ndim == 3:
        out = np.zeros(arr.shape, dtype=np.uint8)
        for i in range(arr.shape[-1]):
            vmin, vmax = np.percentile(arr[..., i], [p_low, p_high])
            vmax = vmax if vmax > vmin else vmin + 1e-3
            out[..., i] = np.clip((arr[..., i] - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
        return out
    else:
        raise ValueError("percentile_stretch expects 2D or HxWxC array.")

def hwc_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, c = img.shape
        assert c == 3
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def save_jpeg_with_jgw(out_jpg: str, rgb_u8: np.ndarray, ref: BandData):
    new_transform = ref.transform * Affine.scale(0.5, 0.5)
    Image.fromarray(rgb_u8, mode="RGB").save(out_jpg, quality=100, subsampling=0)
    a, b, c, d, e, f = new_transform.a, new_transform.b, new_transform.c, new_transform.d, new_transform.e, new_transform.f
    x_center, y_center = rasterio.transform.xy(new_transform, 0, 0, offset='center')
    world = [a, d, b, e, x_center, y_center]
    jgw_path = os.path.splitext(out_jpg)[0] + ".jgw"
    with open(jgw_path, "w") as f:
        for v in world:
            f.write(f"{v:.12f}\n")

# ------------------------
# Inference wrapper
# ------------------------

class L1BSRSR:
    def __init__(self, weights_path: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = RCAN(n_colors=4).to(self.device).eval()
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Model file not found: {weights_path}")
        state = load_safetensors(weights_path, device="cpu")
        self.model.load_state_dict(state, strict=False)
        torch.set_grad_enabled(False)

    @torch.inference_mode()
    def super_resolve(self, img_bgrn_u16: np.ndarray) -> np.ndarray:
        ten = to_torch_4ch(img_bgrn_u16, self.device)
        sr = self.model(ten)
        out = from_torch_to_u16(sr)
        return out

# ------------------------
# GUI
# ------------------------

class ImageLabel(QLabel):
    def __init__(self, title="Preview", size=(320, 320)):
        super().__init__(title)
        self.setFixedSize(size[0], size[1])
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setAlignment(Qt.AlignCenter)
        self.setText(title)

    def set_image(self, pix: Optional[QPixmap]):
        if pix is None:
            self.setText("No preview")
        else:
            scaled = pix.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)

class InteractiveImageLabel(QLabel):
    def __init__(self, title="Preview"):
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setText(title)
        self.pix_before: Optional[QPixmap] = None
        self.pix_after: Optional[QPixmap] = None

    def set_images(self, before: Optional[QPixmap], after: Optional[QPixmap]):
        self.pix_before = before
        self.pix_after = after
        self.set_display_image(self.pix_after)

    def set_display_image(self, pix: Optional[QPixmap]):
        if pix is None:
            self.setText("No Image")
        else:
            self.setPixmap(pix)
            self.setFixedSize(pix.size())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pix_before:
            self.set_display_image(self.pix_before)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.pix_after:
            self.set_display_image(self.pix_after)
        super().mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("L1BSR Super-Resolution (BGRN from separate S2 bands)")
        self.resize(1300, 900)

        self.paths: Dict[str, Optional[str]] = {"B02": None, "B03": None, "B04": None, "B08": None}
        self.bands: Dict[str, Optional[BandData]] = {"B02": None, "B03": None, "B04": None, "B08": None}
        self.ref_band: Optional[BandData] = None
        self.result_u16_hwc: Optional[np.ndarray] = None
        self.pixmap_before: Optional[QPixmap] = None
        self.rgb_after_u8_base: Optional[np.ndarray] = None

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        model_box = QGroupBox("Model")
        model_layout = QFormLayout(model_box)
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("REC_Real_L1B.safetensors")
        self.btn_browse_weights = QPushButton("Browse…")
        self.device_combo = QComboBox()
        devs = ["cpu"]
        if torch.cuda.is_available():
            devs.insert(0, "cuda")
        self.device_combo.addItems(devs)
        weights_row = QHBoxLayout()
        weights_row.addWidget(self.weights_edit, 1)
        weights_row.addWidget(self.btn_browse_weights)
        model_layout.addRow("Weights:", weights_row)
        model_layout.addRow("Device:", self.device_combo)
        root.addWidget(model_box)

        input_box = QGroupBox("Inputs (GeoTIFF 16-bit, WGS84, ≤ ~500×500 px)")
        grid = QGridLayout(input_box)
        self.lab_b02 = ImageLabel("B02 (Blue)")
        self.lab_b03 = ImageLabel("B03 (Green)")
        self.lab_b04 = ImageLabel("B04 (Red)")
        self.lab_b08 = ImageLabel("B08 (NIR)")
        self.btn_b02 = QPushButton("Select B02…")
        self.btn_b03 = QPushButton("Select B03…")
        self.btn_b04 = QPushButton("Select B04…")
        self.btn_b08 = QPushButton("Select B08…")
        grid.addWidget(self.lab_b02, 0, 0); grid.addWidget(self.lab_b03, 0, 1)
        grid.addWidget(self.lab_b04, 0, 2); grid.addWidget(self.lab_b08, 0, 3)
        grid.addWidget(self.btn_b02, 1, 0); grid.addWidget(self.btn_b03, 1, 1)
        grid.addWidget(self.btn_b04, 1, 2); grid.addWidget(self.btn_b08, 1, 3)
        root.addWidget(input_box)

        self.btn_process = QPushButton("Process (2× Super-Resolution)")
        root.addWidget(self.btn_process)

        out_box = QGroupBox("Output Preview (RGB) - Click and hold to see 'Before'")
        out_layout = QVBoxLayout(out_box)
        
        self.lab_out = InteractiveImageLabel("SR RGB")
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.lab_out)
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        out_layout.addWidget(scroll_area)
        
        # --- Sharpening and Save controls ---
        save_row = QHBoxLayout()
        self.chk_sharpen = QCheckBox("Sharpen (JPG/Preview only)")
        self.slider_sharpen = QSlider(Qt.Horizontal)
        self.slider_sharpen.setRange(0, 300) # Corresponds to 'percent' in UnsharpMask
        self.slider_sharpen.setValue(150)
        self.slider_sharpen.setEnabled(False)
        self.lbl_sharpen_val = QLabel(f"{self.slider_sharpen.value()}%")
        self.lbl_sharpen_val.setFixedWidth(40)
        
        save_row.addWidget(self.chk_sharpen)
        save_row.addWidget(self.slider_sharpen)
        save_row.addWidget(self.lbl_sharpen_val)
        save_row.addStretch(1)
        
        self.btn_save_tif = QPushButton("Save as GeoTIFF…")
        self.btn_save_jpg = QPushButton("Save as JPG + JGW…")
        save_row.addWidget(self.btn_save_tif)
        save_row.addWidget(self.btn_save_jpg)
        out_layout.addLayout(save_row)
        root.addWidget(out_box)

        # --- Connections ---
        self.btn_browse_weights.clicked.connect(self.choose_weights)
        self.btn_b02.clicked.connect(lambda: self.choose_band("B02"))
        self.btn_b03.clicked.connect(lambda: self.choose_band("B03"))
        self.btn_b04.clicked.connect(lambda: self.choose_band("B04"))
        self.btn_b08.clicked.connect(lambda: self.choose_band("B08"))
        self.btn_process.clicked.connect(self.on_process)
        self.btn_save_tif.clicked.connect(self.on_save_tif)
        self.btn_save_jpg.clicked.connect(self.on_save_jpg)
        self.chk_sharpen.toggled.connect(self.apply_and_update_preview)
        self.slider_sharpen.valueChanged.connect(self.apply_and_update_preview)

        default_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models", "REC_Real_L1B.safetensors")
        if os.path.isfile(default_weights):
            self.weights_edit.setText(default_weights)

    def choose_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select REC_Real_L1B.safetensors", "", "SafeTensors (*.safetensors);;All Files (*)")
        if path:
            self.weights_edit.setText(path)

    def choose_band(self, key: str):
        path, _ = QFileDialog.getOpenFileName(self, f"Select {key} GeoTIFF", "", "GeoTIFF (*.tif *.tiff);;All Files (*)")
        if not path: return
        try:
            bd = read_band(path)
        except Exception as e:
            QMessageBox.critical(self, "Read error", f"Failed to read {key}:\n{e}")
            return
        self.paths[key] = path
        self.bands[key] = bd
        prev = percentile_stretch(bd.arr)
        pix = hwc_to_qpixmap(prev)
        if key == "B02": self.lab_b02.set_image(pix)
        elif key == "B03": self.lab_b03.set_image(pix)
        elif key == "B04": self.lab_b04.set_image(pix)
        elif key == "B08": self.lab_b08.set_image(pix)

    def ensure_all_inputs(self) -> bool:
        missing = [k for k, v in self.bands.items() if v is None]
        if missing:
            QMessageBox.warning(self, "Missing input", f"Please select all bands: missing {', '.join(missing)}")
            return False
        ok, reason = check_alignment(self.bands)
        if not ok:
            QMessageBox.critical(self, "Alignment error", reason)
            return False
        self.ref_band = self.bands["B02"]
        return True

    def on_process(self):
        if not self.ensure_all_inputs(): return
        weights = self.weights_edit.text().strip()
        if not weights:
            QMessageBox.warning(self, "Weights", "Please select the model file REC_Real_L1B.safetensors.")
            return
        device = self.device_combo.currentText()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.btn_process.setEnabled(False)
        try:
            rgb_before_u16 = np.stack([self.bands["B04"].arr, self.bands["B03"].arr, self.bands["B02"].arr], axis=-1)
            rgb_before_u8 = percentile_stretch(rgb_before_u16)
            h, w, _ = rgb_before_u8.shape
            pil_before = Image.fromarray(rgb_before_u8).resize((w*2, h*2), Image.NEAREST)
            self.pixmap_before = hwc_to_qpixmap(np.array(pil_before))

            img_bgrn = stack_bgrn(self.bands["B02"], self.bands["B03"], self.bands["B04"], self.bands["B08"])
            
            engine = L1BSRSR(weights_path=weights, device=device)
            sr_u16 = engine.super_resolve(img_bgrn)
            self.result_u16_hwc = sr_u16
            
            self.rgb_after_u8_base = percentile_stretch(np.stack([sr_u16[...,2], sr_u16[...,1], sr_u16[...,0]], axis=-1))
            
            self.apply_and_update_preview()
            QMessageBox.information(self, "Done", "Super-resolution completed.")
        except Exception as e:
            QMessageBox.critical(self, "Processing error", f"{e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_process.setEnabled(True)

    def apply_and_update_preview(self):
        if self.rgb_after_u8_base is None:
            return

        is_sharpening = self.chk_sharpen.isChecked()
        self.slider_sharpen.setEnabled(is_sharpening)
        
        if is_sharpening:
            strength = self.slider_sharpen.value()
            self.lbl_sharpen_val.setText(f"{strength}%")
            
            pil_img = Image.fromarray(self.rgb_after_u8_base)
            # UnsharpMask params: radius, percent, threshold
            sharpened_pil = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=strength, threshold=3))
            rgb_to_display = np.array(sharpened_pil)
        else:
            self.lbl_sharpen_val.setText("")
            rgb_to_display = self.rgb_after_u8_base

        pixmap_after = hwc_to_qpixmap(rgb_to_display)
        self.lab_out.set_images(self.pixmap_before, pixmap_after)

    def on_save_tif(self):
        if self.result_u16_hwc is None or self.ref_band is None:
            QMessageBox.warning(self, "No result", "Please run processing first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save GeoTIFF (unsharpened)", "output_sr.tif", "GeoTIFF (*.tif *.tiff)")
        if not out_path: return
        try:
            write_geotiff(out_path, self.result_u16_hwc, self.ref_band)
            QMessageBox.information(self, "Saved", f"Saved unsharpened 16-bit GeoTIFF:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save GeoTIFF:\n{e}")

    def on_save_jpg(self):
        if self.result_u16_hwc is None or self.ref_band is None:
            QMessageBox.warning(self, "No result", "Please run processing first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save JPEG (+JGW)", "output_sr.jpg", "JPEG (*.jpg *.jpeg)")
        if not out_path: return
        try:
            # Generate the final 8-bit RGB image to be saved
            rgb_u8 = percentile_stretch(np.stack([
                self.result_u16_hwc[...,2],
                self.result_u16_hwc[...,1],
                self.result_u16_hwc[...,0],
            ], axis=-1))

            # Apply sharpening if enabled
            if self.chk_sharpen.isChecked():
                strength = self.slider_sharpen.value()
                pil_img = Image.fromarray(rgb_u8)
                sharpened_pil = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=strength, threshold=3))
                rgb_u8 = np.array(sharpened_pil)

            save_jpeg_with_jgw(out_path, rgb_u8, self.ref_band)
            jgw_path = os.path.splitext(out_path)[0] + '.jgw'
            QMessageBox.information(self, "Saved", f"Saved JPEG and JGW:\n{out_path}\n{jgw_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save JPEG/JGW:\n{e}")

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()