# l1bsr_sr_gui_onnx.py
#
# Description:
# PyQt5 GUI for L1BSR super-resolution using the ONNX Runtime engine.
# This version does NOT require PyTorch, making it much more lightweight.
#
# Instructions:
# 1. Make sure you have the required libraries:
#    pip install onnxruntime pyqt5 numpy pillow rasterio
# 2. Run the `convert_to_onnx.py` script first to generate 'rcan_model.onnx'.
# 3. Place 'rcan_model.onnx' in the same directory as this script.
# 4. Run this application:
#    python l1bsr_sr_gui_onnx.py

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import onnxruntime as ort

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLineEdit,
    QMessageBox, QFrame, QScrollArea, QFormLayout, QCheckBox, QSlider
)

import rasterio
from rasterio.transform import Affine
from PIL import Image, ImageFilter

# Note: The entire PyTorch model definition (RCAN, etc.) has been removed.

# ------------------------
# I/O & Geo helpers (Unchanged)
# ------------------------

@dataclass
class BandData:
    path: str
    arr: np.ndarray
    transform: Affine
    crs: any
    width: int
    height: int

def read_band(path: str) -> BandData:
    with rasterio.open(path) as src:
        arr = src.read(1)
        if arr.dtype != np.uint16:
            warnings.warn(f"{os.path.basename(path)} is {arr.dtype}, casting to uint16.")
            arr = arr.astype(np.uint16)
        return BandData(path=path, arr=arr, transform=src.transform, crs=src.crs, width=src.width, height=src.height)

def check_alignment(bands: Dict[str, BandData]) -> Tuple[bool, Optional[str]]:
    keys = ["B02","B03","B04","B08"]
    ref = bands[keys[0]]
    for k in keys[1:]:
        b = bands[k]
        if (b.width != ref.width) or (b.height != ref.height): return False, f"Size mismatch: {k}={b.width}x{b.height} vs B02={ref.width}x{ref.height}"
        if (b.transform != ref.transform): return False, f"GeoTransform mismatch between B02 and {k}."
        if (b.crs != ref.crs): return False, f"CRS mismatch between B02 and {k}."
    return True, None

def stack_bgrn(b02: BandData, b03: BandData, b04: BandData, b08: BandData) -> np.ndarray:
    h, w = b02.arr.shape
    out = np.zeros((h, w, 4), dtype=np.uint16)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = b02.arr, b03.arr, b04.arr, b08.arr
    return out

def write_geotiff(out_path: str, data_u16_hwc: np.ndarray, ref: BandData):
    h2, w2, c = data_u16_hwc.shape
    new_transform = ref.transform * Affine.scale(0.5, 0.5)
    profile = {"driver": "GTiff", "height": h2, "width": w2, "count": 4, "dtype": rasterio.uint16, "crs": ref.crs, "transform": new_transform, "compress": "lzw", "tiled": True, "interleave": "pixel"}
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(4): dst.write(data_u16_hwc[..., i], i+1)

def percentile_stretch(arr: np.ndarray, p_low=2.0, p_high=98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        out = np.zeros(arr.shape, dtype=np.uint8)
        for i in range(arr.shape[-1]):
            vmin, vmax = np.percentile(arr[..., i], [p_low, p_high])
            vmax = vmax if vmax > vmin else vmin + 1e-3
            out[..., i] = np.clip((arr[..., i] - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
        return out
    else: # Grayscale
        vmin, vmax = np.percentile(arr, [p_low, p_high])
        vmax = vmax if vmax > vmin else vmin + 1e-3
        return np.clip((arr - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)

def hwc_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img.ndim == 2: h, w = img.shape; qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else: h, w, c = img.shape; qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def save_jpeg_with_jgw(out_jpg: str, rgb_u8: np.ndarray, ref: BandData):
    new_transform = ref.transform * Affine.scale(0.5, 0.5)
    Image.fromarray(rgb_u8, mode="RGB").save(out_jpg, quality=95, subsampling=0)
    a, b, c, d, e, f = new_transform.a, new_transform.b, new_transform.c, new_transform.d, new_transform.e, new_transform.f
    x_center, y_center = rasterio.transform.xy(new_transform, 0, 0, offset='center')
    world = [a, d, b, e, x_center, y_center]
    jgw_path = os.path.splitext(out_jpg)[0] + ".jgw"
    with open(jgw_path, "w") as f:
        for v in world: f.write(f"{v:.12f}\n")

# ------------------------
# ONNX Inference Wrapper (Replaces L1BSRSR)
# ------------------------

def to_numpy_4ch(img_bgrn_u16: np.ndarray) -> np.ndarray:
    """HxWx4 uint16 -> 1x4xHxW float32 normalized numpy array."""
    # Permute HWC to CHW, add batch dim, convert to float32, and normalize
    img_fp32 = img_bgrn_u16.astype(np.float32)
    img_chw = np.transpose(img_fp32, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch / 400.0

#def from_numpy_to_u16(sr_batch: np.ndarray) -> np.ndarray:
#    """1x4xHxW numpy array -> HxWx4 uint16."""
#    sr_np = (sr_batch * 400.0).astype(np.uint16)
#    sr_chw = sr_np[0] # Remove batch dimension
#    sr_hwc = np.moveaxis(sr_chw, 0, -1) # CHW -> HWC
#    return sr_hwc

def from_numpy_to_u16(sr_batch: np.ndarray) -> np.ndarray:
    """1x4xHxW numpy array -> HxWx4 uint16."""
    # De-normalize first
    sr_denormalized = sr_batch * 400.0
    
    # --- ADD THIS LINE ---
    # Clip the values to the valid range of uint16 to prevent wrap-around.
    np.clip(sr_denormalized, 0, 65535, out=sr_denormalized)
    
    # Now, safely cast to uint16
    sr_np = sr_denormalized.astype(np.uint16)
    
    sr_chw = sr_np[0] # Remove batch dimension
    sr_hwc = np.moveaxis(sr_chw, 0, -1) # CHW -> HWC
    return sr_hwc

class ONNXModel:
    def __init__(self, onnx_path: str):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # For GPU, use: providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        # The list order defines priority. It will fall back to CPU if CUDA is not available.
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def super_resolve(self, img_bgrn_u16: np.ndarray) -> np.ndarray:
        """Runs super-resolution on a HxWx4 uint16 NumPy array."""
        input_data = to_numpy_4ch(img_bgrn_u16)
        
        # Run inference
        result = self.session.run(None, {self.input_name: input_data})
        
        # The result is a list of outputs, we only have one
        output_data = result[0]
        
        return from_numpy_to_u16(output_data)

# ------------------------
# GUI (Mostly unchanged, but simplified)
# ------------------------

class ImageLabel(QLabel):
    def __init__(self, title="Preview", size=(320, 320)):
        super().__init__(title)
        self.setFixedSize(size[0], size[1])
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setAlignment(Qt.AlignCenter)
        self.setText(title)

    def set_image(self, pix: Optional[QPixmap]):
        if pix is None: self.setText("No preview")
        else: self.setPixmap(pix.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class InteractiveImageLabel(QLabel):
    def __init__(self, title="Preview"):
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setText(title)
        self.pix_before: Optional[QPixmap] = None
        self.pix_after: Optional[QPixmap] = None

    def set_images(self, before: Optional[QPixmap], after: Optional[QPixmap]):
        self.pix_before, self.pix_after = before, after
        self.set_display_image(self.pix_after)

    def set_display_image(self, pix: Optional[QPixmap]):
        if pix is None: self.setText("No Image")
        else: self.setPixmap(pix); self.setFixedSize(pix.size())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pix_before: self.set_display_image(self.pix_before)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.pix_after: self.set_display_image(self.pix_after)
        super().mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("L1BSR Super-Resolution (ONNX Runtime)")
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
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("rcan_model.onnx")
        self.btn_browse_model = QPushButton("Browse…")
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(self.btn_browse_model)
        model_layout.addRow("ONNX Model:", model_row)
        root.addWidget(model_box)

        input_box = QGroupBox("Inputs (GeoTIFF 16-bit, WGS84, ≤ ~500×500 px)")
        grid = QGridLayout(input_box)
        self.lab_b02, self.lab_b03, self.lab_b04, self.lab_b08 = ImageLabel("B02 (Blue)"), ImageLabel("B03 (Green)"), ImageLabel("B04 (Red)"), ImageLabel("B08 (NIR)")
        self.btn_b02, self.btn_b03, self.btn_b04, self.btn_b08 = QPushButton("Select B02…"), QPushButton("Select B03…"), QPushButton("Select B04…"), QPushButton("Select B08…")
        grid.addWidget(self.lab_b02, 0, 0); grid.addWidget(self.lab_b03, 0, 1); grid.addWidget(self.lab_b04, 0, 2); grid.addWidget(self.lab_b08, 0, 3)
        grid.addWidget(self.btn_b02, 1, 0); grid.addWidget(self.btn_b03, 1, 1); grid.addWidget(self.btn_b04, 1, 2); grid.addWidget(self.btn_b08, 1, 3)
        root.addWidget(input_box)

        self.btn_process = QPushButton("Process (2× Super-Resolution)")
        root.addWidget(self.btn_process)

        out_box = QGroupBox("Output Preview (RGB) - Click and hold to see 'Before'")
        out_layout = QVBoxLayout(out_box)
        self.lab_out = InteractiveImageLabel("SR RGB")
        scroll_area = QScrollArea(); scroll_area.setWidget(self.lab_out); scroll_area.setWidgetResizable(True); scroll_area.setAlignment(Qt.AlignCenter)
        out_layout.addWidget(scroll_area)
        
        save_row = QHBoxLayout()
        self.chk_sharpen = QCheckBox("Sharpen (JPG/Preview only)")
        self.slider_sharpen = QSlider(Qt.Horizontal); self.slider_sharpen.setRange(0, 300); self.slider_sharpen.setValue(150); self.slider_sharpen.setEnabled(False)
        self.lbl_sharpen_val = QLabel(f"{self.slider_sharpen.value()}%"); self.lbl_sharpen_val.setFixedWidth(40)
        save_row.addWidget(self.chk_sharpen); save_row.addWidget(self.slider_sharpen); save_row.addWidget(self.lbl_sharpen_val); save_row.addStretch(1)
        self.btn_save_tif = QPushButton("Save as GeoTIFF…"); self.btn_save_jpg = QPushButton("Save as JPG + JGW…")
        save_row.addWidget(self.btn_save_tif); save_row.addWidget(self.btn_save_jpg)
        out_layout.addLayout(save_row)
        root.addWidget(out_box)

        self.btn_browse_model.clicked.connect(self.choose_model)
        self.btn_b02.clicked.connect(lambda: self.choose_band("B02")); self.btn_b03.clicked.connect(lambda: self.choose_band("B03")); self.btn_b04.clicked.connect(lambda: self.choose_band("B04")); self.btn_b08.clicked.connect(lambda: self.choose_band("B08"))
        self.btn_process.clicked.connect(self.on_process)
        self.btn_save_tif.clicked.connect(self.on_save_tif); self.btn_save_jpg.clicked.connect(self.on_save_jpg)
        self.chk_sharpen.toggled.connect(self.apply_and_update_preview); self.slider_sharpen.valueChanged.connect(self.apply_and_update_preview)

        default_model = "rcan_model.onnx"
        if os.path.isfile(default_model): self.model_edit.setText(default_model)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Model", "", "ONNX Model (*.onnx);;All Files (*)")
        if path: self.model_edit.setText(path)

    def choose_band(self, key: str):
        path, _ = QFileDialog.getOpenFileName(self, f"Select {key} GeoTIFF", "", "GeoTIFF (*.tif *.tiff);;All Files (*)")
        if not path: return
        try: bd = read_band(path)
        except Exception as e: QMessageBox.critical(self, "Read error", f"Failed to read {key}:\n{e}"); return
        self.paths[key], self.bands[key] = path, bd
        pix = hwc_to_qpixmap(percentile_stretch(bd.arr))
        if key == "B02": self.lab_b02.set_image(pix)
        elif key == "B03": self.lab_b03.set_image(pix)
        elif key == "B04": self.lab_b04.set_image(pix)
        elif key == "B08": self.lab_b08.set_image(pix)

    def ensure_all_inputs(self) -> bool:
        missing = [k for k, v in self.bands.items() if v is None]
        if missing: QMessageBox.warning(self, "Missing input", f"Please select all bands: missing {', '.join(missing)}"); return False
        ok, reason = check_alignment(self.bands)
        if not ok: QMessageBox.critical(self, "Alignment error", reason); return False
        self.ref_band = self.bands["B02"]; return True

    def on_process(self):
        if not self.ensure_all_inputs(): return
        model_path = self.model_edit.text().strip()
        if not model_path: QMessageBox.warning(self, "Model", "Please select the ONNX model file."); return
        QApplication.setOverrideCursor(Qt.WaitCursor); self.btn_process.setEnabled(False)
        try:
            rgb_before_u16 = np.stack([self.bands["B04"].arr, self.bands["B03"].arr, self.bands["B02"].arr], axis=-1)
            rgb_before_u8 = percentile_stretch(rgb_before_u16)
            h, w, _ = rgb_before_u8.shape
            self.pixmap_before = hwc_to_qpixmap(np.array(Image.fromarray(rgb_before_u8).resize((w*2, h*2), Image.NEAREST)))

            img_bgrn = stack_bgrn(self.bands["B02"], self.bands["B03"], self.bands["B04"], self.bands["B08"])
            
            engine = ONNXModel(onnx_path=model_path)
            sr_u16 = engine.super_resolve(img_bgrn)
            self.result_u16_hwc = sr_u16
            
            self.rgb_after_u8_base = percentile_stretch(np.stack([sr_u16[...,2], sr_u16[...,1], sr_u16[...,0]], axis=-1))
            
            self.apply_and_update_preview()
            QMessageBox.information(self, "Done", "Super-resolution completed.")
        except Exception as e: QMessageBox.critical(self, "Processing error", f"{e}")
        finally: QApplication.restoreOverrideCursor(); self.btn_process.setEnabled(True)

    def apply_and_update_preview(self):
        if self.rgb_after_u8_base is None: return
        is_sharpening = self.chk_sharpen.isChecked()
        self.slider_sharpen.setEnabled(is_sharpening)
        if is_sharpening:
            strength = self.slider_sharpen.value()
            self.lbl_sharpen_val.setText(f"{strength}%")
            sharpened_pil = Image.fromarray(self.rgb_after_u8_base).filter(ImageFilter.UnsharpMask(radius=2, percent=strength, threshold=3))
            rgb_to_display = np.array(sharpened_pil)
        else:
            self.lbl_sharpen_val.setText(""); rgb_to_display = self.rgb_after_u8_base
        self.lab_out.set_images(self.pixmap_before, hwc_to_qpixmap(rgb_to_display))

    def on_save_tif(self):
        if self.result_u16_hwc is None or self.ref_band is None: QMessageBox.warning(self, "No result", "Please run processing first."); return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save GeoTIFF (unsharpened)", "output_sr.tif", "GeoTIFF (*.tif *.tiff)")
        if not out_path: return
        try: write_geotiff(out_path, self.result_u16_hwc, self.ref_band); QMessageBox.information(self, "Saved", f"Saved unsharpened 16-bit GeoTIFF:\n{out_path}")
        except Exception as e: QMessageBox.critical(self, "Save error", f"Failed to save GeoTIFF:\n{e}")

    def on_save_jpg(self):
        if self.result_u16_hwc is None or self.ref_band is None: QMessageBox.warning(self, "No result", "Please run processing first."); return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save JPEG (+JGW)", "output_sr.jpg", "JPEG (*.jpg *.jpeg)")
        if not out_path: return
        try:
            rgb_u8 = percentile_stretch(np.stack([self.result_u16_hwc[...,2], self.result_u16_hwc[...,1], self.result_u16_hwc[...,0]], axis=-1))
            if self.chk_sharpen.isChecked():
                strength = self.slider_sharpen.value()
                rgb_u8 = np.array(Image.fromarray(rgb_u8).filter(ImageFilter.UnsharpMask(radius=2, percent=strength, threshold=3)))
            save_jpeg_with_jgw(out_path, rgb_u8, self.ref_band)
            jgw_path = os.path.splitext(out_path)[0] + '.jgw'
            QMessageBox.information(self, "Saved", f"Saved JPEG and JGW:\n{out_path}\n{jgw_path}")
        except Exception as e: QMessageBox.critical(self, "Save error", f"Failed to save JPEG/JGW:\n{e}")

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()