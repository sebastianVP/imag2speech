from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

processor = TrOCRProcessor.from_pretrained("qantev/trocr-base-spanish")
model = VisionEncoderDecoderModel.from_pretrained("qantev/trocr-base-spanish")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"================{device}============")
model.to(device)

# =========================
# PREPROCESAMIENTO
# =========================
def preprocess_image(image):
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)
    return image

# =========================
# SEGMENTACIÓN POR VALLES
# =========================
def segment_lines_valleys(image, debug=False):
    gray = preprocess_image(image)
    img  = np.array(gray)
    height, width = img.shape

    # --- Binarización adaptativa (mejor para manuscrito con fondo no uniforme) ---
    block = max(11, (height // 20) | 1)   # debe ser impar
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, 10
    )

    # --- Limpiar ruido pequeño ---
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    # --- Proyección horizontal ---
    projection = np.sum(thresh, axis=1).astype(np.float32)

    # --- Suavizado: sigma más pequeño para preservar valles reales ---
    sigma = max(1, height // 100)
    proj_smooth = gaussian_filter1d(projection, sigma=sigma)

    # --- Estimar altura típica de línea ---
    # Buscamos picos (zonas con texto) para estimar el interlineado
    peak_thresh = np.max(proj_smooth) * 0.10
    peaks, props = find_peaks(
        proj_smooth,
        height=peak_thresh,
        distance=max(5, height // 20)   # mínima distancia entre picos
    )

    if len(peaks) < 2:
        # Fallback: segmentación simple por umbral
        return _segment_simple(img, thresh, proj_smooth, height, debug, image)

    # Distancia media entre picos consecutivos = interlineado estimado
    inter_line = int(np.median(np.diff(peaks)))

    # Altura mínima de un valle para considerarlo separador entre líneas
    # Un valle real baja bastante respecto a los picos vecinos
    valley_min_drop = 0.35   # el valle debe ser < 35% del promedio de sus picos vecinos

    # --- Encontrar valles entre picos consecutivos ---
    separators = []   # filas donde se corta
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        segment = proj_smooth[p1:p2]
        valley_idx = np.argmin(segment) + p1
        valley_val = proj_smooth[valley_idx]
        avg_peaks  = (proj_smooth[p1] + proj_smooth[p2]) / 2

        if valley_val < avg_peaks * valley_min_drop:
            # Valle profundo → separador real entre líneas
            separators.append(valley_idx)
        else:
            # Valle poco profundo → posiblemente el mismo bloque de texto
            # Intentar buscar el mínimo local con ventana más fina
            fine = proj_smooth[p1:p2]
            fine_min_idx = np.argmin(fine) + p1
            # Forzar corte si la distancia entre picos es > 60% del interlineado
            if (p2 - p1) > inter_line * 0.6:
                separators.append(fine_min_idx)

    # --- Construir límites de línea a partir de los separadores ---
    # Añadir bordes superior e inferior
    boundaries = [0] + separators + [height]

    pad = max(3, height // 80)
    line_imgs   = []
    line_bounds = []

    for i in range(len(boundaries) - 1):
        y1 = int(boundaries[i])
        y2 = int(boundaries[i + 1])

        # Ignorar segmentos vacíos o muy pequeños
        seg_proj = proj_smooth[y1:y2]
        if seg_proj.max() < np.max(proj_smooth) * 0.05:
            continue
        if y2 - y1 < max(6, height // 40):
            continue

        y1p = max(0, y1 - pad)
        y2p = min(height, y2 + pad)
        crop = img[y1p:y2p, :]

        line_imgs.append(Image.fromarray(crop).convert("RGB"))
        line_bounds.append((y1p, y2p))

    # --- Debug ---
    debug_img  = None
    debug_plot = None
    if debug:
        debug_pil = image.convert("RGB").copy()
        draw = ImageDraw.Draw(debug_pil)
        colors = ["red", "blue", "green", "orange", "purple", "cyan"]
        for idx, (y1, y2) in enumerate(line_bounds):
            draw.rectangle([0, y1, width - 1, y2],
                           outline=colors[idx % len(colors)], width=3)
        debug_img = debug_pil

        fig, ax = plt.subplots(figsize=(6, max(3, height / 80)))
        ax.plot(proj_smooth, range(len(proj_smooth)), color="steelblue", lw=1.2)
        for p in peaks:
            ax.axhline(y=p, color="orange", linestyle=":", lw=1, alpha=0.7)
        for s in separators:
            ax.axhline(y=s, color="red", linestyle="--", lw=1.5,
                       label="corte" if s == separators[0] else "")
        for (y1, y2) in line_bounds:
            ax.axhspan(y1, y2, alpha=0.15, color="green")
        ax.set_xlabel("Densidad píxeles")
        ax.set_ylabel("Fila")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.set_title(f"Proyección — {len(line_bounds)} líneas — interlineado≈{inter_line}px")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        debug_plot = Image.open(buf).copy()

    return line_imgs, line_bounds, debug_img, debug_plot


def _segment_simple(img, thresh, proj_smooth, height, debug, orig_image):
    """Fallback para imágenes con muy pocas líneas o texto simple."""
    threshold = np.max(proj_smooth) * 0.08
    pad = max(3, height // 60)
    min_h = max(6, height // 50)
    line_imgs, line_bounds = [], []
    in_line, start = False, 0
    width = img.shape[1]

    for i, v in enumerate(proj_smooth):
        if v > threshold and not in_line:
            start = i; in_line = True
        elif v <= threshold and in_line:
            if i - start > min_h:
                y1 = max(0, start - pad)
                y2 = min(height, i + pad)
                line_imgs.append(Image.fromarray(img[y1:y2]).convert("RGB"))
                line_bounds.append((y1, y2))
            in_line = False
    if in_line and height - start > min_h:
        y1 = max(0, start - pad)
        line_imgs.append(Image.fromarray(img[y1:height]).convert("RGB"))
        line_bounds.append((y1, height))

    debug_img = debug_plot = None
    if debug:
        pil = orig_image.convert("RGB").copy()
        draw = ImageDraw.Draw(pil)
        colors = ["red","blue","green","orange"]
        for i,(y1,y2) in enumerate(line_bounds):
            draw.rectangle([0,y1,width-1,y2], outline=colors[i%len(colors)], width=3)
        debug_img = pil

    return line_imgs, line_bounds, debug_img, debug_plot

# =========================
# OCR POR LÍNEA
# =========================
def trocr_line(line_img):
    with torch.no_grad():
        pixels = processor(line_img, return_tensors="pt").pixel_values.to(device)
        ids    = model.generate(pixels, max_new_tokens=80)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

# =========================
# PROCESO PRINCIPAL
# =========================
def process(image):
    line_imgs, bounds, debug_img, debug_plot = segment_lines_valleys(image, debug=True)

    if not line_imgs:
        return [], "No se detectaron líneas.", debug_img, debug_plot

    texts, gallery = [], []
    for i, li in enumerate(line_imgs, 1):
        txt = trocr_line(li)
        texts.append(f"[L{i}] {txt}")
        gallery.append((li, f"Línea {i}: {txt}"))

    return gallery, "\n".join(texts), debug_img, debug_plot

# =========================
# INTERFAZ
# =========================
with gr.Blocks(title="OCR línea por línea") as iface:
    gr.Markdown("## OCR línea por línea — Segmentación por valles")
    img_input = gr.Image(type="pil", label="Imagen de entrada")

    with gr.Row():
        gallery_out = gr.Gallery(label="Líneas detectadas", columns=2)
        text_out    = gr.Textbox(label="Texto reconocido", lines=10)

    gr.Markdown("### 🔍 Debug")
    with gr.Row():
        debug_img_out  = gr.Image(label="Líneas marcadas")
        debug_plot_out = gr.Image(label="Proyección horizontal")

    gr.Button("Procesar").click(
        fn=process,
        inputs=img_input,
        outputs=[gallery_out, text_out, debug_img_out, debug_plot_out]
    )
    img_input.change(
        fn=process,
        inputs=img_input,
        outputs=[gallery_out, text_out, debug_img_out, debug_plot_out]
    )

iface.launch(debug=True)