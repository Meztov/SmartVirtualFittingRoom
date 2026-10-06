# -*- coding: utf-8 -*-
"""
SMART FITTING ROOM V4.9.1 - SEAM SAFE TEXTURE

ARQUITECTURA
============
1. SegFormer detecta Upper-clothes y crea un KEYFRAME.
2. La malla del keyframe queda CONGELADA como geometria de referencia.
3. MediaPipe Pose calcula cada frame la posicion ABSOLUTA de cada vertice
   desde ese keyframe. No se integra movimiento frame a frame.
4. Lucas-Kanade NO mueve la malla directamente:
      - mide movimiento local;
      - se resta el movimiento que Pose ya explica;
      - queda un RESIDUO pequeno de tela/pliegue;
      - ese residuo se limita y se aplica sobre la posicion absoluta.
5. Los puntos LK se eliminan si salen de la ROI de la prenda y se renuevan
   con frecuencia. Por tanto pueden desaparecer sin arrastrar la malla.

OBJETIVO
========
Eliminar el drift acumulativo:
    MAL:  vertice_n = vertice_(n-1) + desplazamiento
    BIEN: vertice_n = deformar(vertice_KEYFRAME, pose_actual) + residuo_LK

Controles
=========
H = mostrar/ocultar Pose
G = mostrar/ocultar malla
F = mostrar/ocultar puntos LK
M = mostrar/ocultar mascara
T = mostrar/ocultar shirt.png
O = mostrar/ocultar overlay
R = crear NUEVO keyframe mediante segmentacion
C = perder keyframe y readquirir automaticamente
P = pausar/reanudar adquisicion automatica
Q / ESC = salir
"""

import os
import time
import math

import cv2
import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
except ImportError as exc:
    print("\nFALTAN DEPENDENCIAS DE SEGMENTACION.")
    print("Ejecuta dentro de tu .venv:")
    print("  python -m pip install torch torchvision transformers pillow")
    print("\nError original:", exc)
    raise

try:
    import mediapipe as mp
except ImportError as exc:
    print("\nFALTA MEDIAPIPE.")
    print("Ejecuta dentro de tu .venv:")
    print("  python -m pip install mediapipe")
    print("\nError original:", exc)
    raise


# ============================================================
# CONFIGURACION
# ============================================================

WINDOW_NAME = "Smart Fitting Room V4.9.1 - SEAM SAFE TEXTURE"

CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MODEL_NAME = "mattmdjaga/segformer_b2_clothes"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POSE_MODEL_PATH = os.path.join("models", "pose_landmarker.task")

AUTO_START_DELAY = 4.0
AUTO_RETRY_SECONDS = 2.0

# ----------------------------
# Segmentacion / mascara
# ----------------------------
MIN_MASK_AREA_RATIO = 0.012
MAX_MASK_AREA_RATIO = 0.50
MIN_BBOX_WIDTH_RATIO = 0.10
MIN_BBOX_HEIGHT_RATIO = 0.10

MORPH_KERNEL = 5
MIN_COMPONENT_AREA_RATIO = 0.006
MASK_DILATE_PIXELS = 3

# ----------------------------
# Pose
# ----------------------------
POSE_MIN_VISIBILITY = 0.40
POSE_SMOOTHING = 0.62

# Cuantos frames consecutivos sin estructura antes de readquirir.
POSE_LOST_FRAMES_TO_RESEGMENT = 18

# ----------------------------
# Malla anclada
# ----------------------------
MESH_SPACING = 32

# V4.2: la silueta ya NO se genera con convexHull.
# Muestreamos el contorno real y triangulamos SOLO el interior de la mascara.
CONTOUR_SAMPLE_DISTANCE = 16.0
TRIANGLE_MIN_MASK_OVERLAP = 0.78
TRIANGLE_MIN_AREA = 8.0
TRIANGLE_MAX_EDGE_FACTOR = 2.8
BOUNDARY_REGULARIZATION = 0.008

# V4.3 ANTI-BALLOON:
# Los huesos del brazo rotan la manga, pero NO tienen permiso de escalarla
# libremente cuando MediaPipe cambia la longitud aparente hombro-codo.
ARM_SCALE_MIN = 0.985
ARM_SCALE_MAX = 1.015
TORSO_SCALE_MIN = 0.94
TORSO_SCALE_MAX = 1.06

# Un punto debe estar realmente hacia FUERA del hombro para considerarse manga.
SLEEVE_OUTWARD_THRESHOLD = -0.04
SLEEVE_ARM_DISTANCE_FACTOR = 0.48
SLEEVE_ARM_DOMINANCE = 0.86
TORSO_ARM_MAX_INFLUENCE = 0.025

# Suavizado SOLO espacial del resultado del frame actual.
# No mezcla con el frame anterior, por lo que no crea drift.
MESH_SPATIAL_REGULARIZATION = 0.055
MESH_SPATIAL_ITERS = 1

# ----------------------------
# V4.4 - CLOTH CONSTRAINT SOLVER
# ----------------------------
# La pose propone la posicion. Estas restricciones impiden que la tela
# convierta una arista/triangulo normal del keyframe en una "vela".
CLOTH_SOLVER_ITERS = 7

# Limites de longitud respecto al KEYFRAME.
# Torso mas rigido; manga permite algo mas porque rota y se pliega.
CLOTH_TORSO_EDGE_MIN_RATIO = 0.88
CLOTH_TORSO_EDGE_MAX_RATIO = 1.12
CLOTH_SLEEVE_EDGE_MIN_RATIO = 0.82
CLOTH_SLEEVE_EDGE_MAX_RATIO = 1.20

# Cuanto corregimos una violacion por iteracion.
CLOTH_EDGE_STIFFNESS = 0.72

# Conservacion de area por triangulo.
CLOTH_AREA_MIN_RATIO = 0.58
CLOTH_AREA_MAX_RATIO = 1.55
CLOTH_AREA_STIFFNESS = 0.34

# El solver no puede alejar indefinidamente un vertice de la posicion
# ABSOLUTA propuesta por Pose + LK.
CLOTH_MAX_CORRECTION_FROM_POSE = 24.0

# Seguridad de render: aunque MediaPipe tenga un frame raro, un triangulo
# grotescamente estirado NO se pinta como una gran mancha.
RENDER_TRIANGLE_MAX_AREA_RATIO = 1.85
RENDER_EDGE_MAX_RATIO = 1.48

# ----------------------------
# V4.5 - MANGAS ARTICULADAS
# ----------------------------
# Una manga deja de ser una coleccion de vertices independientes.
# Su movimiento principal es una transformacion rigida alrededor del hombro.
ARTICULATED_SLEEVE_MIN_ARM_WEIGHT = 0.45
ARTICULATED_SLEEVE_FULL_ARM_WEIGHT = 0.78

# Practicamente rigida: rota mucho, pero casi no cambia de tamano.
ARTICULATED_ARM_SCALE_MIN = 0.995
ARTICULATED_ARM_SCALE_MAX = 1.005

# En la costura hombro/manga mezclamos torso y brazo para evitar una grieta.
SLEEVE_MIN_RIGID_BLEND = 0.42
SLEEVE_MAX_RIGID_BLEND = 0.96

# LK puede corregir pliegues, pero no debe redibujar una manga completa.
SLEEVE_LK_RESIDUAL_FACTOR = 0.35

# Costura manga-torso: mucho mas estricta que una arista normal.
SEAM_EDGE_MIN_RATIO = 0.90
SEAM_EDGE_MAX_RATIO = 1.10
SEAM_EDGE_STIFFNESS = 0.88

# En una arista de costura, movemos principalmente la manga.
# El torso actua como ancla.
SEAM_SLEEVE_CORRECTION_SHARE = 0.82

# ----------------------------
# V4.6 - GARMENT EASE / HOLGURA
# ----------------------------
# El esqueleto es una ARMADURA interna. NO define el ancho de la playera.
TORSO_WIDTH_EASE = 1.12
TORSO_VERTICAL_EASE = 1.01

# La manga conserva volumen alrededor del eje hombro->codo.
SLEEVE_WIDTH_EASE = 1.16
SLEEVE_LENGTH_EASE = 1.035

# Pequeño espacio adicional hacia afuera en hombros/mangas.
SHOULDER_OUTWARD_EASE = 0.035

# Protege la zona central del cuello.
NECK_CENTER_PROTECTION = 0.22

# ----------------------------
# V4.7 - TOPOLOGY LOCK + SHAPE MATCHING
# ----------------------------
# Ya no ocultamos triangulos individualmente cuando se deforman.
# Primero los CORREGIMOS para mantener una tela continua.

TRI_SHAPE_SOLVER_ITERS = 5

# Fuerza normal de recuperacion de forma del triangulo.
TRI_SHAPE_STIFFNESS = 0.34

# Si un triangulo intenta invertirse, la correccion es mucho mas fuerte.
TRI_INVERSION_STIFFNESS = 0.92

# La forma puede respirar un poco, pero no crecer como una vela.
TRI_SHAPE_SCALE_MIN = 0.90
TRI_SHAPE_SCALE_MAX = 1.10

# Area firmada minima respecto al keyframe.
# Evita que un triangulo colapse casi a una linea.
TRI_SIGNED_AREA_MIN_RATIO = 0.24

# Limite adicional de correccion respecto a Pose+LK.
TOPOLOGY_MAX_CORRECTION_FROM_POSE = 30.0

# ----------------------------
# V4.8 - REGION AWARE CLOTH
# ----------------------------
# Regiones de triangulo:
#   0  = torso
#  -1  = manga izquierda
#   1  = manga derecha
#   2  = costura / axila / transicion
REGION_TORSO = 0
REGION_LEFT_SLEEVE = -1
REGION_RIGHT_SLEEVE = 1
REGION_SEAM = 2

# Shape matching: fuerte dentro de cada pieza, NO en la bisagra.
TORSO_SHAPE_STIFFNESS = 0.38
SLEEVE_SHAPE_STIFFNESS = 0.30
SEAM_SHAPE_STIFFNESS = 0.0

# Las axilas necesitan plegarse al levantar el brazo.
SEAM_RAISED_MIN_RATIO = 0.62
SEAM_RAISED_MAX_RATIO = 1.30
SEAM_RAISED_STIFFNESS = 0.30

# Con brazos abajo mantenemos una costura mas firme.
SEAM_RELAXED_MIN_RATIO = 0.88
SEAM_RELAXED_MAX_RATIO = 1.12
SEAM_RELAXED_STIFFNESS = 0.72

# Area por region.
TORSO_AREA_MIN_RATIO = 0.62
TORSO_AREA_MAX_RATIO = 1.45
TORSO_AREA_STIFFNESS = 0.30

SLEEVE_AREA_MIN_RATIO = 0.48
SLEEVE_AREA_MAX_RATIO = 1.70
SLEEVE_AREA_STIFFNESS = 0.22

# La costura no intenta conservar area: es la bisagra de la prenda.
# Solo aristas + Pose evitan que se vaya al infinito.

# LK pierde autoridad en la manga cuanto mas se levanta el brazo.
SLEEVE_LK_RELAXED_FACTOR = 0.32
SLEEVE_LK_RAISED_FACTOR = 0.07

# El solver no debe transmitir una correccion de una manga al torso entero.
SEAM_TORSO_CORRECTION_MAX_SHARE = 0.08

# ----------------------------
# Lucas-Kanade
# ----------------------------
LK_MAX_CORNERS = 180
LK_QUALITY_LEVEL = 0.005
LK_MIN_DISTANCE = 6
LK_BLOCK_SIZE = 7

LK_WIN_SIZE = (27, 27)
LK_MAX_LEVEL = 3
LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    25,
    0.01,
)

LK_FB_MAX_ERROR = 2.2

# LK es opcional. Incluso cero puntos es legal.
LK_REFRESH_EVERY = 10

# Si un punto se aleja de la ROI predicha, se descarta.
LK_ROI_MARGIN = 18

# ----------------------------
# Residuo LK
# ----------------------------
# Un residuo grande significa que LK probablemente se equivoco.
LK_RESIDUAL_HARD_REJECT = 12.0

# Incluso un residuo valido nunca puede corregir mas de esto por frame.
LK_RESIDUAL_MAX = 5.0

# Influencia espacial del residuo sobre vertices cercanos.
LK_RESIDUAL_SIGMA = 68.0
LK_RESIDUAL_NEIGHBORS = 7

# Maxima autoridad de LK sobre la geometria.
LK_RESIDUAL_GAIN = 0.72

# ----------------------------
# Visual
# ----------------------------
OVERLAY_ALPHA = 0.30
MASK_COLOR = (255, 180, 40)

# ----------------------------
# V4.9 - TEXTURED GARMENT
# ----------------------------
GARMENT_TEXTURE_PATH = os.path.join("garments", "shirt.png")
GARMENT_TEXTURE_OPACITY = 0.96

# False: la silueta final la decide la malla V4.8.
# Es lo mas robusto cuando el PNG tiene transparencia con una silueta
# diferente a la playera detectada sobre la persona.
TEXTURE_USE_SOURCE_ALPHA = False

# Suavizado del borde exterior final.
TEXTURE_EDGE_BLUR = 3

# Margen de seguridad dentro del PNG.
TEXTURE_SOURCE_MARGIN_X = 0.015
TEXTURE_SOURCE_MARGIN_Y = 0.010

# V4.9.1 - raster seam protection.
# Cada triangulo se expande SOLO en rasterizacion; la geometria real no cambia.
TEXTURE_TRIANGLE_OVERLAP_PX = 1

# Microhuecos internos maximos que pueden aparecer por redondeo subpixel.
TEXTURE_HOLE_CLOSE_KERNEL = 3

# Inpaint SOLO para pixeles de la silueta que no recibieron ninguna muestra
# de textura. No se aplica sobre estampados ni sobre el resto de la prenda.
TEXTURE_HOLE_INPAINT_RADIUS = 2.0


# ============================================================
# UTILIDADES
# ============================================================

def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
    )


def find_upper_clothes_ids(id2label):
    wanted = {
        "upper-clothes",
        "upper-clothing",
        "uppercloth",
        "upperclothes",
        "shirt",
        "t-shirt",
        "tshirt",
        "top",
    }

    compact = {
        x.replace("-", "")
        for x in wanted
    }

    found = []

    for idx, label in id2label.items():
        n = normalize_label(label)

        if (
            n in wanted
            or n.replace("-", "") in compact
        ):
            found.append(int(idx))

    return sorted(set(found))


def draw_text(
    img,
    text,
    x,
    y,
    scale=0.58,
    thickness=2,
):
    cv2.putText(
        img,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def mask_bbox(mask):
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )


def largest_relevant_component(mask):
    mask_u8 = (
        (mask > 0).astype(np.uint8)
        * 255
    )

    h, w = mask_u8.shape

    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    if n <= 1:
        return mask_u8

    min_area = int(
        h
        * w
        * MIN_COMPONENT_AREA_RATIO
    )

    image_center_x = w * 0.5
    candidates = []

    for i in range(1, n):
        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        if area < min_area:
            continue

        cx, cy = centroids[i]

        if cy > h * 0.90:
            continue

        center_penalty = (
            abs(cx - image_center_x)
            / max(1.0, w)
        )

        score = (
            area
            * (
                1.0
                - min(
                    center_penalty,
                    0.45,
                )
            )
        )

        candidates.append(
            (score, i)
        )

    if candidates:
        best = max(
            candidates,
            key=lambda x: x[0],
        )[1]
    else:
        areas = stats[
            1:,
            cv2.CC_STAT_AREA,
        ]

        best = (
            1
            + int(np.argmax(areas))
        )

    out = np.zeros_like(mask_u8)
    out[labels == best] = 255

    return out


def clean_mask(mask):
    mask = (
        (mask > 0).astype(np.uint8)
        * 255
    )

    k = max(
        3,
        int(MORPH_KERNEL),
    )

    if k % 2 == 0:
        k += 1

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask = largest_relevant_component(
        mask
    )

    if MASK_DILATE_PIXELS > 0:
        dk = (
            MASK_DILATE_PIXELS
            * 2
            + 1
        )

        dkernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dk, dk),
            )
        )

        mask = cv2.dilate(
            mask,
            dkernel,
            iterations=1,
        )

    return mask


def mask_is_valid(mask):
    if (
        mask is None
        or not np.any(mask)
    ):
        return False, "sin playera"

    h, w = mask.shape[:2]

    area_ratio = (
        np.count_nonzero(mask)
        / float(h * w)
    )

    if area_ratio < MIN_MASK_AREA_RATIO:
        return (
            False,
            f"muy pequena "
            f"({area_ratio*100:.1f}%)",
        )

    if area_ratio > MAX_MASK_AREA_RATIO:
        return (
            False,
            f"demasiado grande "
            f"({area_ratio*100:.1f}%)",
        )

    bbox = mask_bbox(mask)

    if bbox is None:
        return False, "sin bbox"

    x1, y1, x2, y2 = bbox

    bw = max(
        1,
        x2 - x1 + 1,
    )

    bh = max(
        1,
        y2 - y1 + 1,
    )

    if (
        bw / w
        < MIN_BBOX_WIDTH_RATIO
    ):
        return (
            False,
            "demasiado angosta",
        )

    if (
        bh / h
        < MIN_BBOX_HEIGHT_RATIO
    ):
        return (
            False,
            "demasiado baja",
        )

    cx = (
        x1 + x2
    ) * 0.5

    if (
        abs(cx - w * 0.5)
        > w * 0.40
    ):
        return (
            False,
            "fuera del centro",
        )

    return True, "ok"


def point_segment_distance(
    point,
    a,
    b,
):
    p = np.asarray(
        point,
        dtype=np.float32,
    )

    a = np.asarray(
        a,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    ab = b - a
    denom = float(
        np.dot(ab, ab)
    )

    if denom < 1e-8:
        return float(
            np.linalg.norm(
                p - a
            )
        )

    t = float(
        np.dot(
            p - a,
            ab,
        )
        / denom
    )

    t = float(
        np.clip(
            t,
            0.0,
            1.0,
        )
    )

    q = (
        a
        + t * ab
    )

    return float(
        np.linalg.norm(
            p - q
        )
    )


def similarity_transform_point(
    point,
    ref_a,
    ref_b,
    cur_a,
    cur_b,
    scale_min=None,
    scale_max=None,
):
    """
    Transformacion 2D hueso-referencia -> hueso-actual.

    V4.3: la escala puede limitarse. Esto es CRITICO para las mangas:
    al levantar el brazo, la longitud 2D hombro-codo cambia por perspectiva
    y MediaPipe. Si dejamos scale=n1/n0 sin limite, la manga se "infla".
    """
    p = np.asarray(point, dtype=np.float32)
    ref_a = np.asarray(ref_a, dtype=np.float32)
    ref_b = np.asarray(ref_b, dtype=np.float32)
    cur_a = np.asarray(cur_a, dtype=np.float32)
    cur_b = np.asarray(cur_b, dtype=np.float32)

    u0 = ref_b - ref_a
    u1 = cur_b - cur_a
    n0 = float(np.linalg.norm(u0))
    n1 = float(np.linalg.norm(u1))

    if n0 < 1e-5 or n1 < 1e-5:
        return (p + (cur_a - ref_a)).astype(np.float32)

    e0 = u0 / n0
    e1 = u1 / n1
    cos_t = float(np.clip(np.dot(e0, e1), -1.0, 1.0))
    sin_t = float(e0[0] * e1[1] - e0[1] * e1[0])

    rotation = np.array(
        [[cos_t, -sin_t], [sin_t, cos_t]],
        dtype=np.float32,
    )

    scale = n1 / n0
    if scale_min is not None:
        scale = max(float(scale_min), scale)
    if scale_max is not None:
        scale = min(float(scale_max), scale)

    local = p - ref_a
    return (cur_a + scale * (rotation @ local)).astype(np.float32)


def clip_vector(
    vector,
    max_length,
):
    v = np.asarray(
        vector,
        dtype=np.float32,
    )

    mag = float(
        np.linalg.norm(v)
    )

    if (
        mag > max_length
        and mag > 1e-6
    ):
        v = (
            v
            * (
                max_length
                / mag
            )
        )

    return v.astype(
        np.float32
    )


# ============================================================
# SEGMENTACION
# ============================================================

class UpperClothesSegmenter:
    def __init__(self):
        self.processor = None
        self.model = None
        self.upper_ids = []
        self.loaded = False

    def load(self):
        if self.loaded:
            return

        print(
            "\n[SEG] Cargando modelo "
            "de ropa..."
        )

        print(
            "[SEG] Modelo:",
            MODEL_NAME,
        )

        print(
            "[SEG] Device:",
            DEVICE,
        )

        t0 = time.perf_counter()

        self.processor = (
            AutoImageProcessor
            .from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            SegformerForSemanticSegmentation
            .from_pretrained(
                MODEL_NAME
            )
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.upper_ids = (
            find_upper_clothes_ids(
                self.model.config.id2label
            )
        )

        if not self.upper_ids:
            print(
                "[SEG] Clases disponibles:"
            )

            for idx, label in (
                self.model
                .config
                .id2label
                .items()
            ):
                print(
                    f"  {idx}: {label}"
                )

            raise RuntimeError(
                "No encontre la clase "
                "Upper-clothes."
            )

        labels = [
            self.model.config.id2label[i]
            for i in self.upper_ids
        ]

        print(
            "[SEG] Upper-clothes:",
            self.upper_ids,
            labels,
        )

        print(
            f"[SEG] Modelo listo en "
            f"{(time.perf_counter()-t0)*1000:.1f} ms"
        )

        self.loaded = True

    @torch.inference_mode()
    def segment(
        self,
        frame_bgr,
    ):
        self.load()

        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        pil = Image.fromarray(rgb)

        t0 = time.perf_counter()

        inputs = self.processor(
            images=pil,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v
            in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        logits = (
            torch.nn.functional
            .interpolate(
                outputs.logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        )

        pred = (
            logits
            .argmax(dim=1)[0]
        )

        mask_t = (
            torch.zeros_like(
                pred,
                dtype=torch.bool,
            )
        )

        for class_id in self.upper_ids:
            mask_t |= (
                pred
                == int(class_id)
            )

        mask = (
            mask_t
            .to(torch.uint8)
            .mul(255)
            .cpu()
            .numpy()
        )

        seg_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        return (
            clean_mask(mask),
            seg_ms,
        )


# ============================================================
# POSE ABSOLUTA
# ============================================================

class PoseTracker:
    CONNECTIONS = (
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
    )

    IMPORTANT_IDS = (
        11, 12,
        13, 14,
        15, 16,
        23, 24,
    )

    def __init__(self):
        if not os.path.exists(
            POSE_MODEL_PATH
        ):
            raise FileNotFoundError(
                "\nNo encuentro:\n"
                f"  {POSE_MODEL_PATH}\n\n"
                "Debe estar dentro de:\n"
                "  models/pose_landmarker.task\n"
            )

        BaseOptions = (
            mp.tasks.BaseOptions
        )

        PoseLandmarker = (
            mp.tasks.vision
            .PoseLandmarker
        )

        PoseLandmarkerOptions = (
            mp.tasks.vision
            .PoseLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision
            .RunningMode
        )

        options = (
            PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=(
                        POSE_MODEL_PATH
                    )
                ),
                running_mode=(
                    RunningMode.VIDEO
                ),
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
                output_segmentation_masks=False,
            )
        )

        self.landmarker = (
            PoseLandmarker
            .create_from_options(
                options
            )
        )

        self.prev_points = None
        self.curr_points = None
        self.visibility = None

        self.detected = False
        self.last_ms = 0.0
        self.last_timestamp_ms = 0

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    def valid(self, idx):
        return (
            self.curr_points is not None
            and self.visibility is not None
            and self.visibility[idx]
            >= POSE_MIN_VISIBILITY
        )

    def update(
        self,
        frame_bgr,
    ):
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=np.ascontiguousarray(
                rgb
            ),
        )

        timestamp_ms = int(
            time.monotonic()
            * 1000
        )

        if (
            timestamp_ms
            <= self.last_timestamp_ms
        ):
            timestamp_ms = (
                self.last_timestamp_ms
                + 1
            )

        self.last_timestamp_ms = (
            timestamp_ms
        )

        t0 = time.perf_counter()

        result = (
            self.landmarker
            .detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        self.last_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        old = (
            None
            if self.curr_points is None
            else self.curr_points.copy()
        )

        if not result.pose_landmarks:
            self.prev_points = old
            self.detected = False
            return False

        landmarks = (
            result.pose_landmarks[0]
        )

        raw = np.zeros(
            (33, 2),
            dtype=np.float32,
        )

        visibility = np.zeros(
            33,
            dtype=np.float32,
        )

        for i, lm in enumerate(
            landmarks
        ):
            raw[i, 0] = (
                float(lm.x)
                * w
            )

            raw[i, 1] = (
                float(lm.y)
                * h
            )

            vis = float(
                getattr(
                    lm,
                    "visibility",
                    1.0,
                )
            )

            pres = float(
                getattr(
                    lm,
                    "presence",
                    1.0,
                )
            )

            visibility[i] = min(
                vis,
                pres,
            )

        if old is not None:
            current = (
                POSE_SMOOTHING
                * raw
                + (
                    1.0
                    - POSE_SMOOTHING
                )
                * old
            ).astype(
                np.float32
            )
        else:
            current = raw

        self.prev_points = old
        self.curr_points = current
        self.visibility = visibility
        self.detected = True

        return True

    def draw(
        self,
        image,
    ):
        if self.curr_points is None:
            return

        h, w = image.shape[:2]

        for a, b in self.CONNECTIONS:
            if (
                not self.valid(a)
                or not self.valid(b)
            ):
                continue

            pa = self.curr_points[a]
            pb = self.curr_points[b]

            p1 = (
                int(
                    np.clip(
                        pa[0],
                        0,
                        w - 1,
                    )
                ),
                int(
                    np.clip(
                        pa[1],
                        0,
                        h - 1,
                    )
                ),
            )

            p2 = (
                int(
                    np.clip(
                        pb[0],
                        0,
                        w - 1,
                    )
                ),
                int(
                    np.clip(
                        pb[1],
                        0,
                        h - 1,
                    )
                ),
            )

            cv2.line(
                image,
                p1,
                p2,
                (255, 80, 255),
                2,
                cv2.LINE_AA,
            )

        for idx in self.IMPORTANT_IDS:
            if not self.valid(idx):
                continue

            p = self.curr_points[idx]

            x = int(
                np.clip(
                    p[0],
                    0,
                    w - 1,
                )
            )

            y = int(
                np.clip(
                    p[1],
                    0,
                    h - 1,
                )
            )

            cv2.circle(
                image,
                (x, y),
                5,
                (255, 80, 255),
                -1,
                cv2.LINE_AA,
            )


# ============================================================
# MODELO DE ESQUELETO ANCLADO
# ============================================================

class AnchoredPoseModel:
    """
    Guarda Pose del keyframe.

    La deformacion siempre se calcula:
        referencia -> pose actual

    Nunca:
        frame anterior -> frame actual -> frame siguiente...
    """

    # nombre, landmark A, landmark B, sigma, fuerza
    BONE_SPECS = (
        # Torso
        ("shoulders", 11, 12, 150.0, 1.20),
        ("left_torso", 11, 23, 145.0, 1.05),
        ("right_torso", 12, 24, 145.0, 1.05),
        ("hips", 23, 24, 170.0, 0.65),

        # Mangas
        ("left_upper_arm", 11, 13, 92.0, 1.85),
        ("right_upper_arm", 12, 14, 92.0, 1.85),

        # Antebrazo: influencia menor, ayuda en poses extremas.
        ("left_forearm", 13, 15, 68.0, 0.30),
        ("right_forearm", 14, 16, 68.0, 0.30),
    )

    def __init__(self):
        self.reference_points = None
        self.reference_visibility = None
        self.ready = False

    def reset(self):
        self.reference_points = None
        self.reference_visibility = None
        self.ready = False

    def capture(
        self,
        pose,
    ):
        if (
            pose is None
            or not pose.detected
            or pose.curr_points is None
        ):
            return False

        required = (
            11, 12,
            13, 14,
            23, 24,
        )

        if not all(
            pose.valid(i)
            for i in required
        ):
            return False

        self.reference_points = (
            pose.curr_points.copy()
        )

        self.reference_visibility = (
            pose.visibility.copy()
        )

        self.ready = True

        print(
            "[ANCHOR] Pose de referencia "
            "capturada."
        )

        return True

    def _bone_is_usable(
        self,
        pose,
        a,
        b,
    ):
        if (
            not self.ready
            or pose is None
            or pose.curr_points is None
        ):
            return False

        if (
            self.reference_visibility[a]
            < POSE_MIN_VISIBILITY
            or self.reference_visibility[b]
            < POSE_MIN_VISIBILITY
        ):
            return False

        return (
            pose.valid(a)
            and pose.valid(b)
        )

    def weights_for_point(self, reference_point):
        """
        V4.3 - SKINNING SEMANTICO.

        V4.2 usaba gaussianas para TODOS los huesos. Por eso un vertice del
        pecho podia recibir una fraccion apreciable del brazo; al levantarlo,
        esa rotacion tiraba del torso hacia afuera y la playera se inflaba.

        Ahora primero decidimos si el vertice pertenece geometricamente a una
        manga o al torso. Los brazos casi no influyen en el torso.
        """
        if not self.ready:
            return []

        p = np.asarray(reference_point, dtype=np.float32)
        rp = self.reference_points

        ls = rp[11]
        rs = rp[12]
        le = rp[13]
        re = rp[14]

        shoulder_width = max(20.0, float(np.linalg.norm(ls - rs)))

        # Direcciones que salen del torso hacia cada hombro.
        left_out = ls - rs
        right_out = rs - ls
        left_out /= max(1e-6, float(np.linalg.norm(left_out)))
        right_out /= max(1e-6, float(np.linalg.norm(right_out)))

        left_projection = float(np.dot(p - ls, left_out)) / shoulder_width
        right_projection = float(np.dot(p - rs, right_out)) / shoulder_width

        left_arm_dist = point_segment_distance(p, ls, le)
        right_arm_dist = point_segment_distance(p, rs, re)
        arm_dist_limit = shoulder_width * SLEEVE_ARM_DISTANCE_FACTOR

        left_sleeve_score = 0.0
        right_sleeve_score = 0.0

        if left_projection > SLEEVE_OUTWARD_THRESHOLD and left_arm_dist < arm_dist_limit:
            left_sleeve_score = (
                max(0.0, 1.0 - left_arm_dist / arm_dist_limit)
                * max(0.20, min(1.0, (left_projection + 0.12) / 0.42))
            )

        if right_projection > SLEEVE_OUTWARD_THRESHOLD and right_arm_dist < arm_dist_limit:
            right_sleeve_score = (
                max(0.0, 1.0 - right_arm_dist / arm_dist_limit)
                * max(0.20, min(1.0, (right_projection + 0.12) / 0.42))
            )

        # Evita que un punto ambiguo pertenezca a ambas mangas.
        if left_sleeve_score >= right_sleeve_score and left_sleeve_score > 0.10:
            sleeve_name = "left_upper_arm"
            sleeve_a, sleeve_b = 11, 13
            sleeve_score = min(1.0, left_sleeve_score)
        elif right_sleeve_score > 0.10:
            sleeve_name = "right_upper_arm"
            sleeve_a, sleeve_b = 12, 14
            sleeve_score = min(1.0, right_sleeve_score)
        else:
            sleeve_name = None
            sleeve_a = sleeve_b = None
            sleeve_score = 0.0

        # Base del torso. Hombros mandan en la parte alta; laterales del torso
        # y cadera estabilizan la parte baja. Ninguno depende del brazo.
        torso_raw = []
        for name, a, b, sigma, strength in self.BONE_SPECS:
            if "upper_arm" in name or "forearm" in name:
                continue
            dist = point_segment_distance(p, rp[a], rp[b])
            weight = math.exp(-(dist * dist) / (2.0 * sigma * sigma)) * strength
            torso_raw.append((name, a, b, float(weight)))

        torso_total = sum(x[3] for x in torso_raw)
        if torso_total < 1e-8:
            torso_weights = [("shoulders", 11, 12, 1.0)]
        else:
            torso_weights = [
                (name, a, b, weight / torso_total)
                for name, a, b, weight in torso_raw
            ]

        if sleeve_name is None:
            # Torso: influencia del brazo esencialmente cero.
            return torso_weights

        # Manga: brazo dominante, pero dejamos una pequena conexion al torso
        # para que la costura del hombro no se abra.
        arm_share = min(
            SLEEVE_ARM_DOMINANCE,
            0.58 + 0.30 * sleeve_score,
        )
        torso_share = 1.0 - arm_share

        result = [(sleeve_name, sleeve_a, sleeve_b, arm_share)]
        result.extend(
            (name, a, b, weight * torso_share)
            for name, a, b, weight in torso_weights
        )
        return result

    def transform_reference_point(
        self,
        reference_point,
        fixed_weights,
        pose,
    ):
        """
        Posicion ABSOLUTA del punto para la pose actual.
        """
        p = np.asarray(
            reference_point,
            dtype=np.float32,
        )

        if (
            not self.ready
            or pose is None
            or pose.curr_points is None
        ):
            return p.copy()

        result = np.zeros(
            2,
            dtype=np.float32,
        )

        total = 0.0

        for (
            bone_name,
            a,
            b,
            base_weight,
        ) in fixed_weights:
            if not self._bone_is_usable(
                pose,
                a,
                b,
            ):
                continue

            if "arm" in bone_name or "forearm" in bone_name:
                scale_min = ARM_SCALE_MIN
                scale_max = ARM_SCALE_MAX
            else:
                scale_min = TORSO_SCALE_MIN
                scale_max = TORSO_SCALE_MAX

            transformed = similarity_transform_point(
                p,
                self.reference_points[a],
                self.reference_points[b],
                pose.curr_points[a],
                pose.curr_points[b],
                scale_min=scale_min,
                scale_max=scale_max,
            )

            # Visibilidad actual modula, pero NO redefine
            # el peso geometrico del keyframe.
            visibility = min(
                float(
                    pose.visibility[a]
                ),
                float(
                    pose.visibility[b]
                ),
            )

            w = (
                base_weight
                * max(
                    0.15,
                    visibility,
                )
            )

            result += (
                transformed
                * w
            )

            total += w

        if total < 1e-8:
            # Respaldo: traslacion media de hombros.
            deltas = []

            for idx in (
                11, 12,
                23, 24,
            ):
                if pose.valid(idx):
                    deltas.append(
                        pose.curr_points[idx]
                        - self.reference_points[idx]
                    )

            if deltas:
                return (
                    p
                    + np.median(
                        np.asarray(
                            deltas,
                            dtype=np.float32,
                        ),
                        axis=0,
                    )
                ).astype(np.float32)

            return p.copy()

        return (
            result / total
        ).astype(np.float32)

    def transform_articulated_sleeve_point(
        self,
        reference_point,
        fixed_weights,
        pose,
        side,
        arm_weight,
    ):
        """
        V4.5: transformacion explicita de manga.

        La manga rota alrededor del hombro siguiendo hombro->codo, pero su
        escala queda practicamente fija. Cerca de la costura se mezcla con una
        transformacion de torso para mantener continuidad.

        side:
            -1 = manga izquierda
             1 = manga derecha
        """
        p = np.asarray(reference_point, dtype=np.float32)

        if (
            not self.ready
            or pose is None
            or pose.curr_points is None
        ):
            return p.copy()

        if side < 0:
            shoulder_idx, elbow_idx = 11, 13
        else:
            shoulder_idx, elbow_idx = 12, 14

        if not self._bone_is_usable(
            pose,
            shoulder_idx,
            elbow_idx,
        ):
            return self.transform_reference_point(
                p,
                fixed_weights,
                pose,
            )

        # 1) Movimiento rigido principal de la manga.
        arm_position = similarity_transform_point(
            p,
            self.reference_points[shoulder_idx],
            self.reference_points[elbow_idx],
            pose.curr_points[shoulder_idx],
            pose.curr_points[elbow_idx],
            scale_min=ARTICULATED_ARM_SCALE_MIN,
            scale_max=ARTICULATED_ARM_SCALE_MAX,
        )

        # 2) Movimiento del torso SIN huesos del brazo.
        torso_weights = [
            (name, a, b, weight)
            for name, a, b, weight in fixed_weights
            if ("upper_arm" not in name and "forearm" not in name)
        ]

        if torso_weights:
            total = sum(float(x[3]) for x in torso_weights)
            if total > 1e-8:
                torso_weights = [
                    (name, a, b, float(weight) / total)
                    for name, a, b, weight in torso_weights
                ]

            torso_position = self.transform_reference_point(
                p,
                torso_weights,
                pose,
            )
        else:
            # Respaldo estable usando la linea de hombros.
            torso_position = similarity_transform_point(
                p,
                self.reference_points[11],
                self.reference_points[12],
                pose.curr_points[11],
                pose.curr_points[12],
                scale_min=TORSO_SCALE_MIN,
                scale_max=TORSO_SCALE_MAX,
            )

        # 3) Peso de articulacion.
        # Cerca de 0.45 estamos en la costura: aun manda bastante el torso.
        # A partir de ~0.78 la manga se comporta casi como una pieza rigida.
        denom = max(
            1e-6,
            ARTICULATED_SLEEVE_FULL_ARM_WEIGHT
            - ARTICULATED_SLEEVE_MIN_ARM_WEIGHT,
        )

        t = float(
            np.clip(
                (
                    float(arm_weight)
                    - ARTICULATED_SLEEVE_MIN_ARM_WEIGHT
                ) / denom,
                0.0,
                1.0,
            )
        )

        # smoothstep
        t = t * t * (3.0 - 2.0 * t)

        rigid_blend = (
            SLEEVE_MIN_RIGID_BLEND
            + t
            * (
                SLEEVE_MAX_RIGID_BLEND
                - SLEEVE_MIN_RIGID_BLEND
            )
        )

        return (
            torso_position * (1.0 - rigid_blend)
            + arm_position * rigid_blend
        ).astype(np.float32)

    def incremental_flow_at_point(
        self,
        point,
        pose,
    ):
        """
        Cuanto movimiento del frame anterior al actual ya explica Pose
        en una coordenada arbitraria.

        Se usa para quitar a LK el movimiento corporal y conservar solo
        el residuo de tela.
        """
        if (
            pose is None
            or pose.prev_points is None
            or pose.curr_points is None
        ):
            return np.zeros(
                2,
                dtype=np.float32,
            )

        p = np.asarray(
            point,
            dtype=np.float32,
        )

        accum = np.zeros(
            2,
            dtype=np.float32,
        )

        total = 0.0

        for (
            bone_name,
            a,
            b,
            sigma,
            strength,
        ) in self.BONE_SPECS:
            if (
                not pose.valid(a)
                or not pose.valid(b)
            ):
                continue

            prev_a = pose.prev_points[a]
            prev_b = pose.prev_points[b]

            dist = (
                point_segment_distance(
                    p,
                    prev_a,
                    prev_b,
                )
            )

            weight = (
                math.exp(
                    -(
                        dist * dist
                    )
                    / (
                        2.0
                        * sigma
                        * sigma
                    )
                )
                * strength
            )

            if "arm" in bone_name or "forearm" in bone_name:
                scale_min = ARM_SCALE_MIN
                scale_max = ARM_SCALE_MAX
            else:
                scale_min = TORSO_SCALE_MIN
                scale_max = TORSO_SCALE_MAX

            predicted = similarity_transform_point(
                p,
                prev_a,
                prev_b,
                pose.curr_points[a],
                pose.curr_points[b],
                scale_min=scale_min,
                scale_max=scale_max,
            )

            flow = predicted - p

            accum += (
                flow
                * weight
            )

            total += weight

        if total < 1e-8:
            return np.zeros(
                2,
                dtype=np.float32,
            )

        return (
            accum / total
        ).astype(np.float32)


# ============================================================
# LUCAS-KANADE RESTRINGIDO
# ============================================================

class ResidualLKTracker:
    """
    LK ya no conserva autoridad geometrica.

    Solo produce:
        punto_anterior,
        punto_actual,
        residuo = flujo_LK - flujo_Pose

    Ademas:
    - forward/backward check;
    - elimina puntos fuera de ROI;
    - rechaza residuos absurdos;
    - renueva features cada pocos frames.
    """

    def __init__(self):
        self.prev_gray = None
        self.points = None

        self.old_good = None
        self.new_good = None
        self.residuals = None

        self.good_count = 0
        self.residual_count = 0
        self.confidence = 0.0

        self.frame_age = 0
        self.last_ms = 0.0

    def reset(self):
        self.prev_gray = None
        self.points = None

        self.old_good = None
        self.new_good = None
        self.residuals = None

        self.good_count = 0
        self.residual_count = 0
        self.confidence = 0.0

        self.frame_age = 0
        self.last_ms = 0.0

    def _detect_features(
        self,
        gray,
        roi_mask,
    ):
        if (
            roi_mask is None
            or not np.any(roi_mask)
        ):
            return None

        kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 5),
            )
        )

        safe = cv2.erode(
            roi_mask,
            kernel,
            iterations=1,
        )

        return cv2.goodFeaturesToTrack(
            gray,
            mask=safe,
            maxCorners=LK_MAX_CORNERS,
            qualityLevel=LK_QUALITY_LEVEL,
            minDistance=LK_MIN_DISTANCE,
            blockSize=LK_BLOCK_SIZE,
            useHarrisDetector=False,
        )

    def initialize(
        self,
        frame_bgr,
        roi_mask,
    ):
        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        self.prev_gray = gray

        self.points = (
            self._detect_features(
                gray,
                roi_mask,
            )
        )

        self.frame_age = 0

        count = (
            0
            if self.points is None
            else len(self.points)
        )

        print(
            f"[LK] Features iniciales: "
            f"{count}"
        )

    def update(
        self,
        frame_bgr,
        roi_mask,
        pose,
        anchored_pose,
    ):
        t0 = time.perf_counter()

        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        self.old_good = None
        self.new_good = None
        self.residuals = None

        self.good_count = 0
        self.residual_count = 0
        self.confidence = 0.0

        survivors = None

        if (
            self.prev_gray is not None
            and self.points is not None
            and len(self.points) > 0
        ):
            nxt, st1, _ = (
                cv2.calcOpticalFlowPyrLK(
                    self.prev_gray,
                    gray,
                    self.points,
                    None,
                    winSize=LK_WIN_SIZE,
                    maxLevel=LK_MAX_LEVEL,
                    criteria=LK_CRITERIA,
                )
            )

            if (
                nxt is not None
                and st1 is not None
            ):
                back, st2, _ = (
                    cv2.calcOpticalFlowPyrLK(
                        gray,
                        self.prev_gray,
                        nxt,
                        None,
                        winSize=LK_WIN_SIZE,
                        maxLevel=LK_MAX_LEVEL,
                        criteria=LK_CRITERIA,
                    )
                )

                if (
                    back is not None
                    and st2 is not None
                ):
                    old = (
                        self.points
                        .reshape(-1, 2)
                    )

                    new = (
                        nxt
                        .reshape(-1, 2)
                    )

                    rev = (
                        back
                        .reshape(-1, 2)
                    )

                    fb_error = np.linalg.norm(
                        old - rev,
                        axis=1,
                    )

                    good = (
                        st1.reshape(-1)
                        .astype(bool)
                        & st2.reshape(-1)
                        .astype(bool)
                        & np.isfinite(
                            fb_error
                        )
                        & (
                            fb_error
                            <= LK_FB_MAX_ERROR
                        )
                    )

                    # El punto nuevo debe seguir dentro
                    # de la region predicha de la playera.
                    h, w = roi_mask.shape[:2]

                    new_x = np.round(
                        new[:, 0]
                    ).astype(np.int32)

                    new_y = np.round(
                        new[:, 1]
                    ).astype(np.int32)

                    inside_bounds = (
                        (new_x >= 0)
                        & (new_x < w)
                        & (new_y >= 0)
                        & (new_y < h)
                    )

                    inside_roi = np.zeros(
                        len(new),
                        dtype=bool,
                    )

                    valid_ids = np.where(
                        inside_bounds
                    )[0]

                    if len(valid_ids):
                        inside_roi[
                            valid_ids
                        ] = (
                            roi_mask[
                                new_y[valid_ids],
                                new_x[valid_ids],
                            ]
                            > 0
                        )

                    good &= inside_roi

                    old_good = old[good]
                    new_good = new[good]

                    self.good_count = len(
                        new_good
                    )

                    if self.good_count:
                        residuals = []

                        kept_old = []
                        kept_new = []

                        for (
                            p_old,
                            p_new,
                        ) in zip(
                            old_good,
                            new_good,
                        ):
                            observed = (
                                p_new
                                - p_old
                            )

                            expected = (
                                anchored_pose
                                .incremental_flow_at_point(
                                    p_old,
                                    pose,
                                )
                            )

                            residual = (
                                observed
                                - expected
                            )

                            mag = float(
                                np.linalg.norm(
                                    residual
                                )
                            )

                            # Si LK contradice demasiado
                            # al cuerpo, no se le cree.
                            if (
                                mag
                                > LK_RESIDUAL_HARD_REJECT
                            ):
                                continue

                            residual = clip_vector(
                                residual,
                                LK_RESIDUAL_MAX,
                            )

                            kept_old.append(
                                p_old
                            )

                            kept_new.append(
                                p_new
                            )

                            residuals.append(
                                residual
                            )

                        if residuals:
                            self.old_good = (
                                np.asarray(
                                    kept_old,
                                    dtype=np.float32,
                                )
                            )

                            self.new_good = (
                                np.asarray(
                                    kept_new,
                                    dtype=np.float32,
                                )
                            )

                            self.residuals = (
                                np.asarray(
                                    residuals,
                                    dtype=np.float32,
                                )
                            )

                            self.residual_count = (
                                len(
                                    self.residuals
                                )
                            )

                            survivors = (
                                self.new_good
                                .reshape(-1, 1, 2)
                                .astype(np.float32)
                            )

                            valid_ratio = (
                                self.residual_count
                                / max(
                                    1.0,
                                    float(
                                        len(old)
                                    ),
                                )
                            )

                            self.confidence = float(
                                np.clip(
                                    valid_ratio,
                                    0.0,
                                    1.0,
                                )
                            )

        self.prev_gray = gray
        self.frame_age += 1

        self.points = survivors

        # Renovacion frecuente:
        # LK no tiene permiso de "vivir para siempre".
        refresh = (
            self.points is None
            or len(self.points) < 8
            or (
                self.frame_age
                % LK_REFRESH_EVERY
                == 0
            )
        )

        if refresh:
            fresh = self._detect_features(
                gray,
                roi_mask,
            )

            if fresh is not None:
                self.points = (
                    fresh.astype(
                        np.float32
                    )
                )

        self.last_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0


# ============================================================
# MALLA ANCLADA
# ============================================================

class AnchoredGarmentMesh:
    """
    V4.3: MALLA ANCLADA + SKINNING SEMANTICO ANTI-INFLADO.

    Cambios frente a V4.1:
    - reference_vertices sigue siendo INMUTABLE.
    - La deformacion sigue siendo absoluta desde el keyframe.
    - Pero la mascara visual YA NO usa convexHull.
    - Se muestrea el contorno real de SegFormer.
    - Se agregan puntos interiores.
    - Se crea Delaunay y se conservan SOLO triangulos que realmente
      pertenecen a la mascara de Upper-clothes.
    - display_mask() y roi_mask() se reconstruyen con esos triangulos.

    Resultado: cuello, axilas, mangas y cintura pueden conservar una
    silueta de prenda en vez de convertirse en una mancha convexa.
    """

    def __init__(self):
        self.reference_vertices = None
        self.detected_vertices = None
        self.current_vertices = None

        self.triangles = []
        self.edges = []
        self.neighbors = []
        self.boundary_flags = None

        self.vertex_weights = []
        self.vertex_is_sleeve = None
        self.vertex_sleeve_side = None
        self.vertex_arm_weight = None

        # V4.8: estado instantaneo de articulacion.
        # 0 = brazo relajado hacia abajo, 1 = horizontal/arriba.
        self.left_arm_raise = 0.0
        self.right_arm_raise = 0.0

        # Geometria inmutable del keyframe para restricciones de tela.
        self.reference_edge_lengths = {}
        self.reference_triangle_areas = []
        self.reference_triangle_signed_areas = []

        self.reference_mask = None
        self.initialized = False
        self.last_ms = 0.0

    def reset(self):
        self.reference_vertices = None
        self.detected_vertices = None
        self.current_vertices = None

        self.triangles = []
        self.edges = []
        self.neighbors = []
        self.boundary_flags = None

        self.vertex_weights = []
        self.vertex_is_sleeve = None
        self.vertex_sleeve_side = None
        self.vertex_arm_weight = None

        # V4.8: estado instantaneo de articulacion.
        # 0 = brazo relajado hacia abajo, 1 = horizontal/arriba.
        self.left_arm_raise = 0.0
        self.right_arm_raise = 0.0

        # Geometria inmutable del keyframe para restricciones de tela.
        self.reference_edge_lengths = {}
        self.reference_triangle_areas = []
        self.reference_triangle_signed_areas = []

        self.reference_mask = None
        self.initialized = False
        self.last_ms = 0.0

    @staticmethod
    def _sample_contour_by_distance(contour, spacing):
        """Muestrea un contorno cerrado aproximadamente cada `spacing` px."""
        pts = contour.reshape(-1, 2).astype(np.float32)

        if len(pts) < 3:
            return []

        # CHAIN_APPROX_NONE nos da el borde pixel a pixel. Acumulamos distancia
        # para no depender de cuantos pixeles tenga cada segmento.
        sampled = [pts[0].copy()]
        accumulated = 0.0
        previous = pts[0]

        for i in range(1, len(pts)):
            current = pts[i]
            accumulated += float(np.linalg.norm(current - previous))

            if accumulated >= spacing:
                sampled.append(current.copy())
                accumulated = 0.0

            previous = current

        # No duplicar el primer punto al cerrar.
        if len(sampled) >= 2:
            if np.linalg.norm(sampled[-1] - sampled[0]) < spacing * 0.45:
                sampled.pop()

        return sampled

    @staticmethod
    def _deduplicate_points(points, min_distance=3.0):
        if not points:
            return []

        kept = []
        min_d2 = float(min_distance * min_distance)

        for p in points:
            p = np.asarray(p, dtype=np.float32)

            if not kept:
                kept.append(p)
                continue

            arr = np.asarray(kept, dtype=np.float32)
            d2 = np.sum((arr - p) ** 2, axis=1)

            if float(np.min(d2)) >= min_d2:
                kept.append(p)

        return kept

    @staticmethod
    def _triangle_signed_area(a, b, c):
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0]) * 0.5

    @staticmethod
    def _triangle_area(a, b, c):
        return abs(
            AnchoredGarmentMesh._triangle_signed_area(a, b, c)
        )

    @staticmethod
    def _triangle_overlap_with_mask(tri_points, mask):
        """Fraccion del triangulo que cae realmente dentro de Upper-clothes."""
        tri = np.round(np.asarray(tri_points, dtype=np.float32)).astype(np.int32)
        h, w = mask.shape[:2]

        x, y, bw, bh = cv2.boundingRect(tri)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + bw)
        y1 = min(h, y + bh)

        if x1 <= x0 or y1 <= y0:
            return 0.0

        local_tri = tri.copy()
        local_tri[:, 0] -= x0
        local_tri[:, 1] -= y0

        tri_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, local_tri, 255)

        tri_pixels = int(np.count_nonzero(tri_mask))
        if tri_pixels <= 0:
            return 0.0

        garment = mask[y0:y1, x0:x1] > 0
        overlap = np.count_nonzero((tri_mask > 0) & garment)

        return float(overlap) / float(tri_pixels)

    @staticmethod
    def _nearest_vertex_index(point, vertices, tolerance=2.5):
        p = np.asarray(point, dtype=np.float32)
        delta = vertices - p
        d2 = np.sum(delta * delta, axis=1)
        idx = int(np.argmin(d2))

        if float(d2[idx]) <= tolerance * tolerance:
            return idx

        return None

    def _build_delaunay(self, vertices, mask, spacing):
        h, w = mask.shape[:2]

        # Subdiv requiere un rect valido y puntos estrictamente dentro.
        subdiv = cv2.Subdiv2D((0, 0, w, h))

        for p in vertices:
            x = float(np.clip(p[0], 0.001, w - 1.001))
            y = float(np.clip(p[1], 0.001, h - 1.001))
            try:
                subdiv.insert((x, y))
            except cv2.error:
                # Duplicados numericos muy raros: la deduplicacion ya hizo
                # casi todo el trabajo, asi que podemos ignorarlos.
                pass

        raw = subdiv.getTriangleList()
        triangles = []
        seen = set()

        max_edge = float(spacing) * TRIANGLE_MAX_EDGE_FACTOR

        for t in raw:
            tri_coords = np.asarray(
                [[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]],
                dtype=np.float32,
            )

            # Fuera del frame -> no nos sirve.
            if (
                np.any(tri_coords[:, 0] < 0)
                or np.any(tri_coords[:, 0] >= w)
                or np.any(tri_coords[:, 1] < 0)
                or np.any(tri_coords[:, 1] >= h)
            ):
                continue

            ids = []
            failed = False

            for p in tri_coords:
                idx = self._nearest_vertex_index(p, vertices)
                if idx is None:
                    failed = True
                    break
                ids.append(idx)

            if failed or len(set(ids)) != 3:
                continue

            key = tuple(sorted(ids))
            if key in seen:
                continue

            a, b, c = vertices[ids[0]], vertices[ids[1]], vertices[ids[2]]

            area = self._triangle_area(a, b, c)
            if area < TRIANGLE_MIN_AREA:
                continue

            edges = (
                float(np.linalg.norm(a - b)),
                float(np.linalg.norm(b - c)),
                float(np.linalg.norm(c - a)),
            )

            # Evita puentes Delaunay enormes entre mangas/torso.
            if max(edges) > max_edge:
                continue

            overlap = self._triangle_overlap_with_mask(
                np.asarray([a, b, c], dtype=np.float32),
                mask,
            )

            # ESTA ES LA REGLA QUE CONSERVA AXILAS/CUELLO:
            # un triangulo que cruza una zona donde SegFormer dijo "no ropa"
            # se elimina en el keyframe y nunca existira durante el tracking.
            if overlap < TRIANGLE_MIN_MASK_OVERLAP:
                continue

            seen.add(key)
            triangles.append(tuple(ids))

        return triangles

    @staticmethod
    def _largest_connected_triangle_component(triangles):
        """
        V4.7: deja una sola pieza topologica.

        Dos triangulos son vecinos solamente si comparten una ARISTA.
        Esto elimina pequeñas islas Delaunay que puedan aparecer cerca del
        contorno sin romper cuello/axilas de la pieza principal.
        """
        if not triangles:
            return []

        edge_to_triangles = {}

        for tri_idx, (a, b, c) in enumerate(triangles):
            for u, v in ((a, b), (b, c), (c, a)):
                edge = tuple(sorted((int(u), int(v))))
                edge_to_triangles.setdefault(edge, []).append(tri_idx)

        adjacency = [set() for _ in triangles]

        for ids in edge_to_triangles.values():
            if len(ids) < 2:
                continue

            for i in ids:
                for j in ids:
                    if i != j:
                        adjacency[i].add(j)

        visited = set()
        components = []

        for start in range(len(triangles)):
            if start in visited:
                continue

            stack = [start]
            visited.add(start)
            component = []

            while stack:
                current = stack.pop()
                component.append(current)

                for nxt in adjacency[current]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)

            components.append(component)

        largest = max(components, key=len)

        return [triangles[i] for i in largest]

    @staticmethod
    def _smoothstep01(value):
        t = float(np.clip(value, 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    def _apply_garment_ease(self, vertices, anchored_pose):
        """
        V4.6 - GARMENT EASE.

        MediaPipe es la armadura:
          * mueve hombros/codos;
          * rota mangas;
          * mueve el torso.

        Pero la ropa mantiene volumen alrededor de esa armadura.

        La expansión ocurre UNA VEZ en el keyframe. Después, toda la malla
        anclada y el solver trabajan sobre esta geometría holgada.
        """
        if (
            vertices is None
            or len(vertices) == 0
            or anchored_pose is None
            or not anchored_pose.ready
        ):
            return vertices.copy()

        rp = anchored_pose.reference_points
        ls = rp[11].astype(np.float32)
        rs = rp[12].astype(np.float32)
        le = rp[13].astype(np.float32)
        re = rp[14].astype(np.float32)
        lh = rp[23].astype(np.float32)
        rh = rp[24].astype(np.float32)

        shoulder_mid = (ls + rs) * 0.5
        hip_mid = (lh + rh) * 0.5

        lateral = rs - ls
        shoulder_width = float(np.linalg.norm(lateral))
        if shoulder_width < 1e-6:
            return vertices.copy()

        lateral /= shoulder_width

        # Eje perpendicular a hombros, orientado hacia caderas.
        down = np.asarray([-lateral[1], lateral[0]], dtype=np.float32)
        if float(np.dot(down, hip_mid - shoulder_mid)) < 0.0:
            down *= -1.0

        torso_height = max(
            1.0,
            float(np.linalg.norm(hip_mid - shoulder_mid)),
        )

        eased = vertices.astype(np.float32).copy()

        for i, p in enumerate(vertices):
            p = p.astype(np.float32)

            is_sleeve = (
                self.vertex_is_sleeve is not None
                and i < len(self.vertex_is_sleeve)
                and bool(self.vertex_is_sleeve[i])
            )
            side = (
                int(self.vertex_sleeve_side[i])
                if (
                    self.vertex_sleeve_side is not None
                    and i < len(self.vertex_sleeve_side)
                )
                else 0
            )

            if is_sleeve and side != 0:
                # ------------------------------------------
                # MANGA: el hueso es el eje, NO el borde.
                # ------------------------------------------
                if side < 0:
                    shoulder = ls
                    elbow = le
                    outward = ls - rs
                else:
                    shoulder = rs
                    elbow = re
                    outward = rs - ls

                axis = elbow - shoulder
                axis_len = float(np.linalg.norm(axis))
                if axis_len < 1e-6:
                    continue
                axis /= axis_len

                outward_len = float(np.linalg.norm(outward))
                if outward_len > 1e-6:
                    outward /= outward_len
                else:
                    outward = axis.copy()

                rel = p - shoulder
                along = float(np.dot(rel, axis))
                perpendicular = rel - axis * along

                eased[i] = (
                    shoulder
                    + axis * (along * SLEEVE_LENGTH_EASE)
                    + perpendicular * SLEEVE_WIDTH_EASE
                    + outward * shoulder_width * SHOULDER_OUTWARD_EASE
                ).astype(np.float32)

            else:
                # ------------------------------------------
                # TORSO: holgura principalmente lateral.
                # ------------------------------------------
                rel = p - shoulder_mid
                x = float(np.dot(rel, lateral))
                y = float(np.dot(rel, down))

                # Centro del cuello casi inmóvil; laterales reciben más ease.
                normalized_side = abs(x) / max(
                    1e-6,
                    shoulder_width * 0.50,
                )

                side_strength = self._smoothstep01(
                    (normalized_side - NECK_CENTER_PROTECTION)
                    / max(1e-6, 1.0 - NECK_CENTER_PROTECTION)
                )

                # Hacia abdomen/cintura dejamos algo de holgura incluso en
                # vértices interiores para evitar efecto corsé.
                down_strength = self._smoothstep01(
                    (max(0.0, y / torso_height) - 0.12) / 0.88
                )

                width_strength = max(
                    side_strength,
                    0.38 * down_strength,
                )

                width_scale = (
                    1.0
                    + (TORSO_WIDTH_EASE - 1.0) * width_strength
                )

                eased[i] = (
                    shoulder_mid
                    + lateral * (x * width_scale)
                    + down * (y * TORSO_VERTICAL_EASE)
                ).astype(np.float32)

        return eased

    def initialize(self, mask, anchored_pose):
        self.reset()

        bbox = mask_bbox(mask)
        if bbox is None or not anchored_pose.ready:
            return False

        h, w = mask.shape[:2]
        x1, y1, x2, y2 = bbox

        self.reference_mask = mask.copy()

        spacing = max(22, int(MESH_SPACING))

        # ----------------------------------------------------
        # 1. CONTORNO REAL: cuello, mangas, axilas, cintura.
        # ----------------------------------------------------
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            return False

        main_contour = max(contours, key=cv2.contourArea)

        boundary_points = self._sample_contour_by_distance(
            main_contour,
            CONTOUR_SAMPLE_DISTANCE,
        )

        # ----------------------------------------------------
        # 2. PUNTOS INTERIORES.
        #    Solo entran si el centro esta REALMENTE dentro de la mascara.
        # ----------------------------------------------------
        interior_points = []

        for y in range(y1, y2 + 1, spacing):
            yy = int(np.clip(y, 0, h - 1))

            for x in range(x1, x2 + 1, spacing):
                xx = int(np.clip(x, 0, w - 1))

                if mask[yy, xx] > 0:
                    interior_points.append(
                        np.asarray([float(xx), float(yy)], dtype=np.float32)
                    )

        # Unir borde + interior. El borde va primero para poder marcarlo.
        boundary_points = self._deduplicate_points(boundary_points, min_distance=4.0)
        all_points = list(boundary_points)

        for p in interior_points:
            if not all_points:
                all_points.append(p)
                continue

            arr = np.asarray(all_points, dtype=np.float32)
            d2 = np.sum((arr - p) ** 2, axis=1)

            if float(np.min(d2)) >= 7.0 * 7.0:
                all_points.append(p)

        if len(all_points) < 12:
            return False

        vertices = np.asarray(all_points, dtype=np.float32)
        boundary_count = len(boundary_points)

        # ----------------------------------------------------
        # 3. DELAUNAY + FILTRO POR MASCARA REAL.
        # ----------------------------------------------------
        triangles = self._build_delaunay(vertices, mask, spacing)

        # V4.7: la playera debe ser UNA pieza.
        # Eliminamos islas topologicas antes de compactar vertices.
        triangles = self._largest_connected_triangle_component(triangles)

        if len(triangles) < 6:
            print("[MESH] Muy pocos triangulos validos.")
            return False

        # Quitar vertices que no participan en ningun triangulo.
        used = sorted({idx for tri in triangles for idx in tri})
        remap = {old: new for new, old in enumerate(used)}

        compact_vertices = vertices[used].copy()
        compact_triangles = [
            tuple(remap[idx] for idx in tri)
            for tri in triangles
        ]

        compact_boundary = np.asarray(
            [old_idx < boundary_count for old_idx in used],
            dtype=bool,
        )

        edges = set()
        for a, b, c in compact_triangles:
            for u, v in ((a, b), (b, c), (c, a)):
                edges.add(tuple(sorted((u, v))))

        self.reference_vertices = compact_vertices
        self.current_vertices = compact_vertices.copy()
        self.triangles = compact_triangles
        self.edges = sorted(edges)
        self.boundary_flags = compact_boundary

        self.neighbors = [[] for _ in range(len(compact_vertices))]
        for a, b in self.edges:
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)

        self.vertex_weights = [
            anchored_pose.weights_for_point(v)
            for v in self.reference_vertices
        ]

        # V4.5: clasificacion explicita de manga izquierda/derecha.
        # Se obtiene UNA VEZ del keyframe y no cambia frame a frame.
        sleeve_flags = []
        sleeve_sides = []
        arm_weights = []

        for weights in self.vertex_weights:
            left_weight = 0.0
            right_weight = 0.0

            for name, _a, _b, weight in weights:
                if name == "left_upper_arm":
                    left_weight = max(left_weight, float(weight))
                elif name == "right_upper_arm":
                    right_weight = max(right_weight, float(weight))

            best = max(left_weight, right_weight)

            if best >= ARTICULATED_SLEEVE_MIN_ARM_WEIGHT:
                sleeve_flags.append(True)
                sleeve_sides.append(-1 if left_weight >= right_weight else 1)
                arm_weights.append(best)
            else:
                sleeve_flags.append(False)
                sleeve_sides.append(0)
                arm_weights.append(0.0)

        self.vertex_is_sleeve = np.asarray(sleeve_flags, dtype=bool)
        self.vertex_sleeve_side = np.asarray(sleeve_sides, dtype=np.int8)
        self.vertex_arm_weight = np.asarray(arm_weights, dtype=np.float32)

        # ----------------------------------------------------
        # V4.6: EL ESQUELETO ES ARMADURA, NO MOLDE.
        # ----------------------------------------------------
        self.detected_vertices = self.reference_vertices.copy()

        self.reference_vertices = self._apply_garment_ease(
            self.reference_vertices,
            anchored_pose,
        )
        self.current_vertices = self.reference_vertices.copy()

        # Recalcular pesos sobre la geometría ya holgada.
        self.vertex_weights = [
            anchored_pose.weights_for_point(v)
            for v in self.reference_vertices
        ]

        # Verdad geométrica del KEYFRAME CON HOLGURA.
        # El cloth solver conservará estas distancias.
        self.reference_edge_lengths = {}
        for a, b in self.edges:
            length = float(
                np.linalg.norm(
                    self.reference_vertices[b] - self.reference_vertices[a]
                )
            )
            self.reference_edge_lengths[(a, b)] = max(1e-4, length)

        self.reference_triangle_areas = []
        self.reference_triangle_signed_areas = []

        for a, b, c in self.triangles:
            signed_area = self._triangle_signed_area(
                self.reference_vertices[a],
                self.reference_vertices[b],
                self.reference_vertices[c],
            )

            self.reference_triangle_signed_areas.append(
                float(signed_area)
            )
            self.reference_triangle_areas.append(
                max(1e-4, abs(float(signed_area)))
            )

        self.initialized = True

        print(
            f"[MESH V4.8] {len(self.reference_vertices)} vertices | "
            f"{len(self.triangles)} triangulos | "
            f"{int(np.count_nonzero(self.boundary_flags))} anclas de contorno"
        )
        print(
            "[V4.8] HOLGURA | "
            f"torso={TORSO_WIDTH_EASE:.2f}x | "
            f"manga_ancho={SLEEVE_WIDTH_EASE:.2f}x | "
            f"manga_largo={SLEEVE_LENGTH_EASE:.3f}x"
        )

        return True

    def _residual_for_vertex(self, vertex_absolute, lk):
        if (
            lk is None
            or lk.old_good is None
            or lk.residuals is None
            or len(lk.residuals) == 0
        ):
            return np.zeros(2, dtype=np.float32)

        points = lk.new_good if lk.new_good is not None else lk.old_good
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        residuals = np.asarray(lk.residuals, dtype=np.float32).reshape(-1, 2)

        delta = points - vertex_absolute
        d2 = np.sum(delta * delta, axis=1)

        k = min(LK_RESIDUAL_NEIGHBORS, len(points))
        if k <= 0:
            return np.zeros(2, dtype=np.float32)

        if k < len(points):
            ids = np.argpartition(d2, k - 1)[:k]
        else:
            ids = np.arange(len(points))

        sigma2 = LK_RESIDUAL_SIGMA * LK_RESIDUAL_SIGMA
        weights = np.exp(-d2[ids] / (2.0 * sigma2)).astype(np.float32)
        total = float(weights.sum())

        if total < 1e-6:
            return np.zeros(2, dtype=np.float32)

        residual = np.sum(
            residuals[ids] * weights[:, None],
            axis=0,
        ) / total

        nearest = float(math.sqrt(max(0.0, float(np.min(d2)))))
        proximity = math.exp(-(nearest * nearest) / (2.0 * sigma2))

        gain = (
            LK_RESIDUAL_GAIN
            * proximity
            * max(0.15, lk.confidence)
        )

        return clip_vector(residual * gain, LK_RESIDUAL_MAX)

    def _triangle_region(self, a, b, c):
        """
        Clasificacion TOPOLOGICA fija a partir de los vertices del keyframe.

        Un triangulo que mezcla torso/manga es una BISAGRA, no una placa.
        """
        if (
            self.vertex_is_sleeve is None
            or self.vertex_sleeve_side is None
        ):
            return REGION_TORSO

        ids = (a, b, c)
        sleeve_flags = [bool(self.vertex_is_sleeve[i]) for i in ids]
        sides = [
            int(self.vertex_sleeve_side[i])
            if sleeve_flags[k]
            else 0
            for k, i in enumerate(ids)
        ]

        if not any(sleeve_flags):
            return REGION_TORSO

        if all(sleeve_flags):
            nonzero = [s for s in sides if s != 0]
            if nonzero and all(s == -1 for s in nonzero):
                return REGION_LEFT_SLEEVE
            if nonzero and all(s == 1 for s in nonzero):
                return REGION_RIGHT_SLEEVE

        return REGION_SEAM

    def _edge_sleeve_side(self, a, b):
        if (
            self.vertex_is_sleeve is None
            or self.vertex_sleeve_side is None
        ):
            return 0

        candidates = []
        for idx in (a, b):
            if bool(self.vertex_is_sleeve[idx]):
                side = int(self.vertex_sleeve_side[idx])
                if side != 0:
                    candidates.append(side)

        if not candidates:
            return 0

        if all(s == candidates[0] for s in candidates):
            return candidates[0]

        return 0

    def _raise_for_side(self, side):
        if side < 0:
            return float(self.left_arm_raise)
        if side > 0:
            return float(self.right_arm_raise)
        return max(
            float(self.left_arm_raise),
            float(self.right_arm_raise),
        )

    def _update_arm_raise_factors(self, pose):
        """
        Mide cuanto se separa cada brazo de la direccion 'hacia abajo'
        del torso actual.

        brazo abajo       -> ~0
        brazo diagonal    -> intermedio
        brazo horizontal  -> ~1
        brazo arriba      -> 1

        No usa coordenadas absolutas de pantalla: usa el eje hombros->caderas.
        """
        self.left_arm_raise = 0.0
        self.right_arm_raise = 0.0

        if pose is None or pose.curr_points is None:
            return

        required = (11, 12, 13, 14, 23, 24)
        if not all(pose.valid(i) for i in required):
            return

        p = pose.curr_points
        shoulder_mid = (p[11] + p[12]) * 0.5
        hip_mid = (p[23] + p[24]) * 0.5

        down = hip_mid - shoulder_mid
        down_len = float(np.linalg.norm(down))
        if down_len <= 1e-6:
            return
        down /= down_len

        def factor(shoulder_idx, elbow_idx):
            arm = p[elbow_idx] - p[shoulder_idx]
            arm_len = float(np.linalg.norm(arm))
            if arm_len <= 1e-6:
                return 0.0

            arm /= arm_len
            dot_down = float(np.clip(np.dot(arm, down), -1.0, 1.0))

            # 1-dot: abajo=0, horizontal=1, arriba>1 -> clip.
            raw = float(np.clip(1.0 - dot_down, 0.0, 1.0))

            # Smoothstep para evitar cambios bruscos cerca de la pose relajada.
            return raw * raw * (3.0 - 2.0 * raw)

        self.left_arm_raise = factor(11, 13)
        self.right_arm_raise = factor(12, 14)

    def _is_seam_edge(self, a, b):
        if self.vertex_is_sleeve is None:
            return False

        return bool(self.vertex_is_sleeve[a]) != bool(self.vertex_is_sleeve[b])

    def _edge_ratio_limits(self, a, b):
        """
        V4.5:
        - torso: rigido;
        - interior de manga: puede plegarse un poco;
        - costura manga/torso: MUY controlada.
        """
        if self._is_seam_edge(a, b):
            # V4.8: la costura se convierte progresivamente en bisagra
            # conforme sube SU brazo, no el otro.
            side = self._edge_sleeve_side(a, b)
            raise_factor = self._raise_for_side(side)

            min_ratio = (
                SEAM_RELAXED_MIN_RATIO
                + (SEAM_RAISED_MIN_RATIO - SEAM_RELAXED_MIN_RATIO)
                * raise_factor
            )
            max_ratio = (
                SEAM_RELAXED_MAX_RATIO
                + (SEAM_RAISED_MAX_RATIO - SEAM_RELAXED_MAX_RATIO)
                * raise_factor
            )

            return float(min_ratio), float(max_ratio)

        sleeve = (
            self.vertex_is_sleeve is not None
            and bool(self.vertex_is_sleeve[a])
            and bool(self.vertex_is_sleeve[b])
        )

        if sleeve:
            return (
                CLOTH_SLEEVE_EDGE_MIN_RATIO,
                CLOTH_SLEEVE_EDGE_MAX_RATIO,
            )

        return (
            CLOTH_TORSO_EDGE_MIN_RATIO,
            CLOTH_TORSO_EDGE_MAX_RATIO,
        )

    def _solve_edge_constraints(self, vertices):
        """
        Position Based Dynamics 2D muy ligero.

        No intenta simular gravedad. Su trabajo aqui es solamente impedir que
        una arista del keyframe crezca/encoja de forma absurda por un error de
        skinning o perspectiva.
        """
        out = vertices.copy()

        for a, b in self.edges:
            ref_len = self.reference_edge_lengths.get((a, b))
            if ref_len is None:
                ref_len = self.reference_edge_lengths.get((b, a))

            if ref_len is None or ref_len <= 1e-6:
                continue

            pa = out[a]
            pb = out[b]
            delta = pb - pa
            length = float(np.linalg.norm(delta))

            if length <= 1e-6:
                continue

            min_ratio, max_ratio = self._edge_ratio_limits(a, b)
            min_len = ref_len * min_ratio
            max_len = ref_len * max_ratio

            if length < min_len:
                target = min_len
            elif length > max_len:
                target = max_len
            else:
                continue

            direction = delta / length
            excess = length - target

            if self._is_seam_edge(a, b):
                # La costura corrige principalmente la manga.
                # Así el brazo no arrastra el costado del torso.
                side = self._edge_sleeve_side(a, b)
                raise_factor = self._raise_for_side(side)

                seam_stiffness = (
                    SEAM_RELAXED_STIFFNESS
                    + (
                        SEAM_RAISED_STIFFNESS
                        - SEAM_RELAXED_STIFFNESS
                    ) * raise_factor
                )

                total_correction = (
                    direction
                    * (seam_stiffness * excess)
                ).astype(np.float32)

                a_sleeve = bool(self.vertex_is_sleeve[a])
                b_sleeve = bool(self.vertex_is_sleeve[b])

                # La axila puede plegarse, pero NO arrastra el torso.
                torso_share = min(
                    SEAM_TORSO_CORRECTION_MAX_SHARE,
                    0.08 * (1.0 - 0.75 * raise_factor),
                )
                sleeve_share = 1.0 - torso_share

                if a_sleeve and not b_sleeve:
                    out[a] += total_correction * sleeve_share
                    out[b] -= total_correction * torso_share
                elif b_sleeve and not a_sleeve:
                    out[a] += total_correction * torso_share
                    out[b] -= total_correction * sleeve_share
                else:
                    out[a] += total_correction * 0.5
                    out[b] -= total_correction * 0.5
            else:
                correction = (
                    direction
                    * (0.5 * CLOTH_EDGE_STIFFNESS * excess)
                ).astype(np.float32)

                out[a] += correction
                out[b] -= correction

        return out

    def _solve_area_constraints(self, vertices):
        """
        V4.8: area POR REGION.

        Torso y mangas conservan volumen, pero la costura/axila NO intenta
        conservar el area del keyframe porque debe poder plegarse al levantar
        el brazo.
        """
        out = vertices.copy()
        accum = np.zeros_like(out)
        counts = np.zeros((len(out), 1), dtype=np.float32)

        for tri_idx, (a, b, c) in enumerate(self.triangles):
            if tri_idx >= len(self.reference_triangle_areas):
                continue

            region = self._triangle_region(a, b, c)

            # La costura es una bisagra: no imponer area.
            if region == REGION_SEAM:
                continue

            if region == REGION_TORSO:
                min_ratio = TORSO_AREA_MIN_RATIO
                max_ratio = TORSO_AREA_MAX_RATIO
                stiffness = TORSO_AREA_STIFFNESS
            else:
                min_ratio = SLEEVE_AREA_MIN_RATIO
                max_ratio = SLEEVE_AREA_MAX_RATIO
                stiffness = SLEEVE_AREA_STIFFNESS

            ref_area = self.reference_triangle_areas[tri_idx]
            if ref_area <= 1e-6:
                continue

            pa, pb, pc = out[a], out[b], out[c]
            area = self._triangle_area(pa, pb, pc)
            ratio = area / ref_area

            if min_ratio <= ratio <= max_ratio:
                continue

            target_ratio = float(
                np.clip(ratio, min_ratio, max_ratio)
            )

            if ratio <= 1e-8:
                continue

            desired_scale = math.sqrt(target_ratio / ratio)
            applied_scale = (
                1.0
                + (desired_scale - 1.0) * stiffness
            )

            centroid = (pa + pb + pc) / 3.0

            for idx in (a, b, c):
                proposed = (
                    centroid
                    + (out[idx] - centroid) * applied_scale
                )
                accum[idx] += proposed - out[idx]
                counts[idx, 0] += 1.0

        valid = counts[:, 0] > 0
        if np.any(valid):
            out[valid] += accum[valid] / counts[valid]

        return out

    @staticmethod
    def _best_fit_triangle_shape(reference_tri, current_tri):
        """
        Devuelve la version del triangulo de referencia que mejor coincide
        con la posicion/rotacion/escala ACTUAL, pero sin permitir reflexion.

        Es un Procrustes 2D con rotacion propia (det(R) > 0).
        """
        ref = np.asarray(reference_tri, dtype=np.float32)
        cur = np.asarray(current_tri, dtype=np.float32)

        ref_center = np.mean(ref, axis=0)
        cur_center = np.mean(cur, axis=0)

        x = ref - ref_center
        y = cur - cur_center

        ref_norm2 = float(np.sum(x * x))
        cur_norm2 = float(np.sum(y * y))

        if ref_norm2 <= 1e-8 or cur_norm2 <= 1e-8:
            return cur.copy()

        h = x.T @ y

        try:
            u, _s, vt = np.linalg.svd(h)
        except np.linalg.LinAlgError:
            return cur.copy()

        r = u @ vt

        # Prohibir reflexiones: un triangulo no puede darse vuelta.
        if float(np.linalg.det(r)) < 0.0:
            u[:, -1] *= -1.0
            r = u @ vt

        scale = math.sqrt(cur_norm2 / ref_norm2)
        scale = float(
            np.clip(
                scale,
                TRI_SHAPE_SCALE_MIN,
                TRI_SHAPE_SCALE_MAX,
            )
        )

        target = cur_center + (x @ r) * scale
        return target.astype(np.float32)

    def _solve_triangle_shape_constraints(self, vertices):
        """
        V4.8: SHAPE MATCHING POR PIEZA.

        - torso: conserva forma;
        - manga izquierda: conserva SU forma;
        - manga derecha: conserva SU forma;
        - seam/axila: NO hace shape matching. Es una bisagra.

        Esta separacion evita que al subir un brazo los triangulos de axila
        intenten simultaneamente seguir al brazo y parecerse al keyframe.
        """
        out = vertices.copy()
        accum = np.zeros_like(out)
        weights = np.zeros((len(out), 1), dtype=np.float32)

        for tri_idx, (a, b, c) in enumerate(self.triangles):
            if tri_idx >= len(self.reference_triangle_signed_areas):
                continue

            region = self._triangle_region(a, b, c)

            # Punto central de V4.8: la costura NO es una placa rigida.
            if region == REGION_SEAM:
                continue

            if region == REGION_TORSO:
                base_stiffness = TORSO_SHAPE_STIFFNESS
            else:
                base_stiffness = SLEEVE_SHAPE_STIFFNESS

            ids = (a, b, c)

            ref_tri = np.asarray(
                [
                    self.reference_vertices[a],
                    self.reference_vertices[b],
                    self.reference_vertices[c],
                ],
                dtype=np.float32,
            )

            cur_tri = np.asarray(
                [out[a], out[b], out[c]],
                dtype=np.float32,
            )

            ref_signed = float(
                self.reference_triangle_signed_areas[tri_idx]
            )
            cur_signed = self._triangle_signed_area(
                cur_tri[0],
                cur_tri[1],
                cur_tri[2],
            )

            ref_abs = max(1e-6, abs(ref_signed))
            same_orientation = ref_signed * cur_signed > 0.0
            area_ratio = abs(cur_signed) / ref_abs

            target = self._best_fit_triangle_shape(
                ref_tri,
                cur_tri,
            )

            # Anti-inversion solo DENTRO de una pieza estable.
            if (
                not same_orientation
                or area_ratio < TRI_SIGNED_AREA_MIN_RATIO
            ):
                stiffness = max(
                    base_stiffness,
                    TRI_INVERSION_STIFFNESS,
                )
            else:
                stiffness = base_stiffness

            for local_idx, vertex_idx in enumerate(ids):
                delta = (
                    target[local_idx] - out[vertex_idx]
                ) * stiffness

                accum[vertex_idx] += delta
                weights[vertex_idx, 0] += 1.0

        valid = weights[:, 0] > 0.0
        if np.any(valid):
            out[valid] += accum[valid] / weights[valid]

        return out

    def _solve_cloth_constraints(self, proposed):
        """
        Solver espacial del frame ACTUAL.

        MUY IMPORTANTE:
        - empieza en proposed, que fue calculado desde el KEYFRAME;
        - NO usa current_vertices del frame anterior;
        - por tanto las restricciones no reintroducen drift.
        """
        out = proposed.copy()

        # V4.8 necesita menos iteraciones porque las regiones ya no pelean.
        # Esto tambien reduce el tiempo de malla.
        total_iters = 4

        for _ in range(total_iters):
            # 1) longitudes/costuras
            out = self._solve_edge_constraints(out)

            # 2) volumen/area
            out = self._solve_area_constraints(out)

            # 3) V4.8: forma solo dentro de torso/mangas; seam libre
            out = self._solve_triangle_shape_constraints(out)

            # El solver es una correccion, no una nueva fuente de movimiento.
            delta = out - proposed
            norms = np.linalg.norm(delta, axis=1)

            max_correction = max(
                CLOTH_MAX_CORRECTION_FROM_POSE,
                TOPOLOGY_MAX_CORRECTION_FROM_POSE,
            )

            too_far = norms > max_correction
            if np.any(too_far):
                scale = (
                    max_correction
                    / np.maximum(norms[too_far], 1e-6)
                )
                out[too_far] = (
                    proposed[too_far]
                    + delta[too_far] * scale[:, None]
                )

        # Pasada final por las piezas estables. La costura sigue libre.
        out = self._solve_triangle_shape_constraints(out)

        return out

    def _triangle_is_render_safe(self, tri_idx, a, b, c):
        """
        Fusible visual.

        Si un frame de Pose produce una geometria imposible, preferimos un
        pequeño hueco temporal antes que pintar una enorme vela triangular.
        """
        if (
            tri_idx >= len(self.reference_triangle_areas)
            or self.current_vertices is None
            or self.reference_vertices is None
        ):
            return False

        curr = self.current_vertices
        ref = self.reference_vertices

        ref_area = self.reference_triangle_areas[tri_idx]
        curr_area = self._triangle_area(curr[a], curr[b], curr[c])

        if ref_area <= 1e-6:
            return False

        if curr_area / ref_area > RENDER_TRIANGLE_MAX_AREA_RATIO:
            return False

        for u, v in ((a, b), (b, c), (c, a)):
            ref_len = float(np.linalg.norm(ref[v] - ref[u]))
            curr_len = float(np.linalg.norm(curr[v] - curr[u]))

            if ref_len <= 1e-6:
                continue

            if curr_len / ref_len > RENDER_EDGE_MAX_RATIO:
                return False

        return True

    def update(self, anchored_pose, pose, lk):
        if not self.initialized or not anchored_pose.ready:
            return False

        t0 = time.perf_counter()

        # V4.8: detectar articulacion ANTES de resolver la tela.
        self._update_arm_raise_factors(pose)

        # 1. Siempre desde KEYFRAME: cero acumulacion temporal.
        absolute = np.zeros_like(self.reference_vertices)

        for i, ref_vertex in enumerate(self.reference_vertices):
            if (
                self.vertex_is_sleeve is not None
                and bool(self.vertex_is_sleeve[i])
                and self.vertex_sleeve_side is not None
                and int(self.vertex_sleeve_side[i]) != 0
            ):
                # V4.5: la manga se mueve como pieza articulada.
                absolute[i] = anchored_pose.transform_articulated_sleeve_point(
                    ref_vertex,
                    self.vertex_weights[i],
                    pose,
                    int(self.vertex_sleeve_side[i]),
                    float(self.vertex_arm_weight[i]),
                )
            else:
                absolute[i] = anchored_pose.transform_reference_point(
                    ref_vertex,
                    self.vertex_weights[i],
                    pose,
                )

        # 2. LK sigue siendo residuo pequeño.
        # En mangas reducimos todavía más su autoridad: Pose controla la
        # articulación y LK solo aporta micropliegues.
        corrected = absolute.copy()

        for i in range(len(corrected)):
            residual = self._residual_for_vertex(absolute[i], lk)

            if (
                self.vertex_is_sleeve is not None
                and bool(self.vertex_is_sleeve[i])
            ):
                side = (
                    int(self.vertex_sleeve_side[i])
                    if self.vertex_sleeve_side is not None
                    else 0
                )
                raise_factor = self._raise_for_side(side)

                lk_factor = (
                    SLEEVE_LK_RELAXED_FACTOR
                    + (
                        SLEEVE_LK_RAISED_FACTOR
                        - SLEEVE_LK_RELAXED_FACTOR
                    ) * raise_factor
                )

                residual *= lk_factor

            corrected[i] += residual

        # 3. Regularizacion ESPACIAL del frame actual.
        #    Las anclas de contorno reciben casi nada de suavizado para no
        #    borrar cuello/axilas/extremos de manga.
        regularized = corrected.copy()

        for _ in range(MESH_SPATIAL_ITERS):
            nxt = regularized.copy()

            for i, nbrs in enumerate(self.neighbors):
                if not nbrs:
                    continue

                # V4.8: no suavizar una manga usando vertices del torso o
                # de la otra manga. Cada pieza conserva independencia.
                if (
                    self.vertex_is_sleeve is not None
                    and self.vertex_sleeve_side is not None
                ):
                    i_is_sleeve = bool(self.vertex_is_sleeve[i])
                    i_side = int(self.vertex_sleeve_side[i])

                    compatible = []
                    for j in nbrs:
                        j_is_sleeve = bool(self.vertex_is_sleeve[j])
                        j_side = int(self.vertex_sleeve_side[j])

                        if i_is_sleeve:
                            if j_is_sleeve and j_side == i_side:
                                compatible.append(j)
                        else:
                            if not j_is_sleeve:
                                compatible.append(j)

                    use_nbrs = compatible if compatible else nbrs
                else:
                    use_nbrs = nbrs

                mean_neighbor = np.mean(
                    regularized[use_nbrs],
                    axis=0,
                )

                reg = (
                    BOUNDARY_REGULARIZATION
                    if self.boundary_flags is not None and self.boundary_flags[i]
                    else MESH_SPATIAL_REGULARIZATION
                )

                nxt[i] = (
                    (1.0 - reg) * corrected[i]
                    + reg * mean_neighbor
                )

            regularized = nxt

        # 4. V4.4: restricciones de tela. Sigue siendo 100% espacial y
        # reconstruido desde el keyframe en CADA frame.
        constrained = self._solve_cloth_constraints(
            regularized.astype(np.float32)
        )

        self.current_vertices = constrained.astype(np.float32)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return True

    def _triangle_mask(self, shape):
        """Reconstruye la prenda como union de triangulos NO CONVEXA."""
        h, w = shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)

        if (
            not self.initialized
            or self.current_vertices is None
            or not self.triangles
        ):
            return out

        for tri_idx, (a, b, c) in enumerate(self.triangles):
            # V4.7: topologia bloqueada. Todos los triangulos de la pieza
            # principal se renderizan; el solver corrige la geometria ANTES.
            tri = np.round(
                np.asarray(
                    [
                        self.current_vertices[a],
                        self.current_vertices[b],
                        self.current_vertices[c],
                    ],
                    dtype=np.float32,
                )
            ).astype(np.int32)

            tri[:, 0] = np.clip(tri[:, 0], 0, w - 1)
            tri[:, 1] = np.clip(tri[:, 1], 0, h - 1)

            # No descartamos por deformacion. Shape matching ya corrigio
            # inversiones/colapsos antes de llegar aqui.
            cv2.fillConvexPoly(out, tri, 255)

        # Cierre minimo para eliminar microgrietas de rasterizacion, no para
        # convertir la forma en convexa.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)

        return out

    def roi_mask(self, shape):
        # LK busca alrededor de la PRENDA TRIANGULADA, no alrededor de un hull.
        out = self._triangle_mask(shape)

        if not np.any(out):
            return out

        k = LK_ROI_MARGIN * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(out, kernel, iterations=1)

    def display_mask(self, shape):
        # V4.2: esta es la correccion de la "mancha".
        return self._triangle_mask(shape)

    def draw(self, image):
        if not self.initialized or self.current_vertices is None:
            return

        h, w = image.shape[:2]

        # Dibujar triangulos reales, no una rejilla rectangular ficticia.
        for tri_idx, (a, b, c) in enumerate(self.triangles):
            # Dibujamos exactamente la topologia fija del keyframe.
            tri = [a, b, c]

            for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                p = self.current_vertices[u]
                q = self.current_vertices[v]

                p1 = (
                    int(np.clip(round(float(p[0])), 0, w - 1)),
                    int(np.clip(round(float(p[1])), 0, h - 1)),
                )

                p2 = (
                    int(np.clip(round(float(q[0])), 0, w - 1)),
                    int(np.clip(round(float(q[1])), 0, h - 1)),
                )

                cv2.line(image, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)

        for i, p in enumerate(self.current_vertices):
            x = int(np.clip(round(float(p[0])), 0, w - 1))
            y = int(np.clip(round(float(p[1])), 0, h - 1))

            # V4.5 debug:
            # azul = manga articulada
            # naranja = contorno
            # rojo = interior de torso
            if (
                self.vertex_is_sleeve is not None
                and bool(self.vertex_is_sleeve[i])
            ):
                color = (255, 120, 0)
            elif self.boundary_flags is not None and self.boundary_flags[i]:
                color = (0, 200, 255)
            else:
                color = (0, 90, 255)

            cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)


# ============================================================
# V4.9 - TEXTURED GARMENT RENDERER
# ============================================================

class TexturedGarmentRenderer:
    """
    Piecewise affine texture mapping.

    El PNG ORIGINAL es siempre la fuente. Nunca deformamos el resultado del
    frame anterior, por lo que no hay drift ni borrosidad acumulativa.
    """

    def __init__(self, path):
        self.path = path
        self.texture_bgr = None
        self.texture_alpha = None
        self.source_uv = None
        self.ready = False
        self.bound = False
        self.last_ms = 0.0
        self._load()

    def _load(self):
        self.ready = False
        self.bound = False
        self.source_uv = None

        if not os.path.exists(self.path):
            print(f"[TEXTURE] No encontre: {self.path}")
            print("[TEXTURE] Tracking disponible; textura desactivada.")
            return

        image = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"[TEXTURE] No pude leer: {self.path}")
            return

        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        elif image.shape[2] == 4:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3].copy()
        elif image.shape[2] == 3:
            bgr = image.copy()
            alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        else:
            print("[TEXTURE] Formato no soportado.")
            return

        ys, xs = np.where(alpha > 8)

        if len(xs) >= 16 and len(ys) >= 16:
            x0, x1 = int(np.min(xs)), int(np.max(xs))
            y0, y1 = int(np.min(ys)), int(np.max(ys))
            bgr = bgr[y0:y1 + 1, x0:x1 + 1].copy()
            alpha = alpha[y0:y1 + 1, x0:x1 + 1].copy()

        # Extender el color del tejido a los pixeles transparentes.
        # La silueta final la recorta nuestra malla, no este relleno.
        transparent = (alpha < 8).astype(np.uint8) * 255

        if np.any(transparent):
            try:
                bgr = cv2.inpaint(
                    bgr,
                    transparent,
                    5.0,
                    cv2.INPAINT_TELEA,
                )
            except cv2.error:
                pass

        self.texture_bgr = bgr
        self.texture_alpha = alpha
        self.ready = True

        print(
            f"[TEXTURE] OK | {self.path} | "
            f"{bgr.shape[1]}x{bgr.shape[0]}"
        )

    def reset_binding(self):
        self.source_uv = None
        self.bound = False
        self.last_ms = 0.0

    def bind_to_mesh(self, mesh):
        """
        Fija UV al keyframe usando la geometria detectada por SegFormer ANTES
        del garment-ease. Los indices son los mismos que en reference_vertices.
        """
        self.reset_binding()

        if (
            not self.ready
            or mesh is None
            or not mesh.initialized
            or mesh.detected_vertices is None
            or len(mesh.detected_vertices) == 0
        ):
            return False

        vertices = mesh.detected_vertices.astype(np.float32)

        min_xy = np.min(vertices, axis=0)
        max_xy = np.max(vertices, axis=0)
        span = np.maximum(
            max_xy - min_xy,
            np.asarray([1.0, 1.0], dtype=np.float32),
        )

        uv01 = (vertices - min_xy[None, :]) / span[None, :]

        th, tw = self.texture_bgr.shape[:2]
        mx = float(TEXTURE_SOURCE_MARGIN_X)
        my = float(TEXTURE_SOURCE_MARGIN_Y)

        u = (
            mx + uv01[:, 0] * (1.0 - 2.0 * mx)
        ) * float(max(1, tw - 1))

        v = (
            my + uv01[:, 1] * (1.0 - 2.0 * my)
        ) * float(max(1, th - 1))

        self.source_uv = np.stack([u, v], axis=1).astype(np.float32)

        self.bound = (
            len(self.source_uv)
            == len(mesh.reference_vertices)
        )

        if self.bound:
            print(
                f"[TEXTURE V4.9.1] UV OK | "
                f"V:{len(self.source_uv)} T:{len(mesh.triangles)}"
            )

        return self.bound

    @staticmethod
    def _clip_rect(x, y, width, height, max_w, max_h):
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(max_w, int(x + width))
        y1 = min(max_h, int(y + height))

        if x1 <= x0 or y1 <= y0:
            return None

        return x0, y0, x1, y1

    def render(self, frame, mesh):
        """
        V4.9.1 - SEAM SAFE.

        Diferencia contra V4.9:
        - NO escribimos un triangulo encima del anterior.
        - Cada triangulo aporta a un acumulador float.
        - Hay 1 px de solapamiento raster.
        - En las zonas compartidas se PROMEDIA la textura.
        - Los microhuecos internos se reparan antes del alpha final.
        - La silueta final sigue siendo EXACTAMENTE mesh.display_mask().
        """
        t0 = time.perf_counter()

        if (
            not self.ready
            or not self.bound
            or self.source_uv is None
            or mesh is None
            or not mesh.initialized
            or mesh.current_vertices is None
        ):
            self.last_ms = 0.0
            return frame.copy()

        h, w = frame.shape[:2]
        th, tw = self.texture_bgr.shape[:2]

        # Acumuladores float: dos triangulos que se solapan en una arista
        # comparten el pixel en lugar de pelear por el.
        color_sum = np.zeros(
            (h, w, 3),
            dtype=np.float32,
        )
        weight_sum = np.zeros(
            (h, w),
            dtype=np.float32,
        )
        alpha_sum = np.zeros(
            (h, w),
            dtype=np.float32,
        )

        overlap_px = max(
            0,
            int(TEXTURE_TRIANGLE_OVERLAP_PX),
        )

        for a, b, c in mesh.triangles:
            src_tri = np.asarray(
                [
                    self.source_uv[a],
                    self.source_uv[b],
                    self.source_uv[c],
                ],
                dtype=np.float32,
            )

            dst_tri = np.asarray(
                [
                    mesh.current_vertices[a],
                    mesh.current_vertices[b],
                    mesh.current_vertices[c],
                ],
                dtype=np.float32,
            )

            # Evitar triangulos numericamente degenerados.
            dst_area = abs(
                float(
                    cv2.contourArea(
                        dst_tri.reshape(-1, 1, 2)
                    )
                )
            )
            src_area = abs(
                float(
                    cv2.contourArea(
                        src_tri.reshape(-1, 1, 2)
                    )
                )
            )

            if dst_area < 0.35 or src_area < 0.35:
                continue

            # ---------------- SOURCE CROP ----------------
            sx, sy, sw, sh = cv2.boundingRect(src_tri)

            # 2 px extra para que INTER_LINEAR tenga vecinos válidos incluso
            # exactamente sobre una arista.
            src_pad = 2
            src_rect = self._clip_rect(
                sx - src_pad,
                sy - src_pad,
                sw + 2 * src_pad,
                sh + 2 * src_pad,
                tw,
                th,
            )

            if src_rect is None:
                continue

            sx0, sy0, sx1, sy1 = src_rect

            src_crop = self.texture_bgr[
                sy0:sy1,
                sx0:sx1,
            ]

            alpha_crop = self.texture_alpha[
                sy0:sy1,
                sx0:sx1,
            ]

            if src_crop.size == 0:
                continue

            src_local = (
                src_tri
                - np.asarray(
                    [float(sx0), float(sy0)],
                    dtype=np.float32,
                )
            )

            # ---------------- DESTINATION CROP ----------------
            dx, dy, dw, dh = cv2.boundingRect(dst_tri)

            # El crop también recibe margen para poder dilatar 1 px sin
            # cortar el solapamiento.
            dst_pad = overlap_px + 2
            dst_rect = self._clip_rect(
                dx - dst_pad,
                dy - dst_pad,
                dw + 2 * dst_pad,
                dh + 2 * dst_pad,
                w,
                h,
            )

            if dst_rect is None:
                continue

            dx0, dy0, dx1, dy1 = dst_rect
            out_w = dx1 - dx0
            out_h = dy1 - dy0

            if out_w < 2 or out_h < 2:
                continue

            dst_local = (
                dst_tri
                - np.asarray(
                    [float(dx0), float(dy0)],
                    dtype=np.float32,
                )
            )

            try:
                matrix = cv2.getAffineTransform(
                    src_local.astype(np.float32),
                    dst_local.astype(np.float32),
                )
            except cv2.error:
                continue

            warped = cv2.warpAffine(
                src_crop,
                matrix,
                (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            warped_alpha = cv2.warpAffine(
                alpha_crop,
                matrix,
                (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Máscara exacta del triángulo.
            tri_mask = np.zeros(
                (out_h, out_w),
                dtype=np.uint8,
            )

            tri_int = np.round(
                dst_local
            ).astype(np.int32)

            cv2.fillConvexPoly(
                tri_mask,
                tri_int,
                255,
                lineType=cv2.LINE_AA,
            )

            # Solapamiento raster de 1 px.
            #
            # IMPORTANTE:
            # esto NO ensancha la playera porque al final todo vuelve a ser
            # recortado por garment_mask. Solo elimina grietas internas.
            if overlap_px > 0:
                ksize = overlap_px * 2 + 1
                kernel = np.ones(
                    (ksize, ksize),
                    dtype=np.uint8,
                )
                tri_mask = cv2.dilate(
                    tri_mask,
                    kernel,
                    iterations=1,
                )

            tri_weight = (
                tri_mask.astype(np.float32)
                / 255.0
            )

            if TEXTURE_USE_SOURCE_ALPHA:
                source_weight = (
                    warped_alpha.astype(np.float32)
                    / 255.0
                )
            else:
                source_weight = np.ones(
                    (out_h, out_w),
                    dtype=np.float32,
                )

            contribution = tri_weight * source_weight

            if float(np.max(contribution)) <= 0.0:
                continue

            color_region = color_sum[
                dy0:dy1,
                dx0:dx1,
            ]
            weight_region = weight_sum[
                dy0:dy1,
                dx0:dx1,
            ]
            alpha_region = alpha_sum[
                dy0:dy1,
                dx0:dx1,
            ]

            color_region += (
                warped.astype(np.float32)
                * contribution[:, :, None]
            )
            weight_region += contribution

            if TEXTURE_USE_SOURCE_ALPHA:
                alpha_region += (
                    tri_weight
                    * (
                        warped_alpha.astype(np.float32)
                        / 255.0
                    )
                )
            else:
                alpha_region += tri_weight

        # ---------------- RESOLVE OVERLAPS ----------------
        canvas = np.zeros(
            (h, w, 3),
            dtype=np.uint8,
        )

        covered = weight_sum > 1e-6

        if np.any(covered):
            resolved = np.zeros(
                (h, w, 3),
                dtype=np.float32,
            )

            resolved[covered] = (
                color_sum[covered]
                / weight_sum[covered, None]
            )

            canvas = np.clip(
                resolved,
                0,
                255,
            ).astype(np.uint8)

        # La silueta REAL sigue viniendo de V4.8.
        garment_mask = mesh.display_mask(
            frame.shape
        )

        garment_binary = garment_mask > 0
        coverage_binary = covered

        # ---------------- INTERNAL MICRO-HOLE REPAIR ----------------
        #
        # Solo buscamos pixeles:
        #   1) dentro de la playera V4.8
        #   2) que ningún triángulo llegó a pintar.
        #
        # Un cierre morfológico de 3x3 identifica grietas de rasterización,
        # no zonas grandes donde realmente falte geometría.
        coverage_u8 = (
            coverage_binary.astype(np.uint8)
            * 255
        )

        close_k = max(
            1,
            int(TEXTURE_HOLE_CLOSE_KERNEL),
        )
        if close_k % 2 == 0:
            close_k += 1

        close_kernel = np.ones(
            (close_k, close_k),
            dtype=np.uint8,
        )

        closed_coverage = cv2.morphologyEx(
            coverage_u8,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1,
        )

        micro_holes = (
            garment_binary
            & (closed_coverage > 0)
            & (~coverage_binary)
        )

        if np.any(micro_holes):
            hole_mask = (
                micro_holes.astype(np.uint8)
                * 255
            )

            # Inpaint SOLO esos pocos píxeles. El estampado completo no se
            # suaviza ni se toca.
            try:
                canvas = cv2.inpaint(
                    canvas,
                    hole_mask,
                    float(TEXTURE_HOLE_INPAINT_RADIUS),
                    cv2.INPAINT_TELEA,
                )
            except cv2.error:
                # Fallback local muy pequeño.
                canvas = cv2.medianBlur(
                    canvas,
                    3,
                )

            coverage_binary = (
                coverage_binary
                | micro_holes
            )

        # ---------------- FINAL ALPHA ----------------
        #
        # Clave del borde inferior:
        # NUNCA usamos la máscara dilatada de los triángulos para definir la
        # silueta. El límite final es exactamente garment_mask.
        alpha = (
            garment_mask.astype(np.float32)
            / 255.0
        )

        if TEXTURE_USE_SOURCE_ALPHA:
            # Promedio de alpha donde hubo solapamiento.
            denom = np.maximum(
                weight_sum,
                1e-6,
            )
            src_alpha = np.clip(
                alpha_sum / denom,
                0.0,
                1.0,
            )
            alpha *= src_alpha

        # Si excepcionalmente queda una zona GRANDE sin textura, no la
        # pintamos negra. La dejamos transparente para mostrar cámara.
        alpha *= coverage_binary.astype(np.float32)

        alpha *= float(
            np.clip(
                GARMENT_TEXTURE_OPACITY,
                0.0,
                1.0,
            )
        )

        # Suavizado exclusivamente del contorno final.
        blur = int(TEXTURE_EDGE_BLUR)
        if blur >= 3:
            if blur % 2 == 0:
                blur += 1

            alpha = cv2.GaussianBlur(
                alpha,
                (blur, blur),
                0,
            )

            # Después del blur, impedir que el alpha se extienda demasiado
            # fuera de la silueta real.
            hard_limit = cv2.dilate(
                garment_mask,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            ).astype(np.float32) / 255.0

            alpha *= hard_limit

        alpha3 = alpha[:, :, None]

        out = (
            frame.astype(np.float32)
            * (1.0 - alpha3)
            + canvas.astype(np.float32)
            * alpha3
        )

        out = np.clip(
            out,
            0,
            255,
        ).astype(np.uint8)

        self.last_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        return out


# ============================================================
# RENDER
# ============================================================

def render_scene(
    frame,
    mask,
    pose,
    lk,
    mesh,
    garment_renderer,
    show_texture,
    show_overlay,
    show_mask,
    show_pose,
    show_points,
    show_mesh,
):
    if (
        show_texture
        and garment_renderer is not None
        and garment_renderer.ready
        and garment_renderer.bound
        and mesh is not None
        and mesh.initialized
    ):
        out = garment_renderer.render(frame, mesh)
    else:
        out = frame.copy()

    if (
        mask is not None
        and np.any(mask)
    ):
        if show_overlay and not (
            show_texture
            and garment_renderer is not None
            and garment_renderer.ready
            and garment_renderer.bound
        ):
            layer = np.zeros_like(
                out
            )

            layer[:] = MASK_COLOR

            blended = cv2.addWeighted(
                out,
                1.0 - OVERLAY_ALPHA,
                layer,
                OVERLAY_ALPHA,
                0.0,
            )

            region = mask > 0
            out[region] = blended[region]

        contours, _ = (
            cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        )

        cv2.drawContours(
            out,
            contours,
            -1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if show_mask:
            h, w = out.shape[:2]

            preview_w = min(
                280,
                max(
                    180,
                    w // 4,
                ),
            )

            preview_h = int(
                preview_w
                * h
                / max(w, 1)
            )

            preview = cv2.resize(
                mask,
                (
                    preview_w,
                    preview_h,
                ),
                interpolation=(
                    cv2.INTER_NEAREST
                ),
            )

            preview = cv2.cvtColor(
                preview,
                cv2.COLOR_GRAY2BGR,
            )

            x0 = (
                w
                - preview_w
                - 15
            )

            y0 = 15

            out[
                y0:y0 + preview_h,
                x0:x0 + preview_w,
            ] = preview

            cv2.rectangle(
                out,
                (x0, y0),
                (
                    x0 + preview_w,
                    y0 + preview_h,
                ),
                (255, 255, 255),
                2,
            )

    if (
        show_mesh
        and mesh is not None
    ):
        mesh.draw(out)

    if (
        show_pose
        and pose is not None
    ):
        pose.draw(out)

    if (
        show_points
        and lk is not None
        and lk.points is not None
    ):
        for p in (
            lk.points
            .reshape(-1, 2)
        ):
            x = int(
                round(
                    float(p[0])
                )
            )

            y = int(
                round(
                    float(p[1])
                )
            )

            if (
                0 <= x < out.shape[1]
                and 0 <= y < out.shape[0]
            ):
                cv2.circle(
                    out,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA,
                )

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("SMART FITTING ROOM V4.9.1 - SEAM SAFE TEXTURE")
    print("TEXTURED GARMENT + REGION-AWARE CLOTH + ANCHORED TRACKING")
    print("=" * 78)
    print()
    print(
        "La malla ya NO integra movimiento "
        "frame a frame."
    )
    print(
        "Cada frame vuelve a calcularse "
        "desde el keyframe."
    )
    print()

    cap = cv2.VideoCapture(
        CAMERA_ID,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(
            CAMERA_ID
        )

    if not cap.isOpened():
        raise RuntimeError(
            f"No pude abrir la camara "
            f"{CAMERA_ID}."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
    )

    segmenter = (
        UpperClothesSegmenter()
    )

    pose = PoseTracker()

    anchored_pose = (
        AnchoredPoseModel()
    )

    lk = ResidualLKTracker()

    mesh = AnchoredGarmentMesh()

    # V4.9: shirt.png se carga una sola vez.
    garment_renderer = TexturedGarmentRenderer(
        GARMENT_TEXTURE_PATH
    )

    acquired = False
    auto_enabled = True
    force_segment = False

    pose_lost_frames = 0

    start_time = (
        time.perf_counter()
    )

    next_auto_seg_time = (
        start_time
        + AUTO_START_DELAY
    )

    last_seg_ms = 0.0
    segmentation_count = 0

    show_texture = True
    show_overlay = True
    show_mask = True
    show_pose = True
    show_points = True
    show_mesh = True

    status = (
        "Colocate frente a la camara..."
    )

    prev_time = (
        time.perf_counter()
    )

    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()

            if (
                not ok
                or frame is None
            ):
                print(
                    "[CAM] No pude leer frame."
                )
                break

            now = (
                time.perf_counter()
            )

            dt = max(
                1e-6,
                now - prev_time,
            )

            prev_time = now

            fps = 1.0 / dt

            fps_smooth = (
                fps
                if fps_smooth <= 0
                else (
                    0.90
                    * fps_smooth
                    + 0.10
                    * fps
                )
            )

            # ------------------------------------
            # 1. Pose actual
            # ------------------------------------
            pose_ok = pose.update(
                frame
            )

            # ------------------------------------
            # 2. Si tenemos keyframe:
            #    reconstruir malla DESDE CERO
            # ------------------------------------
            if (
                acquired
                and not force_segment
            ):
                if pose_ok:
                    pose_lost_frames = 0

                    # Primero: geometria absoluta
                    # sin usar movimiento anterior.
                    mesh.update(
                        anchored_pose=(
                            anchored_pose
                        ),
                        pose=pose,
                        lk=None,
                    )

                    roi = mesh.roi_mask(
                        frame.shape
                    )

                    # Segundo: LK observa el frame,
                    # pero solo produce residuos.
                    lk.update(
                        frame_bgr=frame,
                        roi_mask=roi,
                        pose=pose,
                        anchored_pose=(
                            anchored_pose
                        ),
                    )

                    # Tercero: recalcular OTRA VEZ
                    # desde el mismo keyframe,
                    # ahora con residuo LK.
                    mesh.update(
                        anchored_pose=(
                            anchored_pose
                        ),
                        pose=pose,
                        lk=lk,
                    )

                    status = (
                        "ANCHORED OK | "
                        f"LK residuals="
                        f"{lk.residual_count}"
                    )

                else:
                    pose_lost_frames += 1

                    status = (
                        "Pose perdida "
                        f"{pose_lost_frames}/"
                        f"{POSE_LOST_FRAMES_TO_RESEGMENT}"
                    )

                if (
                    pose_lost_frames
                    >= POSE_LOST_FRAMES_TO_RESEGMENT
                ):
                    print(
                        "[ANCHOR] Pose perdida "
                        "demasiados frames. "
                        "Readquiriendo..."
                    )

                    acquired = False
                    pose_lost_frames = 0

                    anchored_pose.reset()
                    mesh.reset()
                    garment_renderer.reset_binding()
                    lk.reset()

                    next_auto_seg_time = (
                        now + 0.20
                    )

            # ------------------------------------
            # 3. Adquisicion / nuevo keyframe
            # ------------------------------------
            should_segment = (
                auto_enabled
                and not acquired
                and now
                >= next_auto_seg_time
            )

            if (
                force_segment
                or should_segment
            ):
                status = (
                    "CREANDO KEYFRAME..."
                )

                temp = frame.copy()

                draw_text(
                    temp,
                    "CREANDO KEYFRAME...",
                    28,
                    58,
                    0.86,
                    3,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    temp,
                )

                cv2.waitKey(1)

                try:
                    if not pose_ok:
                        raise RuntimeError(
                            "Pose no disponible "
                            "para crear keyframe."
                        )

                    mask, seg_ms = (
                        segmenter.segment(
                            frame
                        )
                    )

                    last_seg_ms = seg_ms
                    segmentation_count += 1

                    valid, reason = (
                        mask_is_valid(
                            mask
                        )
                    )

                    if not valid:
                        raise RuntimeError(
                            f"Mascara rechazada: "
                            f"{reason}"
                        )

                    if not anchored_pose.capture(
                        pose
                    ):
                        raise RuntimeError(
                            "No pude capturar "
                            "hombros/brazos/caderas "
                            "para el keyframe."
                        )

                    if not mesh.initialize(
                        mask,
                        anchored_pose,
                    ):
                        raise RuntimeError(
                            "No pude crear "
                            "la malla anclada."
                        )

                    # UV fijas del NUEVO keyframe.
                    # Una textura ausente no invalida el tracking.
                    if garment_renderer.ready:
                        if not garment_renderer.bind_to_mesh(mesh):
                            print("[TEXTURE] UV bind fallo; continuo sin textura.")

                    # La malla empieza exactamente
                    # en el keyframe.
                    roi = mesh.roi_mask(
                        frame.shape
                    )

                    lk.initialize(
                        frame,
                        roi,
                    )

                    acquired = True
                    pose_lost_frames = 0

                    status = (
                        "KEYFRAME ADQUIRIDO"
                    )

                    print(
                        f"[KEYFRAME] OK | "
                        f"SEG={seg_ms:.1f} ms | "
                        f"#{segmentation_count}"
                    )

                except Exception as exc:
                    acquired = False

                    anchored_pose.reset()
                    mesh.reset()
                    garment_renderer.reset_binding()
                    lk.reset()

                    next_auto_seg_time = (
                        now
                        + AUTO_RETRY_SECONDS
                    )

                    status = (
                        "Keyframe rechazado - "
                        "revisa terminal"
                    )

                    print(
                        "\n[V4.2] No se creo "
                        "el keyframe:"
                    )

                    print(
                        repr(exc)
                    )

                    print()

                force_segment = False

            # ------------------------------------
            # 4. Cuenta regresiva
            # ------------------------------------
            if (
                not acquired
                and auto_enabled
                and now
                < next_auto_seg_time
            ):
                remaining = max(
                    0.0,
                    next_auto_seg_time
                    - now,
                )

                status = (
                    f"Colocate... keyframe "
                    f"en {remaining:.1f} s"
                )

            # ------------------------------------
            # 5. Render
            # ------------------------------------
            display_mask = (
                mesh.display_mask(
                    frame.shape
                )
                if acquired
                else None
            )

            display = render_scene(
                frame=frame,
                mask=display_mask,
                pose=pose,
                lk=lk,
                mesh=mesh,
                garment_renderer=garment_renderer,
                show_texture=show_texture,
                show_overlay=show_overlay,
                show_mask=show_mask,
                show_pose=show_pose,
                show_points=show_points,
                show_mesh=show_mesh,
            )

            mode = (
                "ANCHORED"
                if acquired
                else "ACQUIRE"
            )

            draw_text(
                display,
                f"V4.9.1 MODE: {mode}",
                18,
                31,
                0.70,
                2,
            )

            draw_text(
                display,
                f"FPS: {fps_smooth:5.1f}",
                18,
                60,
            )

            draw_text(
                display,
                f"POSE: {pose.last_ms:5.2f} ms "
                f"{'OK' if pose.detected else 'NO'}",
                18,
                89,
            )

            if acquired:
                draw_text(
                    display,
                    f"LK: {lk.last_ms:5.2f} ms "
                    f"PTS:{lk.good_count:3d} "
                    f"RES:{lk.residual_count:3d}",
                    18,
                    118,
                )

                draw_text(
                    display,
                    f"MESH: {mesh.last_ms:5.2f} ms "
                    f"V:{len(mesh.current_vertices)} "
                    f"T:{len(mesh.triangles)}",
                    18,
                    147,
                )

                draw_text(
                    display,
                    f"TEXTURE: {garment_renderer.last_ms:5.2f} ms "
                    f"{'ON' if show_texture and garment_renderer.bound else 'OFF'}",
                    18,
                    176,
                    0.52,
                    2,
                )

                draw_text(
                    display,
                    "V4.9.1: SEAM-SAFE TEXTURE + V4.8 CLOTH",
                    18,
                    205,
                    0.52,
                    2,
                )

                draw_text(
                    display,
                    "LK: RESIDUAL ONLY - NO ACCUMULATION",
                    18,
                    234,
                    0.52,
                    2,
                )

            if last_seg_ms > 0:
                draw_text(
                    display,
                    f"LAST SEG: "
                    f"{last_seg_ms:7.1f} ms "
                    f"COUNT:{segmentation_count}",
                    18,
                    263 if acquired else 118,
                )

            draw_text(
                display,
                status,
                18,
                display.shape[0] - 49,
                0.58,
                2,
            )

            draw_text(
                display,
                "T=textura H=pose G=malla F=LK M=mask "
                "O=overlay R=nuevo-keyframe Q=salir",
                18,
                display.shape[0] - 18,
                0.45,
                1,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            elif key in (
                ord("r"),
                ord("R"),
            ):
                # R no "corrige" el tracker viejo:
                # crea una nueva verdad geometrica.
                acquired = False

                anchored_pose.reset()
                mesh.reset()
                garment_renderer.reset_binding()
                lk.reset()

                force_segment = True

            elif key in (
                ord("c"),
                ord("C"),
            ):
                acquired = False

                anchored_pose.reset()
                mesh.reset()
                garment_renderer.reset_binding()
                lk.reset()

                next_auto_seg_time = (
                    time.perf_counter()
                    + 0.25
                )

                status = (
                    "Readquisicion automatica"
                )

            elif key in (
                ord("p"),
                ord("P"),
            ):
                auto_enabled = (
                    not auto_enabled
                )

                if (
                    auto_enabled
                    and not acquired
                ):
                    next_auto_seg_time = (
                        time.perf_counter()
                        + 0.25
                    )

            elif key in (
                ord("t"),
                ord("T"),
            ):
                show_texture = (
                    not show_texture
                )

            elif key in (
                ord("h"),
                ord("H"),
            ):
                show_pose = (
                    not show_pose
                )

            elif key in (
                ord("g"),
                ord("G"),
            ):
                show_mesh = (
                    not show_mesh
                )

            elif key in (
                ord("f"),
                ord("F"),
            ):
                show_points = (
                    not show_points
                )

            elif key in (
                ord("m"),
                ord("M"),
            ):
                show_mask = (
                    not show_mask
                )

            elif key in (
                ord("o"),
                ord("O"),
            ):
                show_overlay = (
                    not show_overlay
                )

    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
