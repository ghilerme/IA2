import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.abspath("Python"))
try:
    from gapy.utilities import bits2bytes
except ImportError:
    print("ERRO: Verifique a pasta 'Python/gapy'.")
    sys.exit(1)

EDGE_PROBE_ANGLES = np.deg2rad([0, 60, 120, 180, 240, 300])
ARC_EVAL_ANGLES = np.deg2rad(np.arange(0, 360, 30))
EDGE_OUTER_RADIUS = 11
EDGE_INNER_RADIUS = 5

class MaskRNA:
    def __init__(self):
        self.input_size = 11
        self.hidden_size = 14
        self.output_size = 3
        self.w1_shape = (self.input_size + 1, self.hidden_size)
        self.w2_shape = (self.hidden_size + 1, self.output_size)

    def decode_weights(self, bits):
        data = bits2bytes(bits, dtype="int8").astype(np.float32) / 10.0
        split = self.w1_shape[0] * self.w1_shape[1]
        need = split + self.w2_shape[0] * self.w2_shape[1]
        if len(data) < need:
            data = np.pad(data, (0, need - len(data)))
        W1 = data[:split].reshape(self.w1_shape)
        W2 = data[split:need].reshape(self.w2_shape)
        return W1, W2

    def forward(self, sensors, W1, W2):
        x = np.append(np.array(sensors, dtype=np.float32), 1.0)
        h = np.tanh(x @ W1)
        h = np.append(h, 1.0)
        return np.tanh(h @ W2)

rna = MaskRNA()

def _edge_probe_features(px, py, img):
    h, w = img.shape
    accum_white, accum_dark = [], []
    dir_vec = np.zeros(2, dtype=np.float32)
    hits = 0
    for ang in EDGE_PROBE_ANGLES:
        ox = int(round(px + np.cos(ang) * EDGE_OUTER_RADIUS))
        oy = int(round(py + np.sin(ang) * EDGE_OUTER_RADIUS))
        ix = int(round(px + np.cos(ang) * EDGE_INNER_RADIUS))
        iy = int(round(py + np.sin(ang) * EDGE_INNER_RADIUS))
        outer_val = float(img[oy, ox]) if 0 <= ox < w and 0 <= oy < h else 255.0
        inner_val = float(img[iy, ix]) if 0 <= ix < w and 0 <= iy < h else 0.0
        accum_white.append(outer_val / 255.0)
        accum_dark.append(inner_val / 255.0)
        if outer_val > 180 and inner_val < 80:
            hits += 1
            dir_vec += np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
    if hits > 0:
        dir_vec /= hits
    return (
        float(np.mean(accum_white)),
        float(np.mean(accum_dark)),
        hits / len(EDGE_PROBE_ANGLES),
        dir_vec[0],
        dir_vec[1],
    )

def get_sensors(px, py, dist_map, img):
    h, w = dist_map.shape
    ix = int(np.clip(px, 2, w - 3))
    iy = int(np.clip(py, 2, h - 3))
    gx = (dist_map[iy, ix + 2] - dist_map[iy, ix - 2]) / 15.0
    gy = (dist_map[iy + 2, ix] - dist_map[iy - 2, ix]) / 15.0
    whiskers = []
    dirs = [(0,-1),(1,0),(0,1),(-1,0)]
    max_reach = 28
    for dx, dy in dirs:
        dist = max_reach
        for r in range(1, max_reach):
            nx, ny = int(px + dx * r), int(py + dy * r)
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                dist = r; break
            if img[ny, nx] > 100:
                dist = r; break
        whiskers.append((dist / max_reach) * 2 - 1)
    probe_stats = _edge_probe_features(px, py, img)
    return [gx, gy] + whiskers + list(probe_stats)

def _sample_ring_values(img, cx, cy, radius, angles):
    h, w = img.shape
    vals = []
    for ang in angles:
        rx = int(round(cx + np.cos(ang) * radius))
        ry = int(round(cy + np.sin(ang) * radius))
        if 0 <= rx < w and 0 <= ry < h:
            vals.append(float(img[ry, rx]))
    return vals

def evaluate_bite(cx, cy, img):
    h, w = img.shape
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return -1
    for r in range(6, min(int(min(h, w) * 0.35), 60), 2):
        border = _sample_ring_values(img, cx, cy, r, ARC_EVAL_ANGLES)
        inner = _sample_ring_values(img, cx, cy, r * 0.55, ARC_EVAL_ANGLES)
        m = min(len(border), len(inner))
        if m < 8:
            continue
        border = np.array(border[:m])
        inner = np.array(inner[:m])
        arc_hits = (border > 180) & (inner < 70)
        if arc_hits.mean() < 0.7:
            continue
        contiguous = 0
        best_contig = 0
        for hit in arc_hits:
            contiguous = contiguous + 1 if hit else 0
            best_contig = max(best_contig, contiguous)
        if best_contig >= 3:
            return r
    return -0.5

def _score_circle(cx, cy, radius, img):
    border = _sample_ring_values(img, cx, cy, radius, ARC_EVAL_ANGLES)
    inner = _sample_ring_values(img, cx, cy, radius * 0.55, ARC_EVAL_ANGLES)
    m = min(len(border), len(inner))
    if m < 8:
        return -1
    border = np.array(border[:m])
    inner = np.array(inner[:m])
    arc_hits = (border > 180) & (inner < 70)
    return arc_hits.mean()

def magnetize_circle(cx, cy, r, img):
    best_score = _score_circle(cx, cy, r, img)
    best = (cx, cy, r)
    for _ in range(20):
        improved = False
        for dx, dy, dr in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            cand = (cx + dx, cy + dy, max(4, r + dr))
            score = _score_circle(*cand, img)
            if score > best_score:
                best_score = score
                best = cand
                cx, cy, r = cand
                improved = True
                break
        if not improved:
            break
    return (*best, best_score)

def remove_overlaps(candidates):
    candidates.sort(key=lambda x: x[3], reverse=True)
    unique = []
    for c in candidates:
        cx, cy, cr, score = c
        keep = True
        for ux, uy, ur, _ in unique:
            dist = np.hypot(cx - ux, cy - uy)
            if dist + cr <= ur + 2 or dist < max(cr, ur):
                keep = False
                break
        if keep:
            unique.append(c)
    return unique

def aplicar_modelo(imagem_arquivo, modelo_arquivo="criatura_melhor.npy"):
    if not os.path.exists(modelo_arquivo):
        print("ERRO: 'criatura_melhor.npy' não encontrado.")
        return
    if not os.path.exists(imagem_arquivo):
        print(f"ERRO: Imagem '{imagem_arquivo}' não encontrada.")
        return

    img = cv2.imread(imagem_arquivo, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("ERRO ao carregar imagem.")
        return
    H, W = img.shape
    print(f"Analisando '{imagem_arquivo}' ({W}x{H})...")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    bits = np.load(modelo_arquivo, allow_pickle=False)
    W1, W2 = rna.decode_weights(bits)

    seeds = np.linspace(0.05, 0.95, 22)
    agents = np.array([(x, y) for y in seeds for x in seeds], dtype=np.float32)

    candidates = []
    max_steps = 70
    for state in agents:
        curr = state.copy()
        for _ in range(max_steps):
            px = int(np.clip(curr[0] * (W - 1), 0, W - 1))
            py = int(np.clip(curr[1] * (H - 1), 0, H - 1))
            sensors = get_sensors(px, py, dist_map, img)
            dx, dy, bite = rna.forward(sensors, W1, W2)
            hint = np.array(sensors[-2:])
            curr += np.array([dx, dy]) * 0.05 + hint * 0.01
            curr = np.clip(curr, 0.0, 1.0)
            if bite > 0.45:
                r_eval = evaluate_bite(px, py, img)
                if r_eval > 0:
                    refined = magnetize_circle(px, py, int(r_eval), img)
                    if refined[3] > 0:
                        candidates.append(refined)
                break

    unique_balls = remove_overlaps(candidates)
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, r, _) in unique_balls:
        cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), -1)

    cv2.imwrite("resultado_usar_rna.png", output)
    print(f"{len(unique_balls)} bolas destacadas em 'resultado_usar_rna.png'.")

if __name__ == "__main__":
    aplicar_modelo("ImagemDemo.png")