import numpy as np
import cv2
import sys
import os

# --- Configuração ---
sys.path.append(os.path.abspath("Python"))
try:
    from gapy.ga import gago
    from gapy.utilities import bits2bytes
except ImportError:
    print("ERRO: Biblioteca 'gapy' não encontrada.")
    sys.exit(1)

IMAGE_FILE = "ImagemDemo.png"
if not os.path.exists(IMAGE_FILE):
    # Fallback caso não tenha o gerador
    print(f"ERRO: {IMAGE_FILE} não encontrado.")
    sys.exit(1)

TARGET_IMG = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
if TARGET_IMG is None:
    print(f"ERRO ao ler {IMAGE_FILE}")
    sys.exit(1)
    
H, W = TARGET_IMG.shape

PRESSURE = {
    "episodes": 4,
    "max_steps": 110,
    "step_penalty": 1.25,
    "move_penalty": 12.0,
    "invalid_bite_penalty": 40.0,
    "edge_penalty": 80.0,
    "ball_reward": 650.0,
    "bonus_chain": 420.0,
    "wander_penalty": 0.25,
    "bite_cooldown": 4,
    "border_penalty": 5.0,
    "border_threshold": 0.08,
    "border_patience": 25
}

EDGE_PROBE_ANGLES = np.deg2rad([0, 60, 120, 180, 240, 300])
ARC_EVAL_ANGLES = np.deg2rad(np.arange(0, 360, 30))
EDGE_OUTER_RADIUS = 11
EDGE_INNER_RADIUS = 5

# --- 2. CÉREBRO ---
class SmartCreatureRNA:
    def __init__(self):
        # ALTERADO: De 11 para 15 para acomodar 4 novos sensores diagonais
        # 2 gradientes + 8 whiskers + 5 probe stats = 15 inputs
        self.input_size = 15  
        self.hidden_size = 14
        self.output_size = 3
        self.w1_shape = (self.input_size + 1, self.hidden_size)
        self.w2_shape = (self.hidden_size + 1, self.output_size)
        self.total_weights = self.w1_shape[0] * self.w1_shape[1] + self.w2_shape[0] * self.w2_shape[1]

    def get_chromosome_size(self):
        return self.total_weights * 8

    def decode_weights(self, bits):
        raw = bits2bytes(bits, dtype="int8").astype(np.float32) / 10.0
        split = self.w1_shape[0] * self.w1_shape[1]
        W1 = raw[:split].reshape(self.w1_shape)
        W2 = raw[split:].reshape(self.w2_shape)
        return W1, W2

    def forward(self, sensors, W1, W2):
        x = np.append(np.array(sensors, dtype=np.float32), 1.0)
        h = np.tanh(x @ W1)
        h = np.append(h, 1.0)
        return np.tanh(h @ W2)

rna_struct = SmartCreatureRNA()

# --- 3. SENSORES COM MÁSCARA LEVE ---
def _edge_probe_features(px, py, img):
    accum_white, accum_dark = [], []
    dir_vec = np.zeros(2, dtype=np.float32)
    hits = 0
    for ang in EDGE_PROBE_ANGLES:
        ox = int(round(px + np.cos(ang) * EDGE_OUTER_RADIUS))
        oy = int(round(py + np.sin(ang) * EDGE_OUTER_RADIUS))
        ix = int(round(px + np.cos(ang) * EDGE_INNER_RADIUS))
        iy = int(round(py + np.sin(ang) * EDGE_INNER_RADIUS))
        outer_val = float(img[oy, ox]) if 0 <= ox < W and 0 <= oy < H else 255.0
        inner_val = float(img[iy, ix]) if 0 <= ix < W and 0 <= iy < H else 0.0
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

def get_advanced_sensors(px, py, dist_map, img):
    h, w = dist_map.shape
    ix = int(np.clip(px, 2, w - 3))
    iy = int(np.clip(py, 2, h - 3))
    gx = (dist_map[iy, ix + 2] - dist_map[iy, ix - 2]) / 15.0
    gy = (dist_map[iy + 2, ix] - dist_map[iy - 2, ix]) / 15.0
    
    whiskers = []
    
    # ALTERADO: Adicionadas as 4 direções diagonais (Total: 8 direções)
    dirs = [
        (0, -1),   # Norte
        (1, -1),   # Nordeste
        (1, 0),    # Leste
        (1, 1),    # Sudeste
        (0, 1),    # Sul
        (-1, 1),   # Sudoeste
        (-1, 0),   # Oeste
        (-1, -1)   # Noroeste
    ]
    
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

# --- 4. AVALIADOR ---
def _sample_ring_values(img, cx, cy, radius, angles):
    vals = []
    for ang in angles:
        rx = int(round(cx + np.cos(ang) * radius))
        ry = int(round(cy + np.sin(ang) * radius))
        if 0 <= rx < W and 0 <= ry < H:
            vals.append(float(img[ry, rx]))
    return vals

def _is_nested(cx, cy, radius, found):
    for fx, fy, fr in found:
        dist = np.hypot(cx - fx, cy - fy)
        if dist + radius * 0.9 <= fr or dist + fr * 0.9 <= radius:
            return True
    return False

def _paint_color_mask(mask, cx, cy, radius):
    c = (int(cx), int(cy))
    cv2.circle(mask, c, int(radius), (0, 255, 255), 2)
    cv2.circle(mask, c, int(radius * 0.8), (255, 0, 0), -1)
    cv2.circle(mask, c, int(radius * 0.4), (0, 0, 255), -1)

def evaluate_bite(cx, cy, img):
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
        return -1
    for r in range(6, min(int(min(H, W) * 0.35), 60), 2):
        border = _sample_ring_values(img, cx, cy, r, ARC_EVAL_ANGLES)
        inner = _sample_ring_values(img, cx, cy, r * 0.55, ARC_EVAL_ANGLES)
        m = min(len(border), len(inner))
        if m < 8:
            continue
        border = np.array(border[:m])
        inner = np.array(inner[:m])
        white_hits = border > 180
        dark_hits = inner < 70
        arc_hits = white_hits & dark_hits
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

_, binary_static = cv2.threshold(TARGET_IMG, 127, 255, cv2.THRESH_BINARY)
DIST_MAP_STATIC = cv2.distanceTransform(binary_static, cv2.DIST_L2, 5)

# --- 5. FITNESS ---
def fitness_function(individual_bits):
    try:
        W1, W2 = rna_struct.decode_weights(individual_bits)
    except Exception:
        return 1e6

    rng = np.random.default_rng(13)
    reward_total = 0.0

    for _ in range(PRESSURE["episodes"]):
        curr = rng.random(2) * 0.8 + 0.1  # afastado das bordas
        prev_pos = curr.copy()
        cooldown = 0
        corner_stick = 0
        local_found = []
        local_reward = 0.0

        for _ in range(PRESSURE["max_steps"]):
            px, py = int(curr[0] * (W - 1)), int(curr[1] * (H - 1))
            sensors = get_advanced_sensors(px, py, DIST_MAP_STATIC, TARGET_IMG)
            dx, dy, bite = rna_struct.forward(sensors, W1, W2)

            move = np.array([dx, dy]) * 0.05
            curr = np.clip(curr + move, 0, 1)
            local_reward -= PRESSURE["step_penalty"]
            local_reward -= np.linalg.norm(curr - prev_pos) * PRESSURE["move_penalty"]
            prev_pos = curr.copy()

            border_dist = min(curr[0], curr[1], 1 - curr[0], 1 - curr[1])
            if border_dist < PRESSURE["border_threshold"]:
                corner_stick += 1
                local_reward -= PRESSURE["border_penalty"]
            else:
                corner_stick = max(corner_stick - 1, 0)

            if corner_stick > PRESSURE["border_patience"]:
                curr = rng.random(2) * 0.8 + 0.1
                prev_pos = curr.copy()
                corner_stick = 0

            if cooldown > 0:
                cooldown -= 1
                continue

            if bite > 0.5:
                radius = evaluate_bite(px, py, TARGET_IMG)
                if radius > 0 and not _is_nested(px, py, radius, local_found):
                    bonus = PRESSURE["ball_reward"] + len(local_found) * PRESSURE["bonus_chain"]
                    local_reward += bonus
                    local_found.append((px, py, radius))
                elif radius == -1:
                    local_reward -= PRESSURE["edge_penalty"]
                else:
                    local_reward -= PRESSURE["invalid_bite_penalty"]
                cooldown = PRESSURE["bite_cooldown"]

        local_reward -= PRESSURE["wander_penalty"] * PRESSURE["max_steps"]
        reward_total += local_reward

    return -reward_total / PRESSURE["episodes"]

# --- 6. EXECUÇÃO ---
def run_learning_creature(best_bits):
    W1, W2 = rna_struct.decode_weights(best_bits)
    final_img = cv2.cvtColor(TARGET_IMG, cv2.COLOR_GRAY2BGR)
    mental_map = TARGET_IMG.copy()
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    found_balls = []

    def find_food(mmap):
        k = np.ones((3, 3), np.uint8)
        cln = cv2.erode(mmap, k, iterations=1)
        pts = cv2.findNonZero(cv2.threshold(cln, 100, 255, cv2.THRESH_BINARY_INV)[1])
        return pts

    food = find_food(mental_map)
    if food is None or len(food) == 0:
        return
    curr = np.array([food[0][0][0] / W, food[0][0][1] / H])
    _, bw = cv2.threshold(mental_map, 127, 255, cv2.THRESH_BINARY)
    curr_dist_map = cv2.distanceTransform(bw, cv2.DIST_L2, 5)

    out = cv2.VideoWriter("criatura_bola.avi", cv2.VideoWriter_fourcc(*"XVID"), 30.0, (W, H))
    found_count = 0
    patience = 0
    loop = 0
    
    # ALTERADO: Atualizado para 8 direções para visualização correta
    dirs_card = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]

    while True:
        loop += 1
        if loop > 20000:
            break
        px, py = int(curr[0] * (W - 1)), int(curr[1] * (H - 1))
        base_frame = final_img.copy()
        sensors = get_advanced_sensors(px, py, curr_dist_map, TARGET_IMG)
        
        # ALTERADO: Range aumentado de 2:6 para 2:10 para cobrir os 8 whiskers
        for i, w_val in enumerate(sensors[2:10]):
            dist_vis = (w_val + 1) / 2 * 28
            dx, dy = dirs_card[i]
            ex, ey = int(px + dx * dist_vis), int(py + dy * dist_vis)
            cv2.line(base_frame, (px, py), (ex, ey), (0, 255, 255), 1)
        
        cv2.circle(base_frame, (px, py), 3, (0, 0, 255), -1)

        dx, dy, bite = rna_struct.forward(sensors, W1, W2)
        hint = sensors[-2:]
        curr = np.clip(curr + np.array([dx, dy]) * 0.04 + np.array(hint) * 0.01, 0, 1)

        if bite > 0.5:
            radius = evaluate_bite(px, py, TARGET_IMG)
            if radius > 0 and mental_map[py, px] <= 100 and not _is_nested(px, py, radius, found_balls):
                found_count += 1
                found_balls.append((px, py, radius))
                cv2.circle(final_img, (px, py), int(radius), (0, 255, 0), 2)
                cv2.circle(mental_map, (px, py), int(radius * 1.4), 255, -1)
                _, bw = cv2.threshold(mental_map, 127, 255, cv2.THRESH_BINARY)
                curr_dist_map = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
                _paint_color_mask(color_mask, px, py, radius)
                patience = 0
            else:
                cv2.line(base_frame, (px - 3, py - 3), (px + 3, py + 3), (0, 0, 255), 1)
                cv2.line(base_frame, (px - 3, py + 3), (px + 3, py - 3), (0, 0, 255), 1)

        patience += 1
        if patience > 150:
            food = find_food(mental_map)
            if food is None or len(food) == 0:
                final_overlay = cv2.addWeighted(final_img, 0.6, color_mask, 0.4, 0)
                for _ in range(30):
                    out.write(final_overlay)
                break
            idx = np.random.randint(len(food))
            tx, ty = food[idx][0]
            curr = np.clip(np.array([tx / W, ty / H]) + np.random.uniform(-0.02, 0.02, 2), 0, 1)
            patience = 0

        display = cv2.addWeighted(base_frame, 0.7, color_mask, 0.3, 0)
        out.write(display)

    out.release()
    final_overlay = cv2.addWeighted(final_img, 0.6, color_mask, 0.4, 0)
    cv2.imwrite("mascaras_coloridas.png", color_mask)
    cv2.imwrite("criatura_overlay.png", final_overlay)
    cv2.imwrite("criatura_resultado.png", final_img)
    print("Arquivos salvos: criatura_resultado.png, criatura_overlay.png, mascaras_coloridas.png, criatura_bola.avi")

def run():
    print("Evoluindo criatura única com máscara leve (8 SENSORES)...")
    nbits = rna_struct.get_chromosome_size()
    ga_options = {
        "PopulationSize": 200,
        "Generations": 140,
        "MutationFcn": 0.28,
        "EliteCount": 2,
        "TournamentSize": 5,
        "SelectionPressure": 1.6
    }
    best_bits, _, fit = gago(fitness_function, nbits, ga_options)
    print(f"Fitness final: {-fit[0]:.2f}")
    np.save("criatura_melhor.npy", best_bits)
    run_learning_creature(best_bits)

if __name__ == "__main__":
    run()