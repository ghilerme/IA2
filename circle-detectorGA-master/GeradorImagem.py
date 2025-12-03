import cv2
import numpy as np
import random
import os

def create_demo_image(filename="ImagemDemo.png", width=255, height=255):
    print(f"--- Gerador de Cenários ---")
    print(f"A criar imagem: {width}x{height}...")
    
    # Cria fundo branco (255)
    img = np.ones((height, width), dtype=np.uint8) * 255
    
    # Número aleatório de bolas entre 5 e 12
    num_balls = random.randint(15, 30)
    balls = []
    
    attempts = 0
    max_attempts = 2000
    
    while len(balls) < num_balls and attempts < max_attempts:
        attempts += 1
        
        # Raio aleatório
        r = random.randint(10, 25)
        # Posição aleatória (garantindo que não sai da borda)
        x = random.randint(r, width - r)
        y = random.randint(r, height - r)
        
        # Verifica sobreposição (Colisão)
        overlap = False
        for (bx, by, br) in balls:
            dist = np.sqrt((x - bx)**2 + (y - by)**2)
            # Margem de segurança de 5 pixels entre bolas
            if dist < (r + br + 5):
                overlap = True
                break
        
        if not overlap:
            balls.append((x, y, r))
            # Desenha a bola preta (0)
            cv2.circle(img, (x, y), r, (0), -1)

    print(f"Sucesso! {len(balls)} bolas desenhadas.")
    
    # Salva o arquivo
    cv2.imwrite(filename, img)
    print(f"Imagem salva como '{filename}'. Agora execute o script de busca.")

if __name__ == "__main__":
    create_demo_image()