"""
Teste do Pipeline ALPR v2
Testa o novo modelo nos videos reais
"""

from ultralytics import YOLO
import cv2
import re
from pathlib import Path
import sys
from collections import Counter


# Caminhos dos modelos
MODELO_CHARS = "runs/detect/runs/treinamento/modelo_v2-3/weights/best.pt"
MODELO_PLACA = "D:/TCC_ALPR/tcc-alpr/models/license_plate_detector.pt"


def limpar_placa(placa):
    """Remove duplicatas consecutivas e valida formato Mercosul LLLNLNN"""
    if not placa:
        return placa

    # Remove duplicatas consecutivas
    resultado = [placa[0]]
    for char in placa[1:]:
        if char != resultado[-1]:
            resultado.append(char)
    placa = ''.join(resultado)

    # Tenta extrair padrão Mercosul exato LLLNLNN
    # Busca 3 letras + 1 número + 1 letra + 2 números
    match = re.search(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', placa)
    if match:
        return match.group()

    # Se não achou padrão exato, tenta limpar zeros extras
    # Ex: MHO08C0207 -> MHO8C27
    # Remove zeros que estão entre letras e números de forma suspeita
    placa_limpa = re.sub(r'0+([1-9])', r'\1', placa)  # 0207 -> 27
    placa_limpa = re.sub(r'([0-9])0+([A-Z])', r'\1\2', placa_limpa)  # 08C -> 8C

    # Tenta novamente após limpeza
    match = re.search(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', placa_limpa)
    if match:
        return match.group()

    # Se ainda tiver mais de 7 chars, pega só os primeiros 7
    if len(placa_limpa) >= 7:
        # Verifica se começa com 3 letras
        if re.match(r'[A-Z]{3}', placa_limpa):
            return placa_limpa[:7]

    return placa_limpa if len(placa_limpa) <= 7 else placa


def testar_video(video_path):
    print("=" * 60)
    print(f"TESTANDO: {video_path}")
    print("=" * 60)

    # Carregar modelos
    print("Carregando modelos...")
    modelo_placa = YOLO(MODELO_PLACA)
    modelo_chars = YOLO(MODELO_CHARS)
    print("✓ Modelos carregados!")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir vídeo: {video_path}")
        return

    # Saída do vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = f"resultados/logs/output_{Path(video_path).stem}.mp4"
    Path("resultados/logs").mkdir(exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    leituras = []
    leituras_raw = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detectar placa
        resultado_placa = modelo_placa(frame, conf=0.2, verbose=False)

        if len(resultado_placa[0].boxes) > 0:
            box = resultado_placa[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_placa = float(box.conf[0]) * 100

            # Recortar placa com margem
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)
            crop = frame[y1:y2, x1:x2]

            # Remover cabeçalho BRASIL (30% superior)
            h_crop = crop.shape[0]
            crop = crop[int(h_crop * 0.3):, :]

            # Redimensionar para altura 120px
            if crop.shape[0] > 0:
                ratio = 120 / crop.shape[0]
                crop = cv2.resize(crop, (int(crop.shape[1] * ratio), 120))

            # Reconhecer caracteres
            resultado_chars = modelo_chars(crop, conf=0.20, verbose=False)

            if len(resultado_chars[0].boxes) > 0:
                chars = []
                for c_box in resultado_chars[0].boxes:
                    cx = float(c_box.xyxy[0][0])
                    cls = int(c_box.cls[0])
                    nome = resultado_chars[0].names[cls]
                    chars.append((cx, nome))

                chars.sort(key=lambda x: x[0])
                placa_raw = ''.join([c[1] for c in chars])

                # Normalizar classes i→I e o→O
                placa_raw = placa_raw.replace('i', 'I').replace('o', 'O')
                leituras_raw.append(placa_raw)

                # Limpar e validar formato Mercosul
                placa = limpar_placa(placa_raw)

                if placa:
                    leituras.append(placa)

                    # Desenhar no frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, placa, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                    print(f"Frame {frame_count}: raw={placa_raw} → limpa={placa} ({conf_placa:.1f}%)")

        out.write(frame)

    cap.release()
    out.release()

    # Resultado final
    print()
    print("=" * 60)
    print("LEITURAS RAW (sem limpeza) - Top 5:")
    if leituras_raw:
        for placa, count in Counter(leituras_raw).most_common(5):
            print(f"  {placa}: {count} frames")

    print()
    print("LEITURAS LIMPAS (após pós-processamento) - Top 5:")
    if leituras:
        mais_comum = Counter(leituras).most_common(5)
        for placa, count in mais_comum:
            print(f"  {placa}: {count} frames")

        placa_final = mais_comum[0][0]
        print()
        print(f"✓ PLACA IDENTIFICADA: {placa_final}")
    else:
        print("Nenhuma placa detectada!")

    print(f"Vídeo salvo em: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    video = sys.argv[1] if len(sys.argv) > 1 else "videos/IMG_4551.mov"
    testar_video(video)