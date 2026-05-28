"""
Sistema de Controle de Acesso ALPR v2
Utiliza novo modelo treinado com placas completas em carros reais
Decisao: LIBERADO (morador ativo) ou NEGADO (inativo/nao cadastrado)
"""

from ultralytics import YOLO
import cv2
import re
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter
import sys

# Caminhos dos modelos
MODELO_CHARS = "runs/detect/runs/treinamento/modelo_v2-3/weights/best.pt"
MODELO_PLACA = "D:/TCC_ALPR/tcc-alpr/models/license_plate_detector.pt"

# Arquivos
MORADORES_CSV = "moradores.csv"
LOG_CSV = "resultados/logs/log_acessos.csv"


def carregar_moradores():
    """Carrega lista de moradores do CSV"""
    moradores = {}
    with open(MORADORES_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            moradores[row['placa']] = {
                'nome': row['nome'],
                'apto': row['apartamento'],
                'status': row['status']
            }
    print(f"✓ {len(moradores)} moradores cadastrados")
    return moradores


def verificar_acesso(placa, moradores):
    """Verifica se placa tem acesso autorizado"""
    if placa not in moradores:
        return False, "Veículo não cadastrado", None, None
    morador = moradores[placa]
    if morador['status'] != 'ativo':
        return False, "Cadastro inativo", morador['nome'], morador['apto']
    return True, "Acesso autorizado", morador['nome'], morador['apto']


def placa_valida(placa):
    """Verifica se a placa tem formato válido para ser registrada"""
    if not placa:
        return False
    # Placa deve ter exatamente 7 caracteres
    if len(placa) != 7:
        return False
    # Deve seguir padrão Mercosul LLLNLNN ou antigo LLLNNNN
    if re.match(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', placa):  # Mercosul
        return True
    if re.match(r'[A-Z]{3}[0-9]{4}', placa):  # Padrão antigo
        return True
    return False


def registrar_log(placa, decisao, nome, apto, motivo):
    """Registra acesso no log CSV - apenas placas com formato válido"""

    # Filtra placas inválidas para não poluir o log
    if not placa_valida(placa):
        return

    Path("resultados/logs").mkdir(parents=True, exist_ok=True)
    existe = Path(LOG_CSV).exists()
    with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(['data_hora', 'placa', 'decisao', 'nome', 'apartamento', 'motivo'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            placa,
            decisao,
            nome or '-',
            apto or '-',
            motivo
        ])


def limpar_placa(placa):
    """Remove duplicatas e valida formato Mercosul LLLNLNN"""
    if not placa:
        return placa

    # Remove duplicatas consecutivas
    resultado = [placa[0]]
    for char in placa[1:]:
        if char != resultado[-1]:
            resultado.append(char)
    placa = ''.join(resultado)

    # Tenta extrair padrão Mercosul exato
    match = re.search(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', placa)
    if match:
        return match.group()

    # Limpa zeros extras
    placa_limpa = re.sub(r'0+([1-9])', r'\1', placa)
    placa_limpa = re.sub(r'([0-9])0+([A-Z])', r'\1\2', placa_limpa)

    match = re.search(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', placa_limpa)
    if match:
        return match.group()

    if len(placa_limpa) >= 7 and re.match(r'[A-Z]{3}', placa_limpa):
        return placa_limpa[:7]

    return placa_limpa if len(placa_limpa) <= 7 else placa


def processar_video(video_path, moradores):
    print("=" * 60)
    print(f"PROCESSANDO: {video_path}")
    print("=" * 60)

    # Carregar modelos
    modelo_placa = YOLO(MODELO_PLACA)
    modelo_chars = YOLO(MODELO_CHARS)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir vídeo!")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = f"resultados/logs/controle_{Path(video_path).stem}.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    leituras = []
    placas_processadas = set()
    frame_count = 0
    cor_atual = (128, 128, 128)
    texto_atual = ""

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

            # Recortar
            margin = 10
            x1c = max(0, x1 - margin)
            y1c = max(0, y1 - margin)
            x2c = min(frame.shape[1], x2 + margin)
            y2c = min(frame.shape[0], y2 + margin)
            crop = frame[y1c:y2c, x1c:x2c]

            # Remover cabeçalho BRASIL
            h_crop = crop.shape[0]
            crop = crop[int(h_crop * 0.3):, :]

            # Redimensionar
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
                placa_raw = placa_raw.replace('i', 'I').replace('o', 'O')
                placa = limpar_placa(placa_raw)

                if placa and placa_valida(placa):
                    leituras.append(placa)

                    # Verifica acesso quando tiver leituras suficientes
                    if len(leituras) >= 5:
                        placa_mais_comum = Counter(leituras[-10:]).most_common(1)[0][0]

                        if placa_mais_comum not in placas_processadas and placa_valida(placa_mais_comum):
                            autorizado, motivo, nome_morador, apto = verificar_acesso(
                                placa_mais_comum, moradores)

                            placas_processadas.add(placa_mais_comum)
                            decisao = "LIBERADO" if autorizado else "NEGADO"
                            cor_atual = (0, 255, 0) if autorizado else (0, 0, 255)
                            texto_atual = f"{placa_mais_comum} - {decisao}"

                            registrar_log(placa_mais_comum, decisao, nome_morador, apto, motivo)

                            print(f"\n{'='*40}")
                            print(f"PLACA: {placa_mais_comum}")
                            print(f"DECISÃO: {decisao}")
                            if nome_morador:
                                print(f"MORADOR: {nome_morador} - Apto {apto}")
                            print(f"MOTIVO: {motivo}")
                            print(f"{'='*40}\n")

                    # Desenhar bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), cor_atual, 3)

        # Overlay de decisão
        if texto_atual:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, texto_atual, (20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, cor_atual, 3)

        out.write(frame)

    cap.release()
    out.release()

    # Resultado final
    print()
    print("=" * 60)
    print("RESUMO DO PROCESSAMENTO:")
    if leituras:
        placa_final = Counter(leituras).most_common(1)[0][0]
        autorizado, motivo, nome_morador, apto = verificar_acesso(placa_final, moradores)
        decisao = "LIBERADO ✅" if autorizado else "NEGADO ❌"
        print(f"Placa identificada: {placa_final}")
        print(f"Decisão: {decisao}")
        if nome_morador:
            print(f"Morador: {nome_morador} - Apto {apto}")
        print(f"Motivo: {motivo}")
    else:
        print("Nenhuma placa válida detectada!")
    print(f"Log salvo em: {LOG_CSV}")
    print(f"Vídeo salvo em: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    print("=" * 60)
    print("SISTEMA DE CONTROLE DE ACESSO - ALPR v2")
    print("=" * 60)

    moradores = carregar_moradores()

    video = sys.argv[1] if len(sys.argv) > 1 else "videos/IMG_4551.mov"
    processar_video(video, moradores)