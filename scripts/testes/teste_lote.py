"""
Teste em Lote - ALPR v2
Avalia o novo modelo sobre o valid set do dataset msipl
Gera metricas, tabelas e exemplos visuais dark mode
"""

from ultralytics import YOLO
import cv2
import re
import csv
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path
from collections import Counter
import numpy as np
import time

# Configurações
MODELO_CHARS = "runs/detect/runs/treinamento/modelo_v2-3/weights/best.pt"
TEST_IMAGES  = Path("datasets/msipl/valid/images")
TEST_LABELS  = Path("datasets/msipl/valid/labels")
RESULTADOS_DIR    = Path("resultados")
VISUALIZACOES_DIR = Path("resultados/visualizacoes")
RESULTADOS_DIR.mkdir(exist_ok=True)
VISUALIZACOES_DIR.mkdir(exist_ok=True)

# Classes do dataset msipl
CLASSES = ['00','01','02','03','04','05','06','07','08','09',
           'A','B','C','D','E','F','G','H','I','J','K','L',
           'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','i','o']

# Dark mode colors
CORES = {
    'fundo':        '#1a1a2e',
    'fundo_card':   '#16213e',
    'sucesso':      '#00f5a0',
    'erro':         '#ff3860',
    'destaque':     '#00d4ff',
    'texto':        '#e8e8e8',
    'texto_sec':    '#a0a0b8',
    'sucesso_glow': '#00f5a020',
    'erro_glow':    '#ff386020',
    'destaque_glow':'#00d4ff20'
}

# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar_classe(nome):
    """Normaliza classes do dataset msipl para caractere único"""
    if nome == 'i':
        return 'I'
    if nome == 'o':
        return 'O'
    # 00->0, 01->1, 02->2 ... 09->9
    if len(nome) == 2 and nome[0] == '0' and nome[1].isdigit():
        return nome[1]
    return nome

# ============================================================
# GROUND TRUTH
# ============================================================

def ler_ground_truth(label_path):
    """Lê o ground truth do label YOLO ordenando por cx"""
    chars = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_idx = int(parts[0])
                cx = float(parts[1])
                chars.append((cx, CLASSES[cls_idx]))
    chars.sort(key=lambda x: x[0])
    return ''.join([normalizar_classe(c[1]) for c in chars])

# ============================================================
# RECONHECIMENTO
# ============================================================

def reconhecer_placa(modelo, img):
    """Reconhece caracteres na imagem"""
    resultado = modelo(img, conf=0.20, verbose=False)
    if len(resultado[0].boxes) == 0:
        return ""
    chars = []
    for box in resultado[0].boxes:
        cx = float(box.xyxy[0][0])
        cls = int(box.cls[0])
        nome = resultado[0].names[cls]
        chars.append((cx, nome))
    chars.sort(key=lambda x: x[0])
    return ''.join([normalizar_classe(c[1]) for c in chars])

# ============================================================
# VISUALIZAÇÃO DARK MODE
# ============================================================

def criar_visualizacao_dark(img_path, ground_truth, detectado, tipo, categoria, output_name):
    """Cria visualização dark mode profissional"""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    fig = plt.figure(figsize=(14, 8), facecolor=CORES['fundo'])
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1],
                          height_ratios=[0.15, 0.7, 0.15],
                          hspace=0.35, wspace=0.25)

    # --- HEADER ---
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.set_facecolor(CORES['fundo'])

    cor_principal = CORES['sucesso'] if tipo == 'sucesso' else CORES['erro']
    icone  = '✓' if tipo == 'sucesso' else '✕'
    titulo = 'RECONHECIMENTO CORRETO' if tipo == 'sucesso' else f'ERRO · {categoria.upper()}'

    for i, alpha in enumerate([0.05, 0.08, 0.12]):
        ax_header.add_patch(FancyBboxPatch(
            (0.05 - i*0.005, 0.15 - i*0.02), 0.9 + i*0.01, 0.7 + i*0.04,
            boxstyle="round,pad=0.02", facecolor=cor_principal,
            edgecolor='none', alpha=alpha, transform=ax_header.transAxes))

    ax_header.add_patch(FancyBboxPatch(
        (0.05, 0.15), 0.9, 0.7, boxstyle="round,pad=0.02",
        facecolor=CORES['fundo_card'], edgecolor=cor_principal,
        linewidth=2, transform=ax_header.transAxes))
    ax_header.text(0.5, 0.5, f'{icone}  {titulo}',
                   transform=ax_header.transAxes,
                   fontsize=18, fontweight='bold',
                   ha='center', va='center', color=cor_principal)

    # --- IMAGEM ---
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.set_facecolor(CORES['fundo'])
    ax_img.imshow(img)
    ax_img.axis('off')
    for lw, alpha in [(8, 0.2), (5, 0.4), (3, 0.8)]:
        ax_img.add_patch(Rectangle((0, 0), w-1, h-1,
                         linewidth=lw, edgecolor=cor_principal,
                         facecolor='none', alpha=alpha))
    ax_img.text(0.5, -0.06, 'IMAGEM DA PLACA VEICULAR',
                transform=ax_img.transAxes, fontsize=9, fontweight='bold',
                ha='center', va='top', color=CORES['texto_sec'], family='monospace')

    # --- PAINEL INFO ---
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis('off')
    ax_info.set_facecolor(CORES['fundo'])
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    ax_info.add_patch(FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
        facecolor=CORES['fundo_card'], edgecolor=CORES['texto_sec'],
        linewidth=1, alpha=0.5, transform=ax_info.transAxes))

    y = 0.92
    ax_info.text(0.5, y, 'ANÁLISE', fontsize=13, fontweight='bold',
                 ha='center', va='top', color=CORES['destaque'],
                 family='monospace', transform=ax_info.transAxes)
    y -= 0.06
    ax_info.plot([0.2, 0.8], [y, y], color=CORES['destaque'],
                linewidth=2, alpha=0.5, transform=ax_info.transAxes)

    # Placa real
    y -= 0.08
    ax_info.text(0.12, y, 'PLACA REAL', fontsize=9, fontweight='bold',
                 ha='left', va='top', color=CORES['texto_sec'],
                 family='monospace', transform=ax_info.transAxes)
    y -= 0.07
    ax_info.add_patch(FancyBboxPatch(
        (0.12, y - 0.07), 0.76, 0.09, boxstyle="round,pad=0.01",
        facecolor=CORES['destaque_glow'], edgecolor=CORES['destaque'],
        linewidth=1.5, transform=ax_info.transAxes))
    ax_info.text(0.5, y - 0.025, ' '.join(ground_truth),
                 fontsize=15, fontweight='bold', family='monospace',
                 ha='center', va='center', color=CORES['destaque'],
                 transform=ax_info.transAxes)

    # Placa detectada
    y -= 0.14
    ax_info.text(0.12, y, 'LIDO PELO SISTEMA', fontsize=9, fontweight='bold',
                 ha='left', va='top', color=CORES['texto_sec'],
                 family='monospace', transform=ax_info.transAxes)
    y -= 0.07
    det_glow = CORES['sucesso_glow'] if tipo == 'sucesso' else CORES['erro_glow']
    det_edge = CORES['sucesso']      if tipo == 'sucesso' else CORES['erro']
    det_text = CORES['sucesso']      if tipo == 'sucesso' else CORES['erro']
    ax_info.add_patch(FancyBboxPatch(
        (0.12, y - 0.07), 0.76, 0.09, boxstyle="round,pad=0.01",
        facecolor=det_glow, edgecolor=det_edge,
        linewidth=1.5, transform=ax_info.transAxes))
    det_show = ' '.join(detectado) if detectado else '∅ NÃO DETECTADO'
    ax_info.text(0.5, y - 0.025, det_show,
                 fontsize=15, fontweight='bold', family='monospace',
                 ha='center', va='center', color=det_text,
                 transform=ax_info.transAxes)

    # Diagnóstico
    if tipo == 'erro' and detectado:
        y -= 0.14
        ax_info.plot([0.12, 0.88], [y, y],
                    color=CORES['texto_sec'], linewidth=1, alpha=0.3,
                    transform=ax_info.transAxes)
        y -= 0.06
        ax_info.text(0.12, y, 'DIAGNÓSTICO', fontsize=9, fontweight='bold',
                    ha='left', va='top', color=CORES['texto_sec'],
                    family='monospace', transform=ax_info.transAxes)
        y -= 0.07
        if len(detectado) > len(ground_truth):
            diag = f"▸ {len(detectado)-len(ground_truth)} caractere(s) extra(s)"
        elif len(detectado) < len(ground_truth):
            diag = f"▸ {len(ground_truth)-len(detectado)} caractere(s) faltando"
        else:
            trocas = sum(1 for a, b in zip(ground_truth, detectado) if a != b)
            diag = f"▸ {trocas} substituição(ões)"
        ax_info.text(0.14, y, diag, fontsize=9,
                    ha='left', va='top', color=det_text,
                    family='monospace', transform=ax_info.transAxes)

    # --- FOOTER ---
    ax_footer = fig.add_subplot(gs[2, :])
    ax_footer.axis('off')
    ax_footer.set_facecolor(CORES['fundo'])
    ax_footer.text(0.05, 0.5, f'◆ {Path(img_path).name}',
                  transform=ax_footer.transAxes, fontsize=9,
                  ha='left', va='center', color=CORES['texto_sec'], family='monospace')
    ax_footer.text(0.95, 0.5, 'Sistema ALPR v2 · YOLOv8 ◆',
                  transform=ax_footer.transAxes, fontsize=9,
                  ha='right', va='center', color=CORES['texto_sec'], family='monospace')

    plt.savefig(str(VISUALIZACOES_DIR / output_name), dpi=200,
                bbox_inches='tight', facecolor=CORES['fundo'], edgecolor='none')
    plt.close()
    return True

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("TESTE EM LOTE - ALPR v2")
    print("=" * 70)
    print()

    print("Carregando modelo...")
    modelo = YOLO(MODELO_CHARS)
    print("✓ Modelo carregado!")
    print()

    imagens = sorted(list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png")))
    print(f"Total de imagens de teste: {len(imagens)}")
    print()

    resultados    = []
    acertos       = 0
    erros_extras  = []
    erros_faltando= []
    erros_trocados= []
    nao_detectados= []
    confusoes     = Counter()
    ac_pos        = [0]*7
    tot_pos       = [0]*7
    tipos_pos     = ['Letra','Letra','Letra','Número','Letra','Número','Número']

    print("Processando imagens...")
    start = time.time()

    for img_path in imagens:
        label_path = TEST_LABELS / (img_path.stem + '.txt')
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt       = ler_ground_truth(label_path)
        detectado = reconhecer_placa(modelo, img)
        acerto   = (gt == detectado)

        if acerto:
            acertos += 1
        else:
            if not detectado:
                nao_detectados.append((img_path, gt, detectado))
            elif len(detectado) > len(gt):
                erros_extras.append((img_path, gt, detectado))
            elif len(detectado) < len(gt):
                erros_faltando.append((img_path, gt, detectado))
            else:
                erros_trocados.append((img_path, gt, detectado))
                for a, b in zip(gt, detectado):
                    if a != b:
                        confusoes[(a, b)] += 1

        # Acurácia por posição (só placas de 7 chars)
        if len(gt) == 7:
            det7 = detectado.ljust(7)[:7]
            for i in range(7):
                tot_pos[i] += 1
                if i < len(det7) and gt[i] == det7[i]:
                    ac_pos[i] += 1

        resultados.append({
            'imagem':       img_path.name,
            'ground_truth': gt,
            'detectado':    detectado,
            'acerto':       acerto
        })

    elapsed  = time.time() - start
    total    = len(resultados)
    taxa     = acertos / total * 100
    fps      = total / elapsed

    print(f"✓ {total} imagens processadas em {elapsed:.1f}s")
    print()

    # Salvar CSV
    csv_path = RESULTADOS_DIR / "teste_lote_v2.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['imagem','ground_truth','detectado','acerto'])
        writer.writeheader()
        writer.writerows(resultados)
    print(f"✓ Resultados salvos em: {csv_path}")
    print()

    erros_total = total - acertos

    # TABELA 1
    print("=" * 70)
    print("TABELA 1 - DESEMPENHO GERAL DO SISTEMA")
    print("=" * 70)
    print(f"{'Métrica':<35}{'Valor':>20}")
    print("-" * 70)
    print(f"{'Total de imagens testadas':<35}{total:>20}")
    print(f"{'Reconhecimentos corretos':<35}{acertos:>20}")
    print(f"{'Reconhecimentos incorretos':<35}{erros_total:>20}")
    print(f"{'Taxa de acerto (acurácia)':<35}{f'{taxa:.2f}%':>20}")
    print(f"{'Taxa de erro':<35}{f'{erros_total/total*100:.2f}%':>20}")
    print(f"{'FPS (pipeline completo)':<35}{f'{fps:.2f}':>20}")
    print()

    # TABELA 2
    print("=" * 70)
    print("TABELA 2 - DISTRIBUIÇÃO DOS ERROS POR CATEGORIA")
    print("=" * 70)
    print(f"{'Categoria':<35}{'Casos':>10}{'% erros':>15}{'% total':>10}")
    print("-" * 70)
    cats = [
        ("Caracteres extras",   len(erros_extras)),
        ("Caracteres faltando", len(erros_faltando)),
        ("Caracteres trocados", len(erros_trocados)),
        ("Placa não detectada", len(nao_detectados)),
    ]
    for nome, n in cats:
        pe = n/erros_total*100 if erros_total > 0 else 0
        pt = n/total*100
        print(f"{nome:<35}{n:>10}{f'{pe:.1f}%':>15}{f'{pt:.1f}%':>10}")
    print("-" * 70)
    print(f"{'TOTAL':<35}{erros_total:>10}{'100.0%':>15}{f'{erros_total/total*100:.1f}%':>10}")
    print()

    # TABELA 3
    if confusoes:
        print("=" * 70)
        print("TABELA 3 - PRINCIPAIS CONFUSÕES ENTRE CARACTERES")
        print("=" * 70)
        print(f"{'Caractere real':<20}{'Lido como':<20}{'Ocorrências':>15}")
        print("-" * 70)
        for (real, lido), n in confusoes.most_common(10):
            print(f"{real:<20}{lido:<20}{n:>15}")
        print()

    # TABELA 4
    print("=" * 70)
    print("TABELA 4 - ACURÁCIA POR POSIÇÃO DO CARACTERE")
    print("=" * 70)
    print(f"{'Posição':<12}{'Tipo':<12}{'Acertos':>12}{'Total':>10}{'Acurácia':>12}")
    print("-" * 70)
    for i in range(7):
        taxa_pos = ac_pos[i]/tot_pos[i]*100 if tot_pos[i] > 0 else 0
        print(f"{'Pos '+str(i+1):<12}{tipos_pos[i]:<12}{ac_pos[i]:>12}{tot_pos[i]:>10}{f'{taxa_pos:.1f}%':>12}")
    print()

    # VISUALIZAÇÕES
    print("=" * 70)
    print("GERANDO VISUALIZAÇÕES DARK MODE...")
    print("=" * 70)

    random.seed(42)
    acertos_list = [r for r in resultados if r['acerto']]
    for idx, r in enumerate(random.sample(acertos_list, min(5, len(acertos_list))), 1):
        img_path = TEST_IMAGES / r['imagem']
        output = f"01_sucesso_{idx:02d}.png"
        criar_visualizacao_dark(img_path, r['ground_truth'], r['detectado'],
                               'sucesso', 'Reconhecimento Correto', output)
        print(f"  ✓ {output}")

    for idx, (img_path, gt, det) in enumerate(erros_extras[:3], 1):
        output = f"02_erro_extras_{idx:02d}.png"
        criar_visualizacao_dark(img_path, gt, det, 'erro', 'Caracteres Extras', output)
        print(f"  ✓ {output}")

    for idx, (img_path, gt, det) in enumerate(erros_faltando[:3], 1):
        output = f"03_erro_faltando_{idx:02d}.png"
        criar_visualizacao_dark(img_path, gt, det, 'erro', 'Caracteres Faltando', output)
        print(f"  ✓ {output}")

    for idx, (img_path, gt, det) in enumerate(erros_trocados[:2], 1):
        output = f"04_erro_trocado_{idx:02d}.png"
        criar_visualizacao_dark(img_path, gt, det, 'erro', 'Caracteres Trocados', output)
        print(f"  ✓ {output}")

    for idx, (img_path, gt, det) in enumerate(nao_detectados[:2], 1):
        output = f"05_nao_detectado_{idx:02d}.png"
        criar_visualizacao_dark(img_path, gt, det, 'erro', 'Placa Não Detectada', output)
        print(f"  ✓ {output}")

    print()
    print("=" * 70)
    print("✅ CONCLUÍDO!")
    print(f"Acurácia final: {taxa:.2f}%")
    print(f"FPS: {fps:.2f}")
    print(f"Visualizações salvas em: {VISUALIZACOES_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()