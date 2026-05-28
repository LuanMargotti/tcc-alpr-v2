"""
Treinamento do Modelo ALPR v2
Dataset: msipl/pr-hlm2d (9.330 imagens de placas completas em carros reais)
Modelo: YOLOv8n
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

def atualizar_yaml():
    data_yaml_path = Path("datasets/msipl/data.yaml")
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    data['train'] = str(Path("datasets/msipl/train/images").resolve())
    data['val']   = str(Path("datasets/msipl/valid/images").resolve())
    data['test']  = str(Path("datasets/msipl/test/images").resolve())
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, allow_unicode=True)
    
    print("✓ data.yaml atualizado!")
    print(f"  Train: {data['train']}")
    print(f"  Valid: {data['val']}")
    print(f"  Test:  {data['test']}")
    print(f"  Classes: {data['nc']}")
    print()
    
    return data_yaml_path

def treinar():
    data_yaml_path = atualizar_yaml()
    
    modelo = YOLO("yolov8n.pt")
    
    print("🚀 Iniciando treinamento...")
    print("  Modelo: YOLOv8n")
    print("  Dataset: msipl (9.330 imagens)")
    print("  Épocas: 50")
    print("  Batch: 16")
    print()
    
    modelo.train(
        data=str(data_yaml_path),
        epochs=50,
        batch=16,
        imgsz=640,
        optimizer="Adam",
        patience=10,
        project="runs/treinamento",
        name="modelo_v2",
        device=0,
        verbose=True,
        workers=4
    )
    
    print()
    print("=" * 60)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("Modelo salvo em: runs/treinamento/modelo_v2/weights/best.pt")
    print("=" * 60)

if __name__ == '__main__':
    treinar()