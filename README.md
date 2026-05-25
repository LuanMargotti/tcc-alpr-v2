# TCC ALPR v2 - Reconhecimento Automático de Placas Veiculares

Sistema de reconhecimento de placas do padrão Mercosul para controle de acesso em condomínios.

## Estrutura do Projeto

tcc-alpr-v2/
├── datasets/          # Datasets de treinamento
│   ├── mercosul/      # Dataset Mercosul Roboflow (caracteres isolados)
│   ├── msipl/         # Dataset msipl (placas completas em carros reais)
│   └── ufpr/          # Dataset UFPR-ALPR (placas antigas - Laroca 2018)
├── models/            # Modelos treinados (.pt)
│   ├── detector/      # Modelo de detecção de placas
│   └── caracteres/    # Modelo de reconhecimento de caracteres
├── scripts/           # Scripts Python
│   ├── treinamento/   # Scripts de treino
│   ├── pipeline/      # Pipeline ALPR completo
│   └── testes/        # Scripts de teste e avaliação
├── resultados/        # Resultados dos testes
│   ├── metricas/      # CSVs com métricas
│   ├── visualizacoes/ # Imagens e gráficos
│   └── logs/          # Logs de acesso
├── videos/            # Vídeos para teste
└── docs/              # Documentação


## Tecnologias

- Python 3.11
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch + CUDA 12.8
- GPU: NVIDIA RTX 3060 12GB

## Datasets Utilizados

- **Mercosul Roboflow:** 6.229 imagens de caracteres isolados
- **msipl/pr-hlm2d:** 9.330 imagens de placas completas em carros reais
- **UFPR-ALPR:** 4.500 imagens (Laroca et al., 2018)

## Autor

Luan Costa Margotti - UNESC