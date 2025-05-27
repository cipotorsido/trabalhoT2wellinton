# Projeto: Classificação de Imagens - Intel Image Classification (Kaggle)

## Descrição do Projeto
Este projeto tem como objetivo construir um pipeline completo de classificação de imagens utilizando PyTorch, empregando a base de dados externa **Intel Image Classification** do Kaggle. O pipeline abrange desde o pré-processamento dos dados, definição e treinamento do modelo, até a avaliação e análise crítica dos resultados.

## Base de Dados: Intel Image Classification (Kaggle)
- **Fonte:** [Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Tamanho:** 25.000 imagens divididas em 6 classes (buildings, forest, glacier, mountain, sea, street)
- **Formato das imagens:** JPEG, 150x150 pixels
- **Tarefa:** Classificação multiclasse (6 classes)
- **Tipo de tarefa:** Classificação de imagens

### Principais Desafios
- **Variabilidade:** Imagens de diferentes ambientes naturais e urbanos.
- **Ruído:** Algumas imagens podem conter objetos irrelevantes ou baixa qualidade.
- **Desbalanceamento:** Algumas classes podem ter mais exemplos que outras.

## Estrutura do Projeto
```
Atv_T2/
│
├── main.py           # Script principal do pipeline
├── README.md         # Documentação do projeto
├── requirements.txt  # Dependências do projeto
├── data/             # Pasta para armazenar as imagens
└── utils.py          # Funções auxiliares (opcional)
```

## Pipeline
1. **Pesquisa e documentação**
2. **Pré-processamento dos dados**
3. **Definição e treinamento do modelo em PyTorch**
4. **Avaliação de métricas (acurácia, F1-score, matriz de confusão)**
5. **Análise crítica dos resultados**

## Como rodar o projeto
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Baixe a base de dados do Kaggle e extraia para a pasta `data/`:
   ```bash
   python download_dataset.py
   ```
3. Execute o script principal:
   ```bash
   python main.py
   ```

## Observações
- É necessário ter uma conta no Kaggle para baixar a base de dados.
- O pipeline pode ser adaptado para outras bases de imagens com pequenas modificações. 