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
├── main.py            # Script principal do pipeline
├── playground.py      # Teste com uma imagem personalizada
├── README.md          # Documentação do projeto
├── requirements.txt   # Dependências do projeto
├── dowload_dataset.py # Dowload do dataset
├── create_subset.py   # Script para selecionar uma quantidade X de imagens
└── data/              # Pasta para armazenar as imagens
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
3. (Opcional) Selecione um número personalizado de imagens para teste:
   ```bash
   python create_subset.py
   ```
4. Execute o script principal:
   ```bash
   python main.py
   ```
5. Execute o script para testar o modelo com imagens personalizadas:
   ```bash
   python playground.py
   ```

## Observações
- É necessário ter uma conta no Kaggle para baixar a base de dados.
- O pipeline pode ser adaptado para outras bases de imagens com pequenas modificações.

## playground.py - Testando sua IA com imagens individuais
O arquivo `playground.py` permite usar o modelo gerado e testar a classificação de qualquer imagem que você escolher.

### Como usar:
1. Execute o script:
   ```bash
   python playground.py
   ```
2. Quando solicitado, digite o caminho da imagem que deseja testar (pode ser o caminho completo ou apenas o nome do arquivo, se estiver na mesma pasta do script).
3. Veja a classe prevista pelo modelo!
