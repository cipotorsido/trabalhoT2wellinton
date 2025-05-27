import os
import shutil
import random
from glob import glob

# Parâmetros
N_IMAGENS = 1000
PASTA_ORIGEM = 'data/seg_train/seg_train'  # Pasta principal do dataset extraído
PASTA_DESTINO = 'data_subset/train'

os.makedirs(PASTA_DESTINO, exist_ok=True)

# Lista todas as classes
classes = [d for d in os.listdir(PASTA_ORIGEM) if os.path.isdir(os.path.join(PASTA_ORIGEM, d))]

# Coleta todas as imagens
todas_imagens = []
for classe in classes:
    imagens = glob(os.path.join(PASTA_ORIGEM, classe, '*.jpg'))
    todas_imagens.extend([(img, classe) for img in imagens])

# Seleciona aleatoriamente N_IMAGENS
amostra = random.sample(todas_imagens, min(N_IMAGENS, len(todas_imagens)))

# Copia as imagens para a nova pasta, mantendo a estrutura de classes
for img_path, classe in amostra:
    destino_classe = os.path.join(PASTA_DESTINO, classe)
    os.makedirs(destino_classe, exist_ok=True)
    shutil.copy(img_path, destino_classe)

print(f"Subset criado com {len(amostra)} imagens em {PASTA_DESTINO}") 