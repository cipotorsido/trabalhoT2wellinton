import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Inicializa a API do Kaggle
def download_and_extract():
    api = KaggleApi()
    api.authenticate()

    os.makedirs('data', exist_ok=True)
    print("Baixando o dataset Intel Image Classification...")
    api.dataset_download_files('puneet6060/intel-image-classification', path='data', unzip=True)
    print("Download e extração concluídos!")

if __name__ == "__main__":
    download_and_extract() 