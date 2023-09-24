# Nikita Oltyan

import requests
import zipfile
import os


def download_default_zip():
    """
    Function for downloading default zip with train\test data
    :return:
    Nothing
    """
    folder_path = "../data/raw"
    # TODO:
    # Zip path must be added later after dataset collecting
    zip_url = ""

    # Check if default zip file already downloaded
    if os.path.isfile(f'{folder_path}/movie_classification.zip'):
        return

    os.makedirs(folder_path, exist_ok=True)
    response = requests.get(zip_url)
    filename = zip_url.split('/')[-1]
    file_path = os.path.join(folder_path, filename)

    # Write the downloaded content to the file
    with open(file_path, 'wb') as file:
        file.write(response.content)


def unzip_default_zip():
    zip_path = "../data/raw/movie_classification.zip"
    extract_path = "../data/raw"

    assert os.path.isfile(zip_path), "File isn't exist"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

download_default_zip()
unzip_default_zip()