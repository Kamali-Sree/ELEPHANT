import os
import requests
import zipfile
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

FILES_TO_DOWNLOAD = {
    'clips.zip': 'http://datadryad.org/downloads/file_stream/4877911',
    'labels.zip': 'http://datadryad.org/downloads/file_stream/4877909',
    'audio.zip': 'http://datadryad.org/downloads/file_stream/4877912'
}

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for filename, url in FILES_TO_DOWNLOAD.items():
        zip_path = os.path.join(DATA_DIR, filename)
        extract_folder = os.path.join(DATA_DIR, filename.replace('.zip', ''))
        
        if not os.path.exists(zip_path):
            download_file(url, zip_path)
        else:
            print(f"{zip_path} already exists. Skipping download.")
            
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder, exist_ok=True)
            extract_zip(zip_path, extract_folder)
        else:
            print(f"{extract_folder} already exists. Skipping extraction.")

if __name__ == "__main__":
    main()
