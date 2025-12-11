import requests
import urllib.parse
import os
import pathlib

def download(link, download_location):
    #  Code from https://github.com/SecFathy/YandexDown/blob/main/YandexCLI.py
    url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={link}"
    response = requests.get(url)
    download_url = response.json()["href"]
    file_name = urllib.parse.unquote(download_url.split("filename=")[1].split("&")[0])
    save_path = os.path.join(download_location, file_name)
    if pathlib.Path(download_location).with_suffix('').exists():
        return

    with open(save_path, "wb") as file:
        download_response = requests.get(download_url, stream=True)
        for chunk in download_response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                file.flush()

    if save_path.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(download_location)
        os.remove(save_path)

    print("Dataset downloaded.")
