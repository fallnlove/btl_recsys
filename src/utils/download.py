import requests
import urllib.parse
import os
import pathlib
import shutil

def download(link, download_location, force_download: bool = False):
    download_location = pathlib.Path(download_location)
    if download_location.exists() and not force_download:
        print(f"file {download_location} already exists")
        return
    if download_location.exists() and force_download:
        if download_location.is_dir():
            shutil.rmtree(download_location)
        else:
            download_location.unlink()

    base_api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    
    parsed_link = urllib.parse.urlparse(link)
    query_params = urllib.parse.parse_qs(parsed_link.query)
    
    public_key = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"
    
    api_params = {'public_key': public_key}
    
    if 'path' in query_params:
        api_params['path'] = query_params['path'][0]

    final_url = f"{base_api_url}?{urllib.parse.urlencode(api_params)}"
    response = requests.get(final_url)
    
    if response.status_code != 200:
        print(f"API error: {response.status_code} for {link}")
        return

    download_url = response.json().get("href")
    
    try:
        file_name = urllib.parse.unquote(download_url.split("filename=")[1].split("&")[0])
    except IndexError:
        file_name = api_params.get('path', 'downloaded_file').strip("/")

    save_path = download_location.parent / file_name
    
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Download: {file_name}")
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
    
    if save_path.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(download_location.parent)
        os.remove(save_path)
