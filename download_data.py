# For zip extraction.
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

# url_dataset = "https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea1/datos4.zip"
url_dataset = "https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea1/datos5.zip"

def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


download_and_unzip(url_dataset, "./data/")
