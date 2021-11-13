import requests
from tqdm import tqdm


def download_file_from_google_drive(file_id, destination):
    """
    Downloading files from Google Drive
    """
    URL = "https://drive.google.com/u/1/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)



def get_confirm_token(response):
    """
    Getting the confirm token
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    """
    Saving response to destination
    """
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), desc="downloading"):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
