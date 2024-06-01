import os

import requests


def download_example(image: str = "getting_started"):
    """Download example images

    Args:
        image (str, optional): Image to download. Defaults to "getting_started".

    Raises:
        ValueError: Unknown image
        ConnectionError: Failed to download image
    """
    assert isinstance(image, str), "image must be a string"
    assert image in ["getting_started"], "image must be one of ['getting_started']"
    if image == "getting_started":
        url = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs"
        file_name = "getting_started.svs"
    else:
        raise ValueError("image must be one of ['getting_started']")
    assert not os.path.exists(file_name), f"{file_name} already exists"
    print(f"Downloading {image} from {os.path.dirname(url)}\nThis may take a few minutes...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"{file_name}", "wb") as f:
            f.write(response.content)
    else:
        raise ConnectionError(f"Failed to download {image}")
