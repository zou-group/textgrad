import os
import requests
import hashlib
import platformdirs
from urllib.parse import urlparse

def is_valid_url(url):
    result = urlparse(url)
    return all([result.scheme, result.netloc])


def download_and_cache_image(image_url: str) -> str:
    # Set up cache directory
    assert is_valid_url(image_url), "Invalid URL"
    root = platformdirs.user_cache_dir("textgrad")
    image_cache_dir = os.path.join(root, "image_cache")
    os.makedirs(image_cache_dir, exist_ok=True)

    # Generate a unique filename
    file_name = hashlib.md5(image_url.encode()).hexdigest() + ".jpg"
    cache_path = os.path.join(image_cache_dir, file_name)

    # Check if the image is already cached
    if os.path.exists(cache_path):
        print(f"Image already cached at: {cache_path}")
        with open(cache_path, "rb") as f:
            image_data = f.read()
    else:
        # Download the image
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = response.content

        # Save to cache
        with open(cache_path, "wb") as f:
            f.write(image_data)
        print(f"Image cached at: {cache_path}")

    with open(cache_path, "rb") as image_file:
        return image_file.read()

