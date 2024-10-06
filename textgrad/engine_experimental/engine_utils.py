from typing import List, Union
import base64

def is_jpeg(data):
    jpeg_signature = b'\xFF\xD8\xFF'
    return data.startswith(jpeg_signature)

def is_png(data):
    png_signature = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'
    return data.startswith(png_signature)

def get_image_type_from_bytes(data):
    if is_jpeg(data):
        return "jpeg"
    elif is_png(data):
        return "png"
    else:
        raise ValueError("Image type not supported, only jpeg and png supported.")

def open_ai_like_formatting(content: List[Union[str, bytes]]) -> List[dict]:
    """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API.
    """
    formatted_content = []
    for item in content:
        if isinstance(item, bytes):
            # For now, bytes are assumed to be images
            image_type = get_image_type_from_bytes(item)
            base64_image = base64.b64encode(item).decode('utf-8')
            formatted_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_type};base64,{base64_image}"
                }
            })
        elif isinstance(item, str):
            formatted_content.append({
                "type": "text",
                "text": item
            })
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")
    return formatted_content