

import requests
import json
import os

API_URL = "http://10.112.97.1:8001/analyze-frame"

IMAGE_PATH = r"C:\Users\rhans\OneDrive\Desktop\Livevidproj\candidates\v001_converted_3_cam1_0000002.50_yolo_rule.jpg"

JSON_PATH = r"C:\Users\rhans\OneDrive\Desktop\Livevidproj\data\v001_results.json"

# Check both files
for path in [IMAGE_PATH, JSON_PATH]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load the actual JSON data
with open(JSON_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Image and JSON files found.")
print("Sending one request to Member 2...")

with open(IMAGE_PATH, "rb") as image_file:
    response = requests.post(
        API_URL,
        files={
            "image": (
                os.path.basename(IMAGE_PATH),
                image_file,
                "image/jpeg"
            )
        },
        data={
            "metadata": json.dumps(metadata)
        },
        timeout=180
    )

print("Status:", response.status_code)
print("Response:", response.text)
print("Request completed successfully!")
