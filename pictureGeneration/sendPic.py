import requests

url = "http://localhost:5002/upload"
file_path = "data/image.jpg"

with open(file_path, 'rb') as f:
    files = {'image': f}
    response = requests.post(url, files=files)

print(response.json())