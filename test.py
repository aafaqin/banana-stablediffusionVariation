# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
from io import BytesIO
from PIL import Image
import sys

# img_name = sys.argv[1:][0]   # to load from argv

img_name="2024_gmc_hummer_ev_suv_4k-t2.jpg"
with open(img_name, "rb") as f:
    bytes = f.read()
    encoded = base64.b64encode(bytes).decode('utf-8')

model_inputs = {'image_data': encoded}

res = requests.post('http://localhost:8000/', json = model_inputs)

image_byte_string = res.json()["image_base64"]

image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output_client.jpg")