import base64
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image

with open("resize.png", "rb") as img_file:
    my_string = base64.b64encode(img_file.read())

enocoded=my_string

im = Image.open(BytesIO(base64.b64decode(enocoded)))
# im.save('recovered.png', 'PNG')
img = im
x = image.img_to_array(img)
print(x.shape)