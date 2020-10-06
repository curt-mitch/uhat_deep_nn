import numpy as np
from PIL import Image

image = Image.open('UHaT/data/data/characters_test_set/alif/alif (1).jpg')
data = np.asarray(image)

print(data.shape)
