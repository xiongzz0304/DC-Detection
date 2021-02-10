import mmcv
from PIL import Image
import numpy as np
img_path='/data/xy_data/datasets/3_26/coco/val2017/T2019_7_2.jpg'
img=Image.open(img_path)

grey=img.convert('L')
grey=np.array(grey).astype(np.float32)
img=np.array(img).astype(np.float32)
img=np.dstack((img,grey))
print(img)
