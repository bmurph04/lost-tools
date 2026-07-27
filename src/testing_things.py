from external.react.demo.demo_model import SGG_Model
from PIL import Image
import numpy as np
import cv2

ssg_config = '/home/mrw4/workspaces/lost-tools/external/react/checkpoints/config.yml'
ssg_weights = '/home/mrw4/workspaces/lost-tools/external/react/checkpoints/best_model.pth'

ssg_model = SGG_Model(config=ssg_config, weights=ssg_weights)

image = '/home/mrw4/workspaces/EgoObjects-Dataset/categories/test01_1E85AC0A08C00A6DC97A3883107697FC/01/1E85AC0A08C00A6DC97A3883107697FC_01_77.jpg'

frame = Image.open(image).convert("RGB")

img = np.asarray(frame)

annotated_img, graph = ssg_model.predict(img, visu_type="image")

annotated_image_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
cv2.imwrite('output/test.jpg', annotated_image_bgr)

combined = ssg_model.nice_plot(annotated_img, graph)
cv2.imwrite("output/combined.jpg", combined)