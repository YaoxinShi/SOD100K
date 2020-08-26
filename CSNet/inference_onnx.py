import onnxruntime as nxrun
import numpy as np
from skimage.transform import resize
from skimage import io
import cv2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sess = nxrun.InferenceSession("out.onnx")

img = io.imread("C:\\ws\\code\\SOD\\SOD100K\\CSNet\\datasets\\sal\\ECSSD\\images\\forza_540p.jpg")
img = np.rollaxis(img, 2, 0) 

print("The model expects input shape: ", sess.get_inputs()[0].shape)
print("The shape of the Image is: ", img.shape)
original_w = img.shape[2]
original_h = img.shape[1]
inference_w = sess.get_inputs()[0].shape[3]
inference_h = sess.get_inputs()[0].shape[2]
print(original_w,original_h,inference_w,inference_h)

# ximg = np.random.rand(1, 3, 224, 224).astype(np.float32)
img224 = resize(img / 255, (3, inference_h, inference_w), anti_aliasing=True)
ximg = img224[np.newaxis, :, :, :]
ximg = ximg.astype(np.float32)

print("@@@@@@")
print(ximg)
print("@@@@@@")

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
result = sess.run(None, {input_name: ximg})

print("@@@@@@")
print(result)
print("@@@@@@")

result = np.squeeze(result) # (1, 224, 224)
result = np.squeeze(result) # (224, 224)

result = sigmoid(result)
print("@@@@@@")
print (result)
print("@@@@@@")
 
result = cv2.resize(result, (original_w, original_h)) ## to 960x540
cv2.imwrite('out_onnx.png', (result*255).astype(np.uint8))