import cv2
import numpy as np
import onnx
import onnxruntime


def get_scale_factor(im_h, im_w, ref_size = 512):
    '''短边对齐512'''
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor

##############################################
#  推理部分
##############################################

# 读取图像
im_org = cv2.imread('test.jpg')
h, w, _ = im_org.shape
im = cv2.cvtColor(im_org, cv2.COLOR_BGR2RGB)

# 归一化至 -1 ~ 1
im = (im - 127.5) / 127.5   

# 获取尺度因子
im_h, im_w, im_c = im.shape
x, y = get_scale_factor(im_h, im_w, 512) 

# 缩放图像
im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)

# 准备输入数据
im = np.transpose(im)
im = np.swapaxes(im, 1, 2)
im = np.expand_dims(im, axis = 0).astype('float32')

# 推理预测
session = onnxruntime.InferenceSession('modnet.onnx', None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: im})

# 获取透明度通道
matte = (np.squeeze(result[0]) * 255).astype('uint8')
matte = cv2.resize(matte, dsize=(w, h), interpolation = cv2.INTER_NEAREST)
matte = cv2.cvtColor(matte,cv2.COLOR_GRAY2BGR)

# 新图像合成
bg = np.ones(im_org.shape,np.uint8)
bg[:,:,1]=255 # 定义绿色背景
img_new = matte/255.0*im_org + (1-matte/255.0)*bg
img_new = img_new.astype(np.uint8)
cv2.imwrite('result.jpg', img_new)