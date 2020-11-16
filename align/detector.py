import numpy as np
import torch
from torch.autograd import Variable
from align.get_nets import PNet, RNet, ONet
from align.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from align.first_stage import run_first_stage
import cv2
from align.visualization_utils import show_results
from PIL import Image

def detect_faces(image, min_face_size = 20.0,
                 thresholds=[0.7, 0.8, 0.9],
                 nms_thresholds=[0.6, 0.6, 0.6]):
    """
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """

    # LOAD MODELS
    # 用GPU对齐
    # pnet = PNet().cuda()
    # rnet = RNet().cuda()
    # onet = ONet().cuda()
    # 用CPU对齐
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()
    #print(image.size)
    # BUILD AN IMAGE PYRAMID   图像金字塔
    width, height = image.size

    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image  缩放图像
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to  最小图片size
    # minimum face size that we want to detect      最小脸部size
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(image, pnet, scale = s, threshold = thresholds[0])
        bounding_boxes.append(boxes)

    #print(bounding_boxes)
    # collect boxes (and offsets, and scores) from different scales  从不同的尺度收集box(以及偏移量和分数)
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    # use offsets predicted by pnet to transform bounding boxes  利用pnet预测的偏移量变换边界框
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])

    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2
    img_boxes = get_image_boxes(bounding_boxes, image, size = 24)
    #img_boxes = Variable(torch.FloatTensor(img_boxes), volatile = True)
    # 用GPU对齐
    # with torch.no_grad():
    #     img_boxes = Variable(torch.cuda.FloatTensor(img_boxes))
    # 用CPU对齐
    with torch.no_grad():
        img_boxes = Variable(torch.FloatTensor(img_boxes))
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    probs = output[1].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, image, size = 48)
    if len(img_boxes) == 0: 
        return [], []
    #img_boxes = Variable(torch.FloatTensor(img_boxes), volatile = True)
    # 用GPU对齐
    # with torch.no_grad():
    #     img_boxes = Variable(torch.cuda.FloatTensor(img_boxes))
    # 用CPU对齐
    with torch.no_grad():
        img_boxes = Variable(torch.FloatTensor(img_boxes))
    output = onet(img_boxes)
    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))

    offsets = offsets[keep]

    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)

    keep = nms(bounding_boxes, nms_thresholds[2], mode = 'min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks

if __name__=="__main__":
    #image = cv2.imread('../data/ycy.jpg')
    image = Image.open('../data/ycy.jpg')
    image2 = Image.open("C:/Users/Administrator/Desktop/timg.jpg")
    #print(image)   #RGB
    bounding_boxes, landmarks = detect_faces(image2)
    print(bounding_boxes)
    #print(landmarks)
    img = show_results(image2, bounding_boxes, landmarks)
    img.show()
    img.close()
    #img.show()
    #Image._show(img)
