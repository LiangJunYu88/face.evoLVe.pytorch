# Helper function for extracting features from pre-trained models
import torch
import cv2
from time import perf_counter
import numpy as np
import os
import face_recognition
from config import configurations
from backbone.model_irse import IR_50
from face_recognition.face_recognition_cli import image_files_in_folder
from align.face_align import detect_face_align
from string import digits
from PIL import Image, ImageDraw, ImageFont

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)#按1维度求2范数 ，保持输出维度和输入维度相同，tensor[[]]
    output = torch.div(input, norm)
    return output

def extract_feature(img, backbone, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):
    # pre-requisites
    # assert(os.path.exists(img_root))
    # print('Testing Data Root:', img_root)
    # assert (os.path.exists(model_root))
    # print('Backbone Model Root:', model_root)

    # resize image(aligned后的人脸图片) to [128, 128]

    resized = cv2.resize(img, (128, 128))  #从112*112放大成128*128
    #cv2.imshow("extract after resizing", resized)
    #get_image_info(resized)
    # center crop image
    a=int((128-112)/2) # x start
    b=int((128-112)/2+112) # x end
    c=int((128-112)/2) # y start
    d=int((128-112)/2+112) # y end
    ccropped = resized[a:b, c:d] # center crop the image [(8,8),(120,120)]
    # ccropped = img[a:b, c:d]
    ccropped = ccropped[...,::-1] # BGR to RGB
    # cv2.imshow("ccropped",ccropped)
    # flip image horizontally 水平翻转
    flipped = cv2.flip(ccropped, 1)
    # cv2.imshow("flipped", flipped)
    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype = np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)


    # load backbone from a checkpoint
    # print("Loading Backbone Checkpoint '{}'".format(model_root))

    backbone.to(device)

    # extract features
    backbone.eval() # set to evaluation mode  使用预训练好的参数，保证每次计算得到的特征是相同的。

    with torch.no_grad(): #反向传播的时候不求导，提升速度，节约显存

        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu() #返回的值是cpu tensor
            # cuda tensor ->cpu tensor -> numpy

            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())

#     np.save("features.npy", features) 
#     features = np.load("features.npy")
    return features

def predict(embeddings1, embeddings2):

    assert (embeddings1.shape[0] == embeddings2.shape[0])
    # assert (embeddings1.shape[1] == embeddings2.shape[1])

    diff = np.subtract(embeddings1, embeddings2)   # -
    dist = np.sum(np.square(diff))           # 平方和

    return dist



def get_image_info(image):
    print(type(image))  # <class 'numpy.ndarray'>
    print(image.shape)  # (496, 751, 3)
    print(image.size)   # 1117488 496x751x3
    print(image.dtype)  # uint8 3通道每个通道像素点8位
    pixel_data = np.array(image)
    # print(pixel_data)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        './simsunb.ttf', textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def process_frame(faces_db, backbone):
    known_faces_feature = []
    face_locations = []
    face_names = []
    name_list = {
        "Lin Can Yuan": {
            "性别": 'man',
            "年龄": '22',
            "学校": 'SCUT',
        },
        "Liang Jun Yu": {
            "性别": 'man',
            "年龄": '21',
            "学校": 'SCUT',
        },
        "Lei Guang Yu": {
            "性别": 'man',
            "年龄": '20',
            "学校": 'SCUT',
        },
        "Liao Hui Kang": {
            "性别": 'man',
            "年龄": '21',
            "学校": 'SCUT',
        },
        "Li Jia Le": {
            "性别": 'man',
            "年龄": '22',
            "学校": 'SCUT',
        },
    }
    for class_dir in os.listdir(faces_db):
        if not os.path.isdir(os.path.join(faces_db, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(faces_db, class_dir)):

            image = face_recognition.load_image_file(img_path)
            known_face_feature = extract_feature(image, backbone, device=DEVICE, tta=True).numpy()
            known_faces_feature.append(known_face_feature)
            name = os.path.basename(img_path)
            face_names.append(name)

    process_this_frame = True

    while True:
    #while (cap.isOpened()):
        ret, frame = video_capture.read()
        # try:
        #     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  #不缩放，会误检 检测人脸的阈值就需要调高 误检率降低 召回率升高
        # #缩放图像大小来缩短后续操作的推理时间，但是会丢失图像信息  可以适当调低人脸检测的阈值来提高准确率
        # except cv2.error as e:
        #     print("invalid frame")
        # print(ret)
        # ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # cv2.imshow("1/16", small_frame)
        # get_image_info(frame)
        small_frame_PIL = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))  # OPENCV -> PIL
        # Image._show(small_frame_PIL)  #缩小16倍的frame
        small_frame_aligned, bounding_boxes = detect_face_align(small_frame_PIL) #得到一帧内所有人脸和人脸位置
        # print(bounding_boxes)

        # print("已对齐")
        if len(bounding_boxes) != 0:
            new_small_frame = []
            for i in range(len(small_frame_aligned)):
                # Image._show(small_frame_aligned[i])
                new_small_frame.append(cv2.cvtColor(np.asarray(small_frame_aligned[i]), cv2.COLOR_RGB2BGR))  # PIL -> OPENCV
                # print("已转换")
                # cv2.imshow("new_frame", small_frame_aligned[i])
            start1 = perf_counter()
            if process_this_frame:
                face_names_rec = []
                face_locations = []
                for i in range(len(bounding_boxes)):
                    # face_locations = bounding_boxes[i][:4]
                    top = bounding_boxes[i][1]  #y1
                    right = bounding_boxes[i][2]  #x2
                    bottom = bounding_boxes[i][3]  #y2
                    left = bounding_boxes[i][0] #x1
                    face_location = (int(top), int(right), int(bottom), int(left))
                    face_locations.append(face_location)

                face_features = []
                # features = []
                # features = extract_feature(new_small_frame, backbone, device=DEVICE,tta=True).numpy()
                # print(features)
                for i in range(len(new_small_frame)):
                    # cv2.imshow("used for extract", new_small_frame[i])   #112*112
                    # get_image_info(new_small_frame[i])
                    face_features.append(extract_feature(new_small_frame[i], backbone,
                                                device=DEVICE,
                                                tta=True).numpy())  # 提取一帧内每个人脸的特征
                #print(face_features)
                min_dists = []
                min_index = []
                for face_feature in face_features:
                    dists = []
                    similarities = []
                    # print(face_feature)
                    for known_face_feature in known_faces_feature:
                        # print(known_face_feature[0])  #[[]]
                        dists.append(predict(embeddings1=face_feature[0], embeddings2=known_face_feature[0]))
                    # print(dists)
                    min_index.append(np.argmin(dists))
                    min_dists.append(min(dists))
                # print(min_index)
                    # print(min_dists)
                for i in range(len(min_dists)):
                    if min_dists[i] > 1.3:  # 距离>1.3识别为unknown
                        name_rec = "unknown"
                    else:
                        name_rec = face_names[min_index[i]].split('.')[0]  # 去掉后缀
                        name_rec = name_rec.translate(str.maketrans('', '', digits))  # 去除数字后缀
                    face_names_rec.append(name_rec)
                # print(face_names_rec)

            for (top, right, bottom, left), name in zip(face_locations, face_names_rec):

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # scale = 1
                #
                # fontScale = ((abs(right - left)) * 4 * (abs(top - bottom) * 4)) / (500 * 500)  # 缩放名字
                cv2.rectangle(frame, (left, top), (right, bottom), (135, 120, 28), 2)

                #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (135, 120, 28), 2)
                # cv2.rectangle(frame, (left, top - 35), (right + 60, top), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                # 打印名字
                name = name.replace("'", '').replace("[", '').replace("]", '').title()

                cv2.rectangle(frame, (left, top-140), (right, top-105), (128, 150, 255), -1)  # 姓名
                cv2.putText(frame,  name, (left, top - 110), font, 0.5, (0, 0, 0), 1)
                # frame = cv2ImgAddText(cv2.imread(frame), "姓名:" + name, left, top-110, (0, 0, 0), 2)

                cv2.rectangle(frame, (left, top - 105), (right, top - 70), (159, 150, 113), -1)  # 性别
                cv2.rectangle(frame, (left, top - 70), (right, top - 35), (131, 175, 155), -1)  # 年龄
                cv2.rectangle(frame, (left, top - 35), (right, top), (47, 47, 161), -1) # 院校
                # print(name)
                for k,v in name_list.items():
                    # print(k, type(k), len(k))
                    # print(name, type(name), len(name))
                    # print(str(k) == "Lin Can Yuan")  # True
                    # print(name == "Lin Can Yuan")  # 去掉最后一个空格 True
                    # print(v['age'])
                    if str(name) == str(k):
                        cv2.putText(frame, v['性别'], (left, top-70), font, 0.5, (0, 0, 0), 1)
                        cv2.putText(frame, v['年龄'], (left, top-40), font, 0.5, (0, 0, 0), 1)
                        cv2.putText(frame, v['学校'], (left, top-5), font, 0.5, (0, 0, 0), 1)
                        # frame = cv2ImgAddText(cv2.imread(frame), "性别:" + v['性别'], left, top - 75, (0, 0, 0), 2)
                        # frame = cv2ImgAddText(cv2.imread(frame), "年龄:" + v['年龄'], left, top - 40, (0, 0, 0), 2)
                        # frame = cv2ImgAddText(cv2.imread(frame), "学校:" + v['学校'], left, top - 5, (0, 0, 0), 2)

            end1 = perf_counter()
            # print("识别人脸耗时：" + str(end1 - start1))
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # print("no face")
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
    #     out.write(frame)
    # cap.release()
    # out.release()

    video_capture.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    cfg = configurations[1]
    INPUT_SIZE = cfg['INPUT_SIZE']
    BACKBONE_NAME = cfg['BACKBONE_NAME']
    BACKBONE_DICT = {'IR_50': IR_50(INPUT_SIZE)}
    BATCH_SIZE = cfg['BATCH_SIZE']
    backbone = BACKBONE_DICT[BACKBONE_NAME]
    DROP_LAST = cfg['DROP_LAST']
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    DEVICE = cfg['DEVICE']
    GPU_ID = cfg['GPU_ID']
    faces_db = "./data/train_aligned"
    # 模型路径
    model_root = "./model/ms1m-ir50/backbone_ir50_ms1m_epoch120.pth"
    # model_root = "./model/ms1m-ir152/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth"
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 调用摄像头
    #video_capture = cv2.VideoCapture("rtsp://admin:qs@123456@192.168.0.250/Streaming/Channels/1")  # 调用4楼摄像头
    backbone.load_state_dict(torch.load(model_root))
    # 处理视频
    # cap = cv2.VideoCapture('./data/DJ1.MP4')
    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2') # mp42
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.25),
    #         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.25))
    #
    # # out = cv2.VideoWriter('DJ1.mp4', 0x00000021, fps, size)
    # out = cv2.VideoWriter('DJ1.mp4', fourcc, fps, size)
    process_frame(faces_db, backbone)

