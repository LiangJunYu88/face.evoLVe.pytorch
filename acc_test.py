import numpy as np
from config import configurations
from face_rec import extract_feature
from backbone.model_irse import IR_50
import torch
import face_recognition


def calculate_accuracy(threshold, image_root1, image_root2, actual_issame):

    image1 = face_recognition.load_image_file(image_root1)
    image2 = face_recognition.load_image_file(image_root2)
    embeddings1 = extract_feature(image1, backbone=backbone, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True).numpy()
    embeddings2 = extract_feature(image2, backbone=backbone, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True).numpy()
    diff = np.subtract(embeddings1, embeddings2)

    dist = np.sum(np.square(diff))

    predict_issame = np.less(dist, threshold)
    print(predict_issame)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tp,fp,tn,fn

def get_img_pairs_list(pairs_txt_path, img_path):
    """ 指定图片组合及其所在文件，返回各图片对的绝对路径
        Args:
            pairs_txt_path：图片pairs文件，里面是6000对图片名字的组合
            img_path：图片所在文件夹
        return:
            img_pairs_list：深度为2的list，每一个二级list存放的是一对图片的绝对路径
    """
    file = open(pairs_txt_path, 'r')
    img_pairs_list, labels = [], []
    while 1:
        img_pairs = []
        line = file.readline().replace('\n', '')
        if line == '':
            break
        line_list = line.split('\t')
        if len(line_list) == 3:
            # 图片路径示例：
            # 'C:\Users\thinkpad1\Desktop\image_set\lfw_funneled\Tina_Fey\Tina_Fey_0001.jpg'
            img_pairs.append(
                img_path + '\\' + line_list[0] + '\\' + line_list[0] + '_' + ('000' + line_list[1])[-4:] + '.jpg')
            img_pairs.append(
                img_path + '\\' + line_list[0] + '\\' + line_list[0] + '_' + ('000' + line_list[2])[-4:] + '.jpg')
            labels.append(1)
        elif len(line_list) == 4:
            img_pairs.append(
                img_path + '\\' + line_list[0] + '\\' + line_list[0] + '_' + ('000' + line_list[1])[-4:] + '.jpg')
            img_pairs.append(
                img_path + '\\' + line_list[2] + '\\' + line_list[2] + '_' + ('000' + line_list[3])[-4:] + '.jpg')
            labels.append(0)
        else:
            continue

        img_pairs_list.append(img_pairs)
    return img_pairs_list

def get_image_pairs():
    img_pairs_list =get_img_pairs_list(pairs_txt_path="C:\\Users\\Administrator\\Desktop\\face_recognition-master\\examples\\knn_examples\\not_pair.txt", img_path="C:\\Users\\Administrator\\Desktop\\face.evoLVe.PyTorch-master\\data\\LFW_aligned")
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(2970):
        img_pairs =img_pairs_list[i]
        img_root1 = str(img_pairs[0])
        print(i)
        img_root2 = str(img_pairs[1])
        tp2, fp2, tn2, fn2 = calculate_accuracy(threshold=1.45, image_root1=img_root1, image_root2=img_root2, actual_issame=False)
        tp = tp + tp2
        fp = fp + fp2
        tn = tn + tn2
        fn = fn + fn2
        print(tp, tn, fp, fn)

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / float(tp + tn + fp + fn)
    print(tp, tn, fp, fn)
    print(tpr)
    print(fpr)
    print(acc)
    return tpr, fpr, acc

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
    #model_root = "./model/ms1m-ir50/backbone_ir50_ms1m_epoch120.pth"
    get_image_pairs()
