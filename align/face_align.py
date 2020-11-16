from PIL import Image, ImageTk
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
import argparse
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
def detect_face_align(img):
    crop_size = 112  #因为backbone的input_size要[112,112]or[224,224]
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    #img = Image.open(os.path.join(image_path))

    bounding_boxes = []
    warped_face = []
    img_warped = []
    facial5point = []
    landmarks = []
    try:
        bounding_boxes, landmarks = detect_faces(img)
    except Exception as e:
        print(e)
    if len(landmarks) == 0:  # If the landmarks cannot be detected, the img will be discarded
        return img_warped, bounding_boxes
    #print(landmarks)
    for i in range(len(landmarks)):
        facial5point.append([[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)])
        #print(facial5points[i])
        warped_face.append(warp_and_crop_face(np.array(img), facial5point[i], reference, crop_size=(crop_size, crop_size)))

        img_warped.append(Image.fromarray(warped_face[i]))
        # img_warped.save("test.jpg")
        # img_warped.show()
    return img_warped, bounding_boxes
    #return warped_face

def detect_one_face_align(img):
    crop_size = 112
    scale = crop_size / 112.

    reference = get_reference_facial_points(default_square=True) * scale

    #img = Image.open(os.path.join(image_path))

    landmarks = []
    bounding_boxes = []
    try:
        bounding_boxes, landmarks = detect_faces(img)

    except Exception as e:
        print(e)
    if len(landmarks) == 0:  # If the landmarks cannot be detected, the img will be discarded
        return None
    #print(landmarks)


    facial5point=[[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    #print(facial5points[i])
    warped_face=warp_and_crop_face(np.array(img), facial5point, reference, crop_size=(crop_size, crop_size))
    img_warped=Image.fromarray(warped_face)

    # img_warped.save("test.jpg")
    # img_warped.show()
    return img_warped, bounding_boxes

if __name__ == '__main__':
    # image = Image.open("../data/ycy.jpg")
    # image2 = Image.open("../data/ljy.jpg")
    # print(image2)
    # img = []
    # # for i in range(len(landmarks)):
    # #     img.append(detect_face_align(image2))
    # #
    # #     Image._show(img[-1])
    # img, bounding_boxes = detect_face_align(image2)
    # for i in range(len(bounding_boxes)):
    #     print(img[i])
    #     img[i].show()
    #     img[i].close()
    #     #Image._show(img[i])
    #     print(bounding_boxes[i])

    parser = argparse.ArgumentParser(description = "face alignment")
    parser.add_argument("-source_root", "--source_root", help = "specify your source dir", default = "../data/train/", type = str)
    parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default = "../data/train_aligned2/", type = str)
    parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default = 112, type = int)
    args = parser.parse_args()

    source_root = args.source_root # specify your source dir
    dest_root = args.dest_root # specify your destination dir
    crop_size = args.crop_size # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale

    cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)

    for subfolder in tqdm(os.listdir(source_root)):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
            img = Image.open(os.path.join(source_root, subfolder, image_name))
            try: # Handle exception
                _, landmarks = detect_faces(img)
            except Exception:
                print("{} is discarded due to exception!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
                print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
                image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
            img_warped.save(os.path.join(dest_root, subfolder, image_name))
