import os
import cv2

labels = open("train/label.txt", "r")
label = labels.read().split(',')
print(len(label))

# path = 'test/test/'
path = 'train/'
for cnt in range(len(label)):
    image_path = (path + str(cnt) + '.png')
    img = cv2.imread(image_path)
    # 根据图片对应的标签分类到对应的文件夹下：
    cv2.imwrite('emnist/Train_png/' + chr(int(label[cnt])+64) + '/' + str(cnt) + '.png', img)
    cnt += 1
