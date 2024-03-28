import os
import cv2

labels = open("test/label.txt", "r")
label = labels.read().split(',')
print(len(label))

# path = 'test/test/'
path = 'test/'
for cnt in range(len(label)):
    image_path = (path + str(cnt) + '.png')
    img = cv2.imread(image_path)
    # 根据图片对应的标签分类到对应的文件夹下：
    cv2.imwrite('emnist/Test_png/' + label[cnt] + '/' + str(cnt) + '.png', img)
    cnt += 1
