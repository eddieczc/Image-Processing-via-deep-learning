'''TO create the path file'''
TXT = './train.txt'
f1 = open(TXT,'a')#open file
for i in range(1,33403):
    f1.write('/content/drive/MyDrive/yolov4/train/{}.jpg'.format(i))
    f1.write("\n")
f1.close()#close file

TXT = 'C:/Users/naip/Desktop/IOC_HW2/dataset/true/test.txt'
f1 = open(TXT,'a')#open file
for i in range(1,13069):
    f1.write('/home/naip/IOC/yolo_tiny/darknet/data/IOC/test/{}.png'.format(i))
    f1.write("\n")
f1.close()#close file
