'''TO create the json file'''
import json
from PIL import Image

with open("result.json","r",encoding="utf-8") as f:
    data = json.load(f)
final = []
for i in range(0,13068):
    FILE_PATH = 'C:/Users/naip/Desktop/IOC_HW2/dataset/test/{}.png'.format(i+1)
    img = Image.open(FILE_PATH)
    imgSize = img.size
    w = img.width
    h = img.height

    value = {}
    score = []
    label = []
    pos_first = []
    for pred in data[i]['objects']:
        pos = []
        x_center = pred['relative_coordinates']['center_x']
        y_center = pred['relative_coordinates']['center_y']
        pr_w = pred['relative_coordinates']['width']
        pr_h = pred['relative_coordinates']['height']
        x1 = w * x_center - (w * pr_w) / 2
        x2 = w * x_center + (w * pr_w) / 2
        y1 = h * y_center - (h * pr_h) / 2
        y2 = h * y_center + (h * pr_h) / 2
        pos.append((y1))
        pos.append((x1))
        pos.append((y2))
        pos.append((x2))
        pos_first.append(pos)
        score.append((pred['confidence']))

        if pred['name'] == 0:
            pred['name'] = 10
            label.append(int(pred['name']))
        else:
            label.append(int(pred['name']))
    value = {"bbox":pos_first,"score":score,"label":label}
    final.append(value)
with open('0851912_6.json', 'w', encoding='utf-8') as f:
    json.dump(final, f)
    