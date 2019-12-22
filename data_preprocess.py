from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import os
from ast import literal_eval


""" turn list of strokes to pillow image file """
def stroke2Pimage(stroke_format, out_width, out_height):
    img = Image.new('P', (256, 256), color=65535)
    img_draw = ImageDraw.Draw(img)

    for stroke in stroke_format:
        for i in range(len(stroke)):
            x = stroke[0]
            y = stroke[1]

            for j in range(len(x)-1):
                x1 = x[j]
                y1 = y[j]
                x2 = x[j+1]
                y2 = y[j+1]
                img_draw.line([x1, y1, x2, y2], fill=0, width=5)

    img = img.resize((out_height, out_width))

    img = np.array(img, dtype='uint16')
    img = Image.fromarray(img, 'I;16')
    img = img.convert('RGB')
    
    return img


""" turn doodle-stroke csv file to png into directory as same class """
def csv2pngs(csv_path, csv_name, image_path, width, height):
    df = pd.read_csv(csv_path + csv_name + '.csv', sep=',', header=None)
    df = df.iloc[1:]    # drop first row which is not data
    df = df.drop(df.columns[0], axis=1) # drop first column which is redundant

    image_path += (csv_name + '/')

    try:
        os.mkdir(image_path)
    except OSError:
        print('directory is already existed')
    else:
        pass

    for row in df.itertuples():
        index = row[0] - 1  # start from 0
        stroke_format = row[1]
        stroke_format = literal_eval(stroke_format)

        img_path = image_path + (csv_name + str(index) + '.png')
        img = stroke2Pimage(stroke_format, width, height)
        img.save(img_path)


#csv2pngs('dataset/train_doodle_csv/', 'banana', 'dataset/train_png/', 150, 150)

csv2pngs('dataset/train_doodle_csv/', 'test_predict', 'dataset/train_png/', 150, 150)