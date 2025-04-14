import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from pathlib import PurePath
import json
import numpy as np

# Master Function that calls all the functions

# TODO make a function that downloads the patches upon request and then performs this preprocessing
def create_coco_json(df, img_dir) -> None:
    contents = {'categories' : [], 'images' : [], 'annotations' : []}
    for i in tqdm(range(len(df))):
        # TODO implement the selection of categories that are relevant to the fine tuning
        row = df.iloc[i, :]
        contents['images'].append(extract_img(row, img_dir))
        contents['annotations'] =  contents['annotations'] + extract_ann(row)
        for obj in row['anns_info']:
            cat_obj = extract_cat(obj)
            if cat_obj not in contents['categories']:
                contents['categories'].append(cat_obj)

    with open('_annotations.coco.json', 'w') as f:
        json.dump(contents, f)

"""
Function that extracts the image info as required in the COCO JSON format and returns it

Parameters:
row (pandas.core.series.Series): The row object that represents all the information associated with the image
img_dir (str|pathlib.Path|pathlib.PurePath): The directory of the image folder contained the patches of the dataset

Returns:
dict: contains the formated dict which can be appended to the 'images' list in the JSON file
"""
def extract_img(row, img_dir) -> dict:
    # TODO check if there is a problem in including licenses when licenses field is empty
    img_object = row['image_info']
    img_dummy_path = PurePath(img_object['file_name'])
    img_actual_path = PurePath(img_dir, img_dummy_path.name)
    img_object['file_name'] = str(img_actual_path)
    return img_object


def extract_cat(obj) -> dict:
    cat_object = {}
    cat_object['name'] = obj['category']
    cat_object['id'] = obj['category_id']
    # TODO Check if None or empty string is better for blank keys
    cat_object['supercategory'] = ''
    return cat_object


def extract_ann(row) -> list[dict]:
    # Converting bounding boxes from xyxy to xywh format
    anns_obj = row.anns_info
    for j in range(len(anns_obj)):
        anns_obj[j]['bbox'] = anns_obj[j]['bbox'].tolist()
        anns_obj[j]['bbox'][2] -= anns_obj[j]['bbox'][0]
        anns_obj[j]['bbox'][3] -= anns_obj[j]['bbox'][1]
    return anns_obj.tolist()

if __name__ == '__main__':
    dataset = load_dataset('jxu124/objects365', split = 'train')
    df = dataset.to_pandas()
    create_coco_json(df.head(1000), '/Users/yash/ML Projects/TempFolder/annotations.json')