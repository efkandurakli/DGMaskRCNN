import os
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util


seg_images_path = "repro_10k_segmentations/VOC2012/SegmentationObject"
xml_files_path = "repro_10k_annotations/VOC2012/Annotations"

seg_image_files = os.listdir(seg_images_path)

ann_id = 1
images = [  ]
for i, seg_image_file in enumerate(seg_image_files):
    print(i)
    seg_img = cv2.imread(os.path.join(seg_images_path, seg_image_file))
    height, width, _ = seg_img.shape
    pixel_values = seg_img.reshape(-1, 3)

    black_color = np.array([0, 0, 0])

    non_black_mask = np.any(pixel_values != black_color, axis=1)

    unique_non_black_colors = np.unique(pixel_values[non_black_mask], axis=0)
    annotations = []
    image_id = i+1
    images.append({"id": image_id, "height": height, "width": width, "file_name": seg_image_file})
    for color in unique_non_black_colors:
        mask = np.all(seg_img == color, axis=2)
        bimask = np.zeros((seg_img.shape[0], seg_img.shape[1]))
        bimask[mask] = 1
        segmentation = mask_util.encode(np.asfortranarray(bimask,dtype=np.uint8))
        segmentation['counts'] = str(segmentation['counts'], encoding='utf-8')
        area = float(mask_util.area(segmentation))
        bbox = mask_util.toBbox(segmentation)
        bbox = [int(b) for b in bbox]
        category_id = 1
        iscrowd = 0
        ann_id += 1

        annotations.append({"id": ann_id, "image_id": image_id, "category_id": category_id, "segmentation": segmentation, "area": area, "bbox": bbox, "iscrowd": iscrowd})


categories = [{"id": 1, "name": "car", "supercategory": "car"}]

dataset = {"images": images, "annotations": annotations, "categories": categories}

with open("dataset.json", "w") as json_file:
    json.dump(dataset, json_file)