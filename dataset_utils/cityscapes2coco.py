import os
import json
import pycocotools.mask as mask_util


classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

def cityscapes2coco(root_dir, out_dir):
    for image_set in ["train", "val"]:
        gtPath = os.path.join(root_dir, image_set)
        images = []
        annotations = []
        image_id = 1
        ann_id = 1
        for foldername, subfolders, filenames in os.walk(gtPath):
            for filename in filenames:
                if filename.endswith("gtFine_polygons.json"):

                    splits = filename.split("_")
                    file_path = os.path.join(foldername, filename)
                    image_filename = "_".join([splits[0], splits[1], splits[2], "leftImg8bit.png"])
                    image_filename = os.path.join(foldername.split("/")[-1], image_filename)
                
                    file = open(file_path)
                    data = json.load(file)
                    file.close()

                    height = data["imgHeight"]
                    width = data["imgWidth"]

                    images.append({"id": image_id, "width": width, "height": height, "file_name": image_filename})
                    image_id += 1

                    objects = data["objects"]
                    for obj in objects:
                        if obj["label"] in classes:
                            category = obj["label"]
                            polygon = obj["polygon"]
                            coco_polygon = [[coord for point in polygon for coord in point]]
                            rles = mask_util.frPyObjects(coco_polygon, height, width)
                            rle = mask_util.merge(rles)
                            rle['counts'] = str(rle['counts'], encoding='utf-8')
                            bbox = mask_util.toBbox(rle).tolist()
                            bbox = [int(b) for b in bbox]
                            area = float(mask_util.area(rle))
                            annotation = {"id": ann_id, "image_id": image_id, "category_id": classes.index(category)+1, "area": area, "bbox": bbox, "segmentation": rle, "iscrowd": 0}
                            annotations.append(annotation)
                            ann_id += 1

        categories = [{"id": i+1, "name": cls, "supercategory": cls} for i, cls in enumerate(classes)]
        dataset = {"images": images, "annotations": annotations, "categories": categories}
        out_file_path = os.path.join(out_dir, image_set + ".json")

        with open(out_file_path, "w") as json_file:
            json.dump(dataset, json_file)

def get_args_parser(add_help=True):
    import argparse
    
    parser = argparse.ArgumentParser(description="Converting Cityscape dataset to COCO dataset", add_help=add_help)

    parser.add_argument("--root", type=str, required=True, help="The root directory of the ground truth files")
    parser.add_argument("--out-dir", type=str, required=True, help="output json file path")

    return parser

def main(args):
    root_dir = args.root
    out_dir = args.out_dir
    cityscapes2coco(root_dir, out_dir)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)