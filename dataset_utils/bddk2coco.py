import json
import pycocotools.mask as mask_util


def get_categories(frames):
    categories = set()
    for frame in frames:
        labels = frame["labels"]
        for label in labels:
            categories.add(label["category"])
    categories = list(categories)
    categories.sort()
    return categories

def bddk2coco(input_path, out_path):

    json_file = open(input_path)

    data = json.load(json_file)

    json_file.close()

    frames = data["frames"]

    images = []
    annotations = []
    categories = get_categories(frames)
    ann_id = 1
    for i, frame in enumerate(frames):
        images.append({"file_name": frame["name"], "height": 720, "width": 1280, "id": i+1})
        labels = frame["labels"]
        for label in labels:
            id = ann_id
            image_id = i+1
            ann_id += 1

            category_id = categories.index(label["category"])+1
            x1 = int(label["box2d"]["x1"])
            y1 = int(label["box2d"]["y1"])
            x2 = int(label["box2d"]["x2"])
            y2 = int(label["box2d"]["y2"])

            bbox = [x1, y1, x2-x1, y2-y1]
            segmentation = {}
            segmentation["size"] = label["rle"]["size"]
            segmentation["counts"] = label["rle"]["counts"]
            area = float(mask_util.area(segmentation))
            iscrowd = 0

            annotations.append({"id": id, "image_id": image_id, "category_id": category_id, "segmentation": segmentation, "area": area, "bbox": bbox, "iscrowd": iscrowd})


    categories = [{"id": i+1, "name": category, "supercategory": category} for i, category in enumerate(categories)]
    dataset = {"images": images, "annotations": annotations, "categories": categories}

    with open(out_path, "w") as json_file:
        json.dump(dataset, json_file)

def get_args_parser(add_help=True):
    import argparse
    
    parser = argparse.ArgumentParser(description="Converting BDDK dataset to COCO dataset", add_help=add_help)

    parser.add_argument("--input-path", default=None, type=str, required=True, help="input json file path")
    parser.add_argument("--out-path", default=None, type=str, required=True, help="output json file path")

    return parser

def main(args):
    input_path = args.input_path
    out_path = args.out_path
    bddk2coco(input_path, out_path)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)