#! /usr/bin/env python

import argparse
from dict2xml import dict2xml
import json
import os


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def read_coco_annotations(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    images = {detail["id"]: detail["filename"] for detail in data["images"]}
    for detail in data["annotations"]:
        detail["image"] = images[detail["image_id"]]
    return data["annotations"]


def read_via_annotations(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data["_via_img_metadata"]


def generate_coco_from_via(via_annotation, prefix=""):
    for _, image_data in via_annotation.items():
        root, filename = os.path.split(image_data["filename"])
        _, folder = os.path.split(root)

        coco = {"filename": filename}
        coco["folder"] = folder
        coco["path"] = os.path.join(prefix, filename)
        coco["source"] = {"database": "Unknown"}
        coco['size'] = {"width": "1920", "height":"1080", "depth":"3"}
        coco["segmented"] = 0
        coco["object"] = []
        for region in image_data["regions"]:
            details = {}
            details["name"] = region["region_attributes"]["type"]
            details["pose"] = "Unspecified"
            details["difficult"] = 0
            details["truncated"] = len(
                region["shape_attributes"]["all_points_x"]) > 4
            details["bndbox"] = {
                "xmin": min(region["shape_attributes"]["all_points_x"]),
                "xmax": max(region["shape_attributes"]["all_points_x"]),
                "ymin": min(region["shape_attributes"]["all_points_y"]),
                "ymax": max(region["shape_attributes"]["all_points_y"]),
            }
            coco["object"].append(details)
        yield dict2xml(coco, wrap="annotation", indent=" "),filename

def via2coco(via_annotation, prefix=""):
    return [xml for xml in generate_coco_from_via(via_annotation, prefix)]


def parse_args():
    parser = argparse.ArgumentParser(
        "Convert a json via project into coco annotations")
    parser.add_argument(
        "--debug",
        "-d",
        help="Show intermediate output",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--input", "-i", help="Path to input file", required=True, type=str
    )
    parser.add_argument(
        "--prefix",
        "-p",
        help="Prefix for creating absolute paths instead of considering the **file-name** as relative",
        default=os.path.abspath(os.curdir),
        type=str,
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    for xml, fn in generate_coco_from_via(read_via_annotations(args.input), args.prefix):
        print(xml)
        f = open(fn[:-4]+".xml", "a")
        f.write(xml)
        f.close()
        

if __name__ == "__main__":
    main()
