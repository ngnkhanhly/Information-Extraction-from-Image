import os
import shutil
import xml.etree.ElementTree as ET


# Extract Data from XML for Detection
def extract_data_from_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    image_paths = []
    image_sizes = []
    image_labels = []
    bounding_boxes = []

    for image in root:
        bbs_of_image = []
        labels_of_image = []

        for bbs in image.findall("taggedRectangles"):
            for bb in bbs:
                # check non-alphabet and non-number
                if not bb[0].text.isalnum():
                    continue

                if "é" in bb[0].text.lower() or "ñ" in bb[0].text.lower():
                    continue

                bbs_of_image.append(
                    [
                        float(bb.attrib["x"]),
                        float(bb.attrib["y"]),
                        float(bb.attrib["width"]),
                        float(bb.attrib["height"]),
                    ]
                )
                labels_of_image.append(bb[0].text.lower())

        image_paths.append(image[0].text)
        image_sizes.append((int(image[1].attrib["x"]), int(image[1].attrib["y"])))
        bounding_boxes.append(bbs_of_image)
        image_labels.append(labels_of_image)

    return image_paths, image_sizes, image_labels, bounding_boxes


# Convert to YOLO Format
def convert_to_yolo_format(image_paths, image_sizes, bounding_boxes):
    yolo_data = []

    for img_path, img_size, bbs in zip(image_paths, image_sizes, bounding_boxes):
        img_w, img_h = img_size
        yolo_bbs = []

        for bb in bbs:
            x, y, w, h = bb
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            yolo_bbs.append([0, x_center, y_center, w_norm, h_norm])

        yolo_data.append((img_path, yolo_bbs))

    return yolo_data


# Save YOLO Data
def save_yolo_data(data, split, save_dir, dataset_dir):
    split_dir = os.path.join(save_dir, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for img_path, yolo_bbs in data:
        img_name = img_path.replace("/", "_")

        img_src = os.path.join(dataset_dir, img_path)
        img_dest = os.path.join(images_dir, img_name)
        shutil.copy(img_src, img_dest)

        label_name = (
            img_name.replace(".JPG", ".txt")
                    .replace(".jpg", ".txt")
        )
        label_dest = os.path.join(labels_dir, label_name)

        with open(label_dest, "w") as f:
            for cls_id, x, y, w, h in yolo_bbs:
                f.write(f"{cls_id} {x} {y} {w} {h}\n")
