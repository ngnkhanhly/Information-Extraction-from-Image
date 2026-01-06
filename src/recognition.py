import os
import cv2
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from PIL import Image
import timm


# ===============================
# XML PARSING FOR RECOGNITION
# ===============================

def extract_data_from_xml_for_recognition(root_dir):
    xml_path = os.path.join(root_dir, "words.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []

    for img in root:
        bbs_of_img = []
        labels_of_img = []

        for bbs in img.findall("taggedRectangles"):
            for bb in bbs:
                # check non-alphabet and non-number
                if not bb[0].text.isalnum():
                    continue

                if "é" in bb[0].text.lower() or "ñ" in bb[0].text.lower():
                    continue

                bbs_of_img.append(
                    [
                        float(bb.attrib["x"]),
                        float(bb.attrib["y"]),
                        float(bb.attrib["width"]),
                        float(bb.attrib["height"]),
                    ]
                )
                labels_of_img.append(bb[0].text.lower())

        img_path = os.path.join(root_dir, img[0].text)
        img_paths.append(img_path)
        img_sizes.append((int(img[1].attrib["x"]), int(img[1].attrib["y"])))
        bboxes.append(bbs_of_img)
        img_labels.append(labels_of_img)

    return img_paths, img_sizes, img_labels, bboxes

# ===============================
# CROP TEXT REGIONS
# ===============================

def crop_text(img_paths, img_labels, bboxes, save_dir="cropped_text"):
    os.makedirs(save_dir, exist_ok=True)

    cropped_paths = []
    labels = []

    for img_path, img_label, bbox in zip(img_paths, img_labels, bboxes):
        img = cv2.imread(img_path)

        # Skip if image could not be loaded
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        for i, (bb, label) in enumerate(zip(bbox, img_label)):
            x, y, w, h = bb
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Ensure bounding box is within image bounds
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            x_end = min(img.shape[1], x + w)
            y_end = min(img.shape[0], y + h)

            # Skip if bounding box is invalid
            if x_end <= x or y_end <= y:
                print(f"Warning: Invalid bounding box for {img_path}")
                continue

            cropped_img = img[y:y_end, x:x_end]

            # Skip if crop is empty
            if cropped_img.size == 0:
                print(f"Warning: Empty crop for {img_path} at bbox {bb}")
                continue

            img_name = os.path.basename(img_path).replace(".JPG", "")
            save_path = os.path.join(save_dir, f"{img_name}_{i}.jpg")

            cv2.imwrite(save_path, cropped_img)

            cropped_paths.append(save_path)
            labels.append(label)

    return cropped_paths, labels

# ===============================
# LABEL ENCODE / DECODE
# ===============================

def encode(text, char_to_idx, max_label_len):
    encoded = []
    for char in text:
        encoded.append(char_to_idx[char])

    label_len = len(encoded)

    # Pad with 0s
    encoded += [0] * (max_label_len - len(encoded))

    return torch.LongTensor(encoded), torch.tensor(label_len, dtype=torch.long)


def decode(encoded_sequences, idx_to_char, blank_char="-"):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None

        for token in seq:
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char

        decoded_sequences.append("".join(decoded_label))

    return decoded_sequences


def decode_label(encoded_sequences, idx_to_char, blank_char="-"):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        for idx, token in enumerate(seq):
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    decoded_label.append(char)

        decoded_sequences.append("".join(decoded_label))

    return decoded_sequences


# ===============================
# DATASET
# ===============================

class STRDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        char_to_idx,
        max_label_len,
        label_encoder=None,
        transform=None,
    ):
        self.transform = transform
        self.img_paths = X
        self.labels = y
        self.char_to_idx = char_to_idx
        self.max_label_len = max_label_len
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.label_encoder:
            encoded_label, label_len = self.label_encoder(
                label, self.char_to_idx, self.max_label_len
            )
        return img, encoded_label, label_len


# ===============================
# CRNN MODEL
# ===============================

class CRNN(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        # Unfreeze the last few layers
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        # ResNet34 outputs 512 channels
        self.mapSeq = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the feature map
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  # Based on CTC

        return x
