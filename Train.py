import os
import torch
import torchvision
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from engine import train_one_epoch, evaluate
import utils


# -----------------------
# Custom Dataset
# -----------------------
class CocoInstanceDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = torchvision.io.read_image(img_path).float() / 255.0

        # Ensure 3 channels
        if img.shape[0] == 4:
            img = img[:3, :, :]
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            masks.append(coco.annToMask(ann))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


# -----------------------
# Transforms
# -----------------------
class ComposeWithTarget:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img = t(img)
        return img, target


def get_transform(train):
    transforms = []
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return ComposeWithTarget(transforms)


# -----------------------
# Model with both weight files
# -----------------------
def get_model(num_classes, backbone_weights=None, maskrcnn_weights=None):
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)

    # Load ResNet50 backbone weights
    if backbone_weights and os.path.exists(backbone_weights):
        state_dict = torch.load(backbone_weights, map_location="cpu")
        missing, unexpected = model.backbone.body.load_state_dict(state_dict, strict=False)
        print("Loaded ResNet50 backbone weights from local file.")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    # Load COCO Mask R-CNN weights (excluding classifier/mask heads)
    if maskrcnn_weights and os.path.exists(maskrcnn_weights):
        state_dict = torch.load(maskrcnn_weights, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items() if "roi_heads.box_predictor" not in k and "roi_heads.mask_predictor" not in k}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Loaded COCO Mask R-CNN weights (excluding box/mask heads).")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    return model


# -----------------------
# Main training function
# -----------------------
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    # Dataset paths
    train_img = r"C:\Users\Nm_User0\Desktop\Computer Vision\new_project\Blower_dataset\train\Images"
    ann_train = r"C:\Users\Nm_User0\Desktop\Computer Vision\new_project\Blower_dataset\train\annotations\labels_my-project-name_2025-10-14-11-41-01.json"

    val_img = r"C:\Users\Nm_User0\Desktop\Computer Vision\new_project\Blower_dataset\val\Images"
    ann_val = r"C:\Users\Nm_User0\Desktop\Computer Vision\new_project\Blower_dataset\val\annotations\labels_my-project-name_2025-10-08-10-41-00.json"

    # Create datasets
    dataset = CocoInstanceDataset(train_img, ann_train, get_transform(train=True))
    dataset_test = CocoInstanceDataset(val_img, ann_val, get_transform(train=False))

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    # Number of classes (background + actual categories)
    num_classes = 1 + len(dataset.coco.loadCats(dataset.coco.getCatIds()))
    print("num_classes:", num_classes)

    # Paths to local pretrained weights
    backbone_weights = r"C:\Users\Nm_User0\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth"
    maskrcnn_weights = r"C:\Users\Nm_User0\.cache\torch\hub\checkpoints\maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"

    # Build model
    #model = get_model(num_classes=num_classes, backbone_weights=backbone_weights, maskrcnn_weights=maskrcnn_weights)
    #model.to(device)

    
    model = get_model(num_classes=num_classes)  # Don't load COCO weights again
    model.load_state_dict(torch.load("blower_maskrcnn_custom.pth"))
    model.to(device)


    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loops
    num_epochs = 22
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Safe evaluation
        try:
            evaluate(model, data_loader_test, device=device)
        except Exception as e:
            print(f"Skipping evaluation this epoch due to error: {e}")

    # Save final model
    torch.save(model.state_dict(), "blower_fintuned_maskrcnn_custom.pth")
    print("Training finished, model saved as maskrcnn_custom.pth")


if __name__ == "__main__":
    main()
