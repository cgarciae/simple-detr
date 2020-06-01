from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import typing as tp
from pathlib import Path

CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


class DETR(torch.nn.Module):
    def __init__(
        self,
        threshold: float = 0.7,
        backbone: str = "detr_resnet50",
        device: str = "cpu",
        pretrained: bool = True,
        **kwargs,
    ):
        """
        Arguments:
            threshold: (float = 0.7) probability threshold required to keep a box.
            backbone: (str = detr_resnet50) one of:
                * detr_resnet50
                * detr_resnet50_dc5
                * detr_resnet101
                * detr_resnet101_dc5
            device: (str = cpu) device to use e.g. "cpu", "cuda", etc
            pretrained: (bool = True) 
            **kwargs: backbone options, for more information see https://github.com/facebookresearch/detr/blob/master/hubconf.py
        """
        super().__init__()

        self.device = device
        self.threshold = threshold
        self.model = torch.hub.load(
            "facebookresearch/detr", backbone, pretrained=pretrained, **kwargs
        )
        self.transform = T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                no_alpha,
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.to(self.device)

    def forward(self, image: torch.Tensor) -> tp.Dict[str, torch.Tensor]:
        return self.model(image)

    def predict(self, image: Image) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the scores and bounding boxes for an image.

        Arguments:
            image: (PIL.Image) image sample.

        Returns:
            (scores, boxes): (Tuple[Tensor, Tensor]) tensors of the scores and boxes.
        """
        # mean-std normalize the input image (batch-size: 1)
        image_t = self.transform(image).unsqueeze(0).to(self.device)

        # propagate through the model
        outputs = self.model(image_t)

        # keep only predictions with 0.7+ confidence
        boxes = outputs["pred_boxes"].to("cpu")
        logits = outputs["pred_logits"].to("cpu")

        probas = logits.softmax(-1)  # get probabilities
        probas = probas[0, :, :-1]  # exclude empty class
        keep = probas.max(-1).values > self.threshold

        scores = probas[keep]
        boxes = boxes[0, keep]

        # convert boxes from [0; 1] to image scales
        boxes = self.rescale_bboxes(boxes, image.size)

        return scores, boxes

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x: torch.Tensor) -> torch.Tensor:
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(
        self, out_bbox: torch.Tensor, size: tp.Tuple[int, ...]
    ) -> torch.Tensor:
        img_w, img_h = size

        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

        return b


def no_alpha(x):
    return x[:3]


def draw_boxes(image: Image, prob: torch.Tensor, boxes: torch.Tensor):
    # plt.figure(figsize=(16,10))
    plt.imshow(image)

    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        cl = p.argmax()
        text = f"{CLASSES[cl]}: {p[cl]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))

    plt.axis("off")
    plt.show()
