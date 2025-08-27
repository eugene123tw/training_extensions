import argparse
import torch
import torch.nn.functional as F
from typing import NamedTuple
from pathlib import Path
from otx.algo.detection.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from otx.algo.detection.dinov3.data.transforms import make_classification_eval_transform
from datumaro import Dataset as DmDataset
from otx.types.transformer_libs import TransformLibType
from otx.config.data import SubsetConfig
from otx.data.factory import OTXDatasetFactory
from otx.types.task import OTXTaskType
from otx.types.image import ImageColorChannel


DatasetInfo = NamedTuple("DatasetInfo", [("name", str), ("path", Path), ("group", str)])

def accuracy(output, target, topk=(1,)):
    """Compute accuracy for top-k predictions."""
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).sum(0, keepdim=True) for k in topk]


def zeroshot_classifier(classnames, templates, tokenizer, model):
    """Create zero-shot classifier weights."""
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def dataset_collections():
    return [
        # DatasetInfo(
        #     name="multiclass_tiny_pneumonia",
        #     path=Path("multiclass_classification/mcls_tiny_pneumonia_12_6_200"),
        #     group="tiny",
        # ),
        # DatasetInfo(
        #     name="multiclass_tiny_cub_woodpecker",
        #     path=Path("multiclass_classification/mcls_tiny_cub_woodpecker_24_12_200"),
        #     group="tiny",
        # ),
        # DatasetInfo(
        #     name="multiclass_small_flowers",
        #     path=Path("multiclass_classification/mcls_small_flowers_60_12_200"),
        #     group="small",
        # ),
        DatasetInfo(
            name="multiclass_small_eurosat",
            path=Path("multiclass_classification/mcls_small_eurosat_80_40_200"),
            group="small",
        ),
        DatasetInfo(
            name="multiclass_medium_resisc",
            path=Path("multiclass_classification/mcls_medium_resisc_500_100_400"),
            group="medium",
        ),
        # DatasetInfo(
        #     name="multiclass_large_cub100",
        #     path=Path("multiclass_classification/mcls_large_cub100_3764_900_1200"),
        #     group="large",
        # ),
]


def main(dataset_root: Path, output_dir: Path, result_name: str):

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]


    print("Loading DinoTXT model...")
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        backbone_weights="/home/yuchunli/git/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        dinotxt_weights="/home/yuchunli/git/dinov3/weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    )
    model = model.cuda()
    
    dataset_infos = dataset_collections()

    # OpenAI ImageNet templates
    openai_imagenet_templates = (
        lambda c: f"a bad photo of a {c}.",
        lambda c: f"a photo of many {c}.",
        lambda c: f"a sculpture of a {c}.",
        lambda c: f"a photo of the hard to see {c}.",
        lambda c: f"a low resolution photo of the {c}.",
        lambda c: f"a rendering of a {c}.",
        lambda c: f"graffiti of a {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a cropped photo of the {c}.",
        lambda c: f"a tattoo of a {c}.",
        lambda c: f"the embroidered {c}.",
        lambda c: f"a photo of a hard to see {c}.",
        lambda c: f"a bright photo of a {c}.",
        lambda c: f"a photo of a clean {c}.",
        lambda c: f"a photo of a dirty {c}.",
        lambda c: f"a dark photo of the {c}.",
        lambda c: f"a drawing of a {c}.",
        lambda c: f"a photo of my {c}.",
        lambda c: f"the plastic {c}.",
        lambda c: f"a photo of the cool {c}.",
        lambda c: f"a close-up photo of a {c}.",
        lambda c: f"a black and white photo of the {c}.",
        lambda c: f"a painting of the {c}.",
        lambda c: f"a painting of a {c}.",
        lambda c: f"a pixelated photo of the {c}.",
        lambda c: f"a sculpture of the {c}.",
        lambda c: f"a bright photo of the {c}.",
        lambda c: f"a cropped photo of a {c}.",
        lambda c: f"a plastic {c}.",
        lambda c: f"a photo of the dirty {c}.",
        lambda c: f"a jpeg corrupted photo of a {c}.",
        lambda c: f"a blurry photo of the {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a good photo of the {c}.",
        lambda c: f"a rendering of the {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"a photo of one {c}.",
        lambda c: f"a doodle of a {c}.",
        lambda c: f"a close-up photo of the {c}.",
        lambda c: f"a photo of a {c}.",
        lambda c: f"the origami {c}.",
        lambda c: f"the {c} in a video game.",
        lambda c: f"a sketch of a {c}.",
        lambda c: f"a doodle of the {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"a low resolution photo of a {c}.",
        lambda c: f"the toy {c}.",
        lambda c: f"a rendition of the {c}.",
        lambda c: f"a photo of the clean {c}.",
        lambda c: f"a photo of a large {c}.",
        lambda c: f"a rendition of a {c}.",
        lambda c: f"a photo of a nice {c}.",
        lambda c: f"a photo of a weird {c}.",
        lambda c: f"a blurry photo of a {c}.",
        lambda c: f"a cartoon {c}.",
        lambda c: f"art of a {c}.",
        lambda c: f"a sketch of the {c}.",
        lambda c: f"a embroidered {c}.",
        lambda c: f"a pixelated photo of a {c}.",
        lambda c: f"itap of the {c}.",
        lambda c: f"a jpeg corrupted photo of the {c}.",
        lambda c: f"a good photo of a {c}.",
        lambda c: f"a plushie {c}.",
        lambda c: f"a photo of the nice {c}.",
        lambda c: f"a photo of the small {c}.",
        lambda c: f"a photo of the weird {c}.",
        lambda c: f"the cartoon {c}.",
        lambda c: f"art of the {c}.",
        lambda c: f"a drawing of the {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"a black and white photo of a {c}.",
        lambda c: f"the plushie {c}.",
        lambda c: f"a dark photo of a {c}.",
        lambda c: f"itap of a {c}.",
        lambda c: f"graffiti of the {c}.",
        lambda c: f"a toy {c}.",
        lambda c: f"itap of my {c}.",
        lambda c: f"a photo of a cool {c}.",
        lambda c: f"a photo of a small {c}.",
        lambda c: f"a tattoo of the {c}.",
    )
    
    for dataset_info in dataset_infos:
        # Configuration
        dataset_path = dataset_root / dataset_info.path
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {device}")
        print(f"Dataset: {dataset_path}")    
        
        # 2. Load dataset
        print("Loading dataset...")
        dm_dataset = DmDataset.import_from(dataset_path, format="imagenet_with_subset_dirs")
        test_subset = dm_dataset.get_subset("test")
        
        # Create OTX dataset with minimal transforms
        test_config = SubsetConfig(
            batch_size=16,
            subset_name="test",
            transform_lib_type=TransformLibType.TORCHVISION,
            num_workers=0,
            to_tv_image=True,
            transforms=[
                {
                    "class_path": "torchvision.transforms.v2.Resize",
                    "init_args": {"size": [512, 512], "antialias": True}
                },
                {
                    "class_path": "torchvision.transforms.v2.CenterCrop",
                    "init_args": {"size": 512}
                },
                {
                    "class_path": "torchvision.transforms.v2.ToDtype",
                    "init_args": {"dtype": torch.float32, "scale": False}
                },
                {
                    "class_path": "torchvision.transforms.v2.Normalize",
                    "init_args": {
                        "mean": mean,
                        "std": std
                    }
                }
            ]
        )
        
        otx_dataset = OTXDatasetFactory.create(
            task=OTXTaskType.MULTI_CLASS_CLS,
            dm_subset=test_subset.as_dataset(),
            cfg_subset=test_config,
            data_format="imagenet_with_subset_dirs",
            image_color_channel=ImageColorChannel.RGB,
        )
        
        print(f"Dataset size: {len(otx_dataset)}")
        print(f"Classes: {otx_dataset.label_info.label_names}")
        
        # 3. Create text prompts for zero-shot classification
        class_names = otx_dataset.label_info.label_names
    
        print("Creating zero-shot classifier...")
        zeroshot_weights = zeroshot_classifier(class_names, openai_imagenet_templates, tokenizer, model)

        model.eval()
        top1, top5, n = 0., 0., 0.
        for i in range(len(otx_dataset)):
            if i % 20 == 0:
                print(f"Processing {i}/{len(otx_dataset)}")
            
            data_item = otx_dataset[i]
            image = data_item.image.unsqueeze(0).to(device)
            target = data_item.label.to(device)

            with torch.autocast('cuda', dtype=torch.float):
                with torch.no_grad():
                    image_features = model.encode_image(image.cuda())
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = 100. * image_features @ zeroshot_weights
                    acc1, acc5 = accuracy(logits, target.cuda(), topk=(1, 5))
                    top1 += acc1
                    top5 += acc5
                    n += len(image)
    
        top1 = (top1.item() / n) * 100
        top5 = (top5.item() / n) * 100 
        
        print(f"Top-1 accuracy: {top1:.2f}%")
        print(f"Top-5 accuracy: {top5:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/yuchunli/datasets/perf-benchmark-dataset"))
    parser.add_argument("--output_dir", type=Path, default=Path("zero_shot_result"))
    parser.add_argument("--result_name", type=str, default="satellite_prompt_satellite_backbone")
    args = parser.parse_args()
    main(args.dataset_root, args.output_dir, args.result_name)
