import argparse
import json
from typing import NamedTuple
from pathlib import Path
import torch
import torch.nn.functional as F
from datumaro import Dataset as DmDataset

from otx.algo.detection.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from otx.config.data import SubsetConfig
from otx.data.factory import OTXDatasetFactory
from otx.metrics.accuracy import MultiClassClsMetricCallable
from otx.types.image import ImageColorChannel
from otx.types.task import OTXTaskType
from otx.types.transformer_libs import TransformLibType


DatasetInfo = NamedTuple("DatasetInfo", [("name", str), ("path", Path), ("group", str)])

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
    """Main function for zero-shot classification evaluation."""

    mean = (109.65 , 104.805,  75.48)
    std = (54.315, 39.78 , 36.465)

    # or 
    
    # mean = [123.675, 116.28, 103.53]
    # std = [58.395, 57.12, 57.375]


    dataset_infos = dataset_collections()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load DinoTXT model
    print("\nLoading DinoTXT model...")
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        # backbone_weights="/home/yuchunli/git/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        backbone_weights="/home/yuchunli/git/dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        dinotxt_weights="/home/yuchunli/git/dinov3/weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    )
    model = model.to(device).eval()

    results = {}
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
            batch_size=1,
            subset_name="test",
            transform_lib_type=TransformLibType.TORCHVISION,
            num_workers=0,
            to_tv_image=True,
            transforms=[
                {
                    "class_path": "torchvision.transforms.v2.Resize",
                    "init_args": {"size": [224, 224], "antialias": True}
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
        text_prompts = [f"satellite photo of {name.lower()}" for name in class_names]
        
        print(f"Text prompts: {text_prompts}")
        
        # 4. Encode text features once
        print("Encoding text features...")
        tokenized_texts = tokenizer.tokenize(text_prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokenized_texts)
            text_features = F.normalize(text_features, dim=-1)
        
        # 5. Setup metrics
        print("Setting up metrics...")
        metrics = MultiClassClsMetricCallable(otx_dataset.label_info).to(device)
        
        # 6. Perform zero-shot classification
        print("Performing zero-shot classification...")
        predictions = []
        targets = []
        
        for i in range(len(otx_dataset)):
            if i % 20 == 0:
                print(f"Processing {i}/{len(otx_dataset)}")
            
            # Get image and target
            data_item = otx_dataset[i]
            image = data_item.image.unsqueeze(0).to(device)
            target = data_item.label.to(device)
            
            # Encode image
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features = F.normalize(image_features, dim=-1)
            
            # Compute similarity and get prediction
            similarity = torch.matmul(text_features, image_features.T).squeeze()
            prediction = torch.argmax(similarity).unsqueeze(0)
            
            predictions.append(prediction)
            targets.append(target)
            
            # Show some examples
            if i < 3:
                pred_class = class_names[prediction.item()]
                true_class = class_names[target.item()]
                confidence = torch.max(similarity).item()
                print(f"  Sample {i}: Predicted '{pred_class}' ({confidence:.3f}), True: '{true_class}'")
        
        # 7. Compute metrics
        print("\nComputing metrics...")
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        
        metrics.update(predictions, targets)
        metric_values = metrics.compute()
        
        # make metric_values serializable
        metric_values = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metric_values.items()}

        results[dataset_info.name] = metric_values

    with open(output_dir / f"{result_name}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/yuchunli/datasets/perf-benchmark-dataset"))
    parser.add_argument("--output_dir", type=Path, default=Path("zero_shot_result"))
    parser.add_argument("--result_name", type=str, default="satellite_prompt_satellite_backbone")
    args = parser.parse_args()
    main(args.dataset_root, args.output_dir, args.result_name)
