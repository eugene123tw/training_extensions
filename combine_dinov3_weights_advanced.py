#!/usr/bin/env python3
"""
Advanced DINOv3 Weights Combination Script

This script intelligently combines DINOv3 backbone pretrained weights with detection head weights,
adding a "backbone." prefix to all backbone weights to avoid conflicts.
"""

import torch
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict


def analyze_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze a checkpoint file and return state dict with metadata."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Analyzing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict and metadata
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            metadata = {k: v for k, v in checkpoint.items() if k != 'model'}
        else:
            state_dict = checkpoint
            metadata = {}
    else:
        state_dict = checkpoint
        metadata = {}
    
    # Analyze state dict
    analysis = {
        'total_params': len(state_dict),
        'param_shapes': {k: list(v.shape) for k, v in state_dict.items()},
        'param_dtypes': {k: str(v.dtype) for k, v in state_dict.items()},
        'key_prefixes': defaultdict(int)
    }
    
    # Count key prefixes to understand structure
    for key in state_dict.keys():
        prefix = key.split('.')[0] if '.' in key else key
        analysis['key_prefixes'][prefix] += 1
    
    print(f"  - Total parameters: {analysis['total_params']}")
    print(f"  - Key prefixes: {dict(analysis['key_prefixes'])}")
    
    return state_dict, analysis


def smart_combine_weights(
    backbone_path: str,
    detection_head_path: str,
    output_path: str,
    overwrite: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Intelligently combine backbone and detection head weights with backbone prefixing.
    
    Args:
        backbone_path: Path to backbone pretrained weights
        detection_head_path: Path to detection head weights
        output_path: Path to save combined weights
        overwrite: Whether to overwrite existing output file
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with combination statistics
    """
    
    if os.path.exists(output_path) and not overwrite:
        print(f"Output file already exists: {output_path}")
        print("Use overwrite=True to overwrite existing file")
        return {}
    
    print("=" * 70)
    print("Advanced DINOv3 Weights Combination")
    print("=" * 70)
    
    # Load and analyze both checkpoints
    print("\n1. Loading and analyzing backbone weights...")
    backbone_state_dict, backbone_analysis = analyze_checkpoint(backbone_path)
    
    print("\n2. Loading and analyzing detection head weights...")
    detection_state_dict, detection_analysis = analyze_checkpoint(detection_head_path)
    
    # Find conflicts and overlaps (after prefixing)
    print("\n3. Analyzing conflicts and overlaps...")
    backbone_keys_prefixed = {f"backbone.{key}" for key in backbone_state_dict.keys()}
    detection_keys = set(detection_state_dict.keys())
    
    conflicts = backbone_keys_prefixed.intersection(detection_keys)
    backbone_only = backbone_keys_prefixed - detection_keys
    detection_only = detection_keys - backbone_keys_prefixed
    
    print(f"  - Backbone-only keys (with prefix): {len(backbone_only)}")
    print(f"  - Detection-only keys: {len(detection_only)}")
    print(f"  - Conflicting keys: {len(conflicts)}")
    
    if verbose and conflicts:
        print("\n  Conflicting keys (detection head will override):")
        for key in sorted(conflicts):
            # Remove prefix for shape lookup
            original_key = key.replace("backbone.", "")
            backbone_shape = backbone_analysis['param_shapes'][original_key]
            detection_shape = detection_analysis['param_shapes'][key]
            print(f"    {key}: backbone {backbone_shape} -> detection {detection_shape}")
    
    if verbose and len(conflicts) == 0:
        print("  - No conflicts found (backbone weights will be prefixed)")
    
    # Combine weights with backbone prefix
    print("\n4. Combining weights...")
    combined_state_dict = {}
    
    # Add backbone weights with "backbone." prefix
    backbone_count = 0
    for key, value in backbone_state_dict.items():
        prefixed_key = f"backbone.0._backbone.backbone.{key}"
        combined_state_dict[prefixed_key] = value
        backbone_count += 1
    
    # Add detection head weights (no prefix needed)
    detection_count = 0
    for key, value in detection_state_dict.items():
        combined_state_dict[key] = value
        detection_count += 1
    
    print(f"  - Backbone parameters added: {backbone_count}")
    print(f"  - Detection parameters added: {detection_count}")
    print(f"  - Total combined parameters: {len(combined_state_dict)}")
    
    # Save combined weights
    print(f"\n5. Saving combined weights to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the combined state dict
    torch.save(combined_state_dict, output_path)
    
    # Verify the saved file
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"  - File size: {file_size:.2f} MB")
    
    # Create detailed report
    report = {
        'backbone_path': backbone_path,
        'detection_head_path': detection_head_path,
        'output_path': output_path,
        'backbone_params': backbone_count,
        'detection_params': detection_count,
        'total_params': len(combined_state_dict),
        'conflicts': list(conflicts),
        'backbone_only': list(backbone_only),
        'detection_only': list(detection_only),
        'backbone_prefix': 'backbone.0._backbone.backbone.',
        'file_size_mb': file_size
    }
    
    # Save report
    report_path = output_path.replace('.pth', '_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  - Detailed report saved to: {report_path}")
    
    print("\n✅ Weights combined successfully!")
    
    # Verify the combined weights
    print("\n6. Verifying combined weights...")
    try:
        loaded_weights = torch.load(output_path, map_location='cpu')
        print(f"  - Successfully loaded {len(loaded_weights)} parameters")
        
        # Check if all expected keys are present
        missing_keys = set(combined_state_dict.keys()) - set(loaded_weights.keys())
        if missing_keys:
            print(f"  - Warning: {len(missing_keys)} keys missing from loaded file")
        else:
            print("  - All keys present in loaded file")
        
        print("✅ Verification passed!")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
    
    return report


def main():
    """Main function to combine weights."""
    
    # Define paths
    backbone_path = "/home/yuchunli/git/dinov3/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    detection_head_path = "/home/yuchunli/git/dinov3/weights/dinov3_vit7b16_coco_detr_head-b0235ff7.pth"
    output_path = "/home/yuchunli/git/dinov3/weights/dinov3_vit7b16_combined.pth"
    
    # Check if input files exist
    if not os.path.exists(backbone_path):
        print(f"❌ Backbone weights not found: {backbone_path}")
        return False
    
    if not os.path.exists(detection_head_path):
        print(f"❌ Detection head weights not found: {detection_head_path}")
        return False
    
    try:
        # Combine weights
        report = smart_combine_weights(
            backbone_path=backbone_path,
            detection_head_path=detection_head_path,
            output_path=output_path,
            overwrite=True,
            verbose=True
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Combined weights saved to: {output_path}")
        print(f"Total parameters: {report['total_params']:,}")
        print(f"File size: {report['file_size_mb']:.2f} MB")
        print(f"Backbone parameters (prefixed): {report['backbone_params']:,}")
        print(f"Detection parameters: {report['detection_params']:,}")
        print(f"Conflicts: {len(report['conflicts'])}")
        print(f"Backbone prefix: {report['backbone_prefix']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to combine weights: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
