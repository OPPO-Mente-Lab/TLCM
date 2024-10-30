import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="time magazine, top 100, leonardo dicaprio, highly realistic, detailed.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers."
    )
    
    parser.add_argument("--infer-steps", type=int, default=4, help="Inference steps")
    args = parser.parse_args()
    return args