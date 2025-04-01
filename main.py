"""Main entry point for ResNet image classifier."""
import argparse
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="ResNet Image Classifier")
    parser.add_argument("--data", required=True, help="Path to image directory")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    args = parser.parse_args()
    trainer = Trainer(args.config)
    if args.mode == "train":
        history = trainer.train(args.data)
        print("Training complete.")
    print("Done.")

if __name__ == "__main__":
    main()
