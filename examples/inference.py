import wandb
import argparse

from restorers.inference import LowLightInferer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to perform inference useing a low-light enhancement model"
    )
    parser.add_argument("--image_file", type=str)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--wandb_entity_name", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default=None)
    parser.add_argument("--wandb_model_artifact", type=str)
    parser.add_argument("--resize_target", nargs="+", type=int, default=None)
    parser.add_argument(
        "--wandb_dataset_artifact", type=str, default="ml-colabs/dataset/LoL:v0"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_run_name,
        entity=args.wandb_entity_name,
        job_type=args.wandb_job_type,
        config=vars(args),
    )

    inferer = LowLightInferer(resize_target=tuple(args.resize_target))
    inferer.initialize_model_from_wandb_artifact(args.wandb_model_artifact)
    inferer.infer(image_file=args.image_file, output_path=self.output_path)

    if wandb.run is not None:
        wandb.init()
