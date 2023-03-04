import wandb
import argparse

from restorers.evaluation import LoLEvaluator
from restorers.metrics import PSNRMetric, SSIMMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to evaluate a low-light enhancement model on the LoL Dataset"
    )
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_entity_name", type=str)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default=None)
    parser.add_argument("--wandb_model_artifact", type=str)
    parser.add_argument(
        "--wandb_dataset_artifact", type=str, default="ml-colabs/dataset/LoL:v0"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_run_name,
        entity=args.wandb_entity_name,
        job_type=args.wandb_job_type,
        config=vars(args),
    ):
        evaluator = LoLEvaluator(
            metrics=[PSNRMetric(max_val=1.0), SSIMMetric(max_val=1.0)],
            dataset_artifact_address=args.wandb_dataset_artifact,
            input_size=256,
        )
        evaluator.initialize_model_from_wandb_artifact(args.wandb_model_artifact)
        evaluator.evaluate()
