import wandb
from restorers.evaluation import LoLEvaluator
from restorers.metrics import PSNRMetric, SSIMMetric


with wandb.init(project="mirnet-v2", entity="ml-colabs"):
    evaluator = LoLEvaluator(
        metrics=[PSNRMetric(max_val=1.0), SSIMMetric(max_val=1.0)],
        dataset_artifact_address="ml-colabs/dataset/LoL:v0",
    )
    evaluator.initialize_model_from_wandb_artifact(
        "ml-colabs/low-light-enhancement/run_v30syo8n_model:v99"
    )
    evaluator.evaluate()
