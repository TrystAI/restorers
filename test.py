import wandb

from restorers.evaluation import LoLEvaluator
from restorers.metrics import PSNRMetric, SSIMMetric


with wandb.init(
    project="zero-dce", entity="ml-colabs", name="test-eval", job_type="test"
):
    evaluator = LoLEvaluator(
        metrics={
            "Peak-Singal-Noise-Ratio": PSNRMetric(max_val=1.0),
            "Structural-Similarity": SSIMMetric(max_val=1.0),
        },
        benchmark_against_input=True,
    )
    evaluator.initialize_model_from_wandb_artifact(
        artifact_address="ml-colabs/zero-dce/run_gu7a7tlx_model:v99"
    )
    evaluator.evaluate()
