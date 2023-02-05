"""
CLI Usage:
evaluate.py:
  --experiment_configs: path to config file.
    (default: 'None')
  --wandb_entity_name: Name of Weights & Biases Entity
  --wandb_job_type: Type of Weights & Biases Job
  --wandb_project_name: Name of Weights & Biases Project
Example of overriding default configs using the CLI:
evaluate.py:
  --experiment_configs configs/low_light.py
  --experiment_configs.data_loader_configs.batch_size 16
  --experiment_configs.model_configs.num_residual_recursive_groups 4
  --experiment_configs.training_configs.learning_rate 2e-4
"""

import wandb

from absl import app, flags, logging
from ml_collections.config_flags import config_flags

from restorers.evaluation import LoLEvaluator
from restorers.metrics import PSNRMetric, SSIMMetric


FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="wandb_project_name", default=None, help="Name of Weights & Biases Project"
)
flags.DEFINE_string(
    name="wandb_run_name", default=None, help="Name of Weights & Biases Run"
)
flags.DEFINE_string(
    name="wandb_entity_name", default=None, help="Name of Weights & Biases Entity"
)
flags.DEFINE_string(
    name="wandb_job_type", default=None, help="Type of Weights & Biases Job"
)
flags.DEFINE_string(
    name="model_artifact_address",
    default=None,
    help="The Weights & Biases artifact address for the model",
)
flags.DEFINE_string(
    name="benchmark_against_input", default=False, help="Benchmark against input"
)
config_flags.DEFINE_config_file("experiment_configs")


def main(_) -> None:
    using_wandb = False
    if FLAGS.wandb_project_name is not None:
        try:
            wandb.init(
                project=FLAGS.wandb_project_name,
                name=FLAGS.wandb_run_name,
                entity=FLAGS.wandb_entity_name,
                job_type=FLAGS.wandb_job_type,
                config=FLAGS.experiment_configs.to_dict(),
            )
            using_wandb = True
        except:
            logging.error("Unable to initialize_device wandb run.")

    evaluator = LoLEvaluator(
        metrics={
            "Peak-Singal-Noise-Ratio": PSNRMetric(max_val=1.0),
            "Structural-Similarity": SSIMMetric(max_val=1.0),
        },
        benchmark_against_input=FLAGS.benchmark_against_input,
    )
    evaluator.initialize_model_from_wandb_artifact(
        artifact_address=FLAGS.model_artifact_address
    )
    evaluator.evaluate()

    if using_wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
