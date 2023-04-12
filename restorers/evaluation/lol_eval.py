import os
from glob import glob
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

from .base import BaseEvaluator
from ..utils import fetch_wandb_artifact


class LoLEvaluator(BaseEvaluator):
    """Evaluator for [LoL Dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset).

    **Usage:**

    ```py
    import wandb
    from restorers.evaluation import LoLEvaluator
    from restorers.metrics import PSNRMetric, SSIMMetric

    # initialize a wandb run for inference
    wandb.init(project="low-light-enhancement", job_type="evaluation")

    # Define the Evaluator for LoL dataset
    evaluator = LoLEvaluator(
        # pass the list of Keras metrics to be evaluated for
        metrics=[PSNRMetric(max_val=1.0), SSIMMetric(max_val=1.0)],
        # pass the wandb artifact for the LoL dataset
        dataset_artifact_address="ml-colabs/dataset/LoL:v0",
        input_size=256,
    )
    # initialize model from wandb artifacts
    evaluator.initialize_model_from_wandb_artifact(
        "artifact-address-of-your-model-checkpoint"
    )
    # evaluate
    evaluator.evaluate()
    ```

    ??? example "Examples"
        - [Evaluating a low-light enhancement model](../../examples/evaluate_low_light).

    Args:
        metrics (List[tf.keras.metrics.Metric]): A list of metrics to be evaluated for.
        model (Optional[tf.keras.Model]): The model that is to be evaluated. Note that passing
            the model during initializing the evaluator is not compulsory. The model can also
            be set using the function `initialize_model_from_wandb_artifact`.
        input_size (Optional[int]): The input size for computing GFLOPs.
        resize_target (Optional[Tuple[int, int]]): The size that the input and the corresponding
            ground truth image should be resized to for inference and evaluation.
        dataset_artifact_address (str): Address of the WandB artifact hosting the LoL dataset.
    """

    def __init__(
        self,
        metrics: List[tf.keras.metrics.Metric],
        model: Optional[tf.keras.Model] = None,
        input_size: Optional[List[int]] = None,
        resize_target: Optional[Tuple[int, int]] = None,
        dataset_artifact_address: str = None,
    ) -> None:
        self.dataset_artifact_address = dataset_artifact_address
        super().__init__(metrics, model, input_size, resize_target)

    def preprocess(self, image: Image) -> Union[np.ndarray, tf.Tensor]:
        """Preprocessing logic for preprocessing a `PIL.Image` and add a batch dimension.

        Args:
            image (PIL.Image): A PIL Image.

        Returns:
            (Union[np.ndarray, tf.Tensor]): A numpy or Tensorflow tensor that would be fed to
                the model.
        """
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    def postprocess(self, model_output: np.ndarray) -> Image:
        """Postprocessing logic for converting the output of the model to a `PIL.Image`.

        Args:
            model_output (np.ndarray): Output of the model.

        Returns:
            (PIL.Image): The model output postprocessed to a PIL Image.
        """
        model_output = model_output * 255.0
        model_output = model_output.clip(0, 255)
        image = model_output[0].reshape(
            (np.shape(model_output)[1], np.shape(model_output)[2], 3)
        )
        return Image.fromarray(np.uint8(image))

    def populate_image_paths(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Populate the split-wise image paths necessary for the evaluation.

        Returns:
            (Dict[str, Tuple[List[str], List[str]]]): A dictionary of Image splits mapped to list
                of paths of input and corresponding ground-truth images. The dictionary in this case would be

        ```python
        {
            "Train-Val": (train_low_light_images, train_ground_truth_images),
            "Eval15": (test_low_light_images, test_ground_truth_images),
        }
        ```
        """
        dataset_path = fetch_wandb_artifact(
            self.dataset_artifact_address, artifact_type="dataset"
        )
        train_low_light_images = sorted(
            glob(os.path.join(dataset_path, "our485", "low", "*"))
        )
        train_ground_truth_images = sorted(
            glob(os.path.join(dataset_path, "our485", "high", "*"))
        )
        test_low_light_images = sorted(
            glob(os.path.join(dataset_path, "eval15", "low", "*"))
        )
        test_ground_truth_images = sorted(
            glob(os.path.join(dataset_path, "eval15", "high", "*"))
        )
        return {
            "Train-Val": (train_low_light_images, train_ground_truth_images),
            "Eval15": (test_low_light_images, test_ground_truth_images),
        }
