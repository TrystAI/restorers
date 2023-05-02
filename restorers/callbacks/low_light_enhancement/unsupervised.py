from typing import List

import numpy as np
import tensorflow as tf
from tqdm.autonotebook import tqdm

import wandb
from wandb.keras import WandbEvalCallback


class UnsupervisedLowLightEvaluationCallback(WandbEvalCallback):
    def __init__(
        self,
        vizualization_data: tf.data.Dataset,
        data_table_columns: List[str],
        pred_table_columns: List[str],
    ):
        super().__init__(data_table_columns, pred_table_columns)
        self.vizualization_data = vizualization_data
        self.dataset_cardinality = tf.data.experimental.cardinality(
            vizualization_data
        ).numpy()

    def postprocess(self, image):
        return (image * 255.0).clip(0, 255).astype(np.uint8)

    def add_ground_truth(self, logs=None):
        for _ in tqdm(range(self.dataset_cardinality)):
            input_image_batch = next(iter(self.vizualization_data))
            input_image_batch = self.postprocess(input_image_batch.numpy())
            self.data_table.add_data(wandb.Image(input_image_batch[0]))

    def add_model_predictions(self, epoch, logs=None):
        for count in tqdm(range(self.dataset_cardinality)):
            input_image_batch = next(iter(self.vizualization_data))
            prediction_batch = self.model.predict(input_image_batch, verbose=0)
            prediction_batch = self.postprocess(prediction_batch)
            data_table_ref = self.data_table_ref
            self.pred_table.add_data(
                epoch, data_table_ref.data[idx + count][0], wandb.Image(prediction[0])
            )
