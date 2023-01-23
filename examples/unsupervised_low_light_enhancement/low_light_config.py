import ml_collections


def get_dataloader_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.image_size = 256
    config.bit_depth = 8
    config.val_split = 0.2
    config.local_batch_size = 4
    config.visualize_on_wandb = False
    config.dataset_artifact_address = "ml-colabs/mirnet-v2/lol-dataset:v0"

    return config


def get_model_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.num_intermediate_filters = 32
    config.num_iterations = 8

    return config


def get_training_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.global_batch_size = 8
    config.learning_rate = 1e-4
    config.save_best_checkpoint_only = False
    config.num_evaluation_batches = 2
    config.epochs = 100

    return config


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.seed = 42
    config.data_loader_configs = get_dataloader_configs()
    config.model_configs = get_model_configs()
    config.training_configs = get_training_configs()

    return config
