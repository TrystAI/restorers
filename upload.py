import wandb

with wandb.init(project="mirnet-v2", entity="ml-colabs", job_type="upload"):
    artifact = wandb.Artifact(
        name="lol-dataset",
        type="dataset",
        metadata={
            "Original-Dataset": "https://daooshee.github.io/BMVC2018website/",
            "Google-Drive": "https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB",
        },
    )
    artifact.add_dir(local_path="/home/soumikrakshit/mirnetv2/artifacts/lol-dataset:v0")
    wandb.log_artifact(
        artifact,
        # aliases=["preprocessed", "expert-c"]
    )
