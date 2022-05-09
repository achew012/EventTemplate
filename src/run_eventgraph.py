from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import LightningDataset
import os
import ast
from typing import Dict, Any, List, Tuple
from omegaconf import OmegaConf
import hydra
import ipdb
from clearml import Task, StorageManager, Dataset as ClearML_Dataset


from data.data import GraphDataset
from model.GraphModel import GraphEmbedding
from torch_geometric.data import DenseDataLoader, DataLoader
#from pytorch_lightning.trainer.supporters import CombinedLoader

Task.force_requirements_env_freeze(
    force=True, requirements_file="requirements.txt")
Task.add_requirements("hydra-core")
Task.add_requirements("pytorch-lightning")


def get_clearml_params(task: Task) -> Dict[str, Any]:
    """
    returns task params as a dictionary
    the values are casted in the required Python type
    """
    string_params = task.get_parameters_as_dict()
    clean_params = {}
    for k, v in string_params["General"].items():
        try:
            # ast.literal eval cannot read empty strings + actual strings
            # i.e. ast.literal_eval("True") -> True, ast.literal_eval("i am cute") -> error
            clean_params[k] = ast.literal_eval(v)
        except:
            # if exception is triggered, it's an actual string, or empty string
            clean_params[k] = v
    return OmegaConf.create(clean_params)


def get_dataloader(split_name, cfg) -> DataLoader:
    """Get training and validation dataloaders"""
    # =============================================================================
    #     clearml_data_object = ClearML_Dataset.get(
    #         dataset_name=cfg.clearml_dataset_name,
    #         dataset_project=cfg.clearml_dataset_project_name,
    #         dataset_tags=list(cfg.clearml_dataset_tags),
    #         # only_published=True,
    #     )
    # =============================================================================
    if cfg.dataset == "LDC":
        roots = ['../../../datasets/LDC_txt/train/',
                 '../../../datasets/LDC_txt/dev/', '../../../datasets/LDC_txt/test/']
    elif cfg.dataset == "IED":
        roots = ['../../../datasets/Wiki_IED_txt/train/',
                 '../../../datasets/Wiki_IED_txt/dev/', '../../../datasets/Wiki_IED_txt/test/']

    dataset_train = GraphDataset(roots[0], mode='train', form='homo')
    train_loader = DataLoader(
        dataset_train, batch_size=cfg.batch_size, shuffle=True)

    dev_loaders = []
    test_loaders = []
    for ratio in cfg.nargs_ratio:
        dataset_dev = GraphDataset(roots[1], mode='eval', form='homo', sample_ratio=ratio)
        dataset_test = GraphDataset(roots[2], mode='eval', form='homo', sample_ratio=ratio)
        dev_loaders.append(DataLoader(dataset_dev, batch_size=cfg.batch_size))
        test_loaders.append(DataLoader(
            dataset_test, batch_size=cfg.batch_size))

    cfg['num_event_types'] = dataset_train.num_event_types
    cfg['num_entity_types'] = dataset_train.num_entity_types
    cfg['num_rels'] = dataset_train.num_rels

    print("Event types:", cfg['num_event_types'])
    print("Entity types", cfg['num_entity_types'])
    print("Relation types:", cfg['num_rels'])

    if split_name == "val":
        return cfg, dev_loaders
    elif split_name == "test":
        return cfg, test_loaders
    else:
        return cfg, train_loader


def train(cfg, task) -> GraphEmbedding:
    callbacks = []

    if cfg.checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath="./",
            filename="best_event_graph_model",
            monitor="best_val_accuracy",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
            every_n_epochs=cfg.every_n_epochs,
        )
        callbacks.append(checkpoint_callback)

    if cfg.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="best_val_accuracy", min_delta=0.00, patience=5, verbose=True, mode="max"
        )
        callbacks.append(early_stop_callback)

    cfg, train_loader = get_dataloader("train", cfg)
    cfg, val_loaders = get_dataloader("val", cfg)

    model = GraphEmbedding(cfg, task)

    trainer = pl.Trainer(
        gpus=cfg.gpu,
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.grad_accum,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, val_dataloaders=val_loaders)
    return model


def test(cfg, model) -> List:
    cfg, test_loader = get_dataloader("test", cfg)
    trainer = pl.Trainer(gpus=cfg.gpu, max_epochs=cfg.num_epochs)
    results = trainer.test(model, test_loader)
    return results


@hydra.main(config_path=os.path.join("..", "config"), config_name="config")
def hydra_main(cfg) -> float:

    pl.seed_everything(cfg.seed, workers=True)

    if cfg.train:
        task = Task.init(
            project_name="Future!1D",
            task_name="future-train",
            output_uri="s3://experiment-logging/storage/",
        )
    else:
        task = Task.init(
            project_name="Future!1D",
            task_name="future-predict",
            output_uri="s3://experiment-logging/storage/",
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_dict)
    cfg = get_clearml_params(task)
    print("Detected config file, initiating task... {}".format(cfg))

    if cfg.remote:
        task.set_base_docker("nvidia/cuda:11.4.0-runtime-ubuntu20.04")
        task.execute_remotely(queue_name=cfg.queue, exit_process=True)

    if cfg.train:
        model = train(cfg, task)

    if cfg.test:
        if cfg.trained_model_path:
            trained_model_path = StorageManager.get_local_copy(
                cfg.trained_model_path)
            model = GraphEmbedding.load_from_checkpoint(
                trained_model_path, cfg=cfg, task=task
            )
        if model:
            results = test(cfg, model)


if __name__ == "__main__":
    hydra_main()
