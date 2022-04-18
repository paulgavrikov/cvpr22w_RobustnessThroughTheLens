import os
from argparse import ArgumentParser
import json
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from module import TrainModule
import data as datasets
import models
from utils import *
import logging
logging.getLogger('lightning').setLevel(0)


def none_or_str(value):  # from https://stackoverflow.com/questions/48295246/how-to-pass-none-keyword-as-command-line-argument
    if value == 'None':
        return None
    return value


def start_training(args):
    seed_everything(args["seed"])
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_id"]
        
    data_dir = os.path.join(args["data_dir"], args["dataset"])
    data = datasets.get_dataset(args["dataset"])(data_dir, args["batch_size"], args["num_workers"])

    args["num_classes"] = data.num_classes
    args["in_channels"] = data.in_channels

    model = TrainModule(args)
    if args["load_checkpoint"] is not None:
        state = torch.load(args["load_checkpoint"], map_location=model.device)
        if "state_dict" in state:
            state = state["state_dict"]
        
        model.model.load_state_dict(dict((key.replace("model.", ""), value) for (key, value) in
                                         state.items()))

    logger = CSVLogger(os.path.join(args["output_dir"], args["dataset"]), args["classifier"] + args["postfix"])
        
    checkpoint = MyCheckpoint(monitor="acc/val", mode="max", save_top_k=-1 if args["checkpoints"] == "all" else 1)

    trainer = Trainer(
        fast_dev_run=False,
        logger=logger,
        gpus=-1,
        deterministic=not args["cudnn_non_deterministic"],
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args["max_epochs"],
        checkpoint_callback=args["checkpoints"] is not None,
        precision=args["precision"],
        callbacks=None if args["checkpoints"] is None else checkpoint,
        num_sanity_val_steps=0  # sanity check must be turned off or bad performance callback will trigger.
    )

    trainer.fit(model, data)


def dump_info():
    print("Available models:")
    for x in models.all_classifiers.keys():
        print(f"\t{x}")
    print()
    print("Available data sets:")
    for x in datasets.all_datasets.keys():
        print(f"\t{x}")
    
    
def main(args):
    if type(args) is not dict:
        args = vars(args)

    if not args["info"]:
        start_training(args)
    else:
        dump_info()

        
if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--info", action="store_true")

    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--params", type=str, default=None)  # load params from json

    parser.add_argument("--checkpoints", type=str, default="last_best", choices=["all", "last_best", None])
    parser.add_argument("--classifier", type=str, default="lowres_resnet9")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--robustbench_model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--postfix", type=str, default="")

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--cudnn_non_deterministic", action="store_false", default=True)
    parser.add_argument("--gpu_id", type=str, default="0")
    
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("--scheduler", type=none_or_str, default="WarmupCosine", choices=["WarmupCosine", "Step", "None", None])
    parser.add_argument("--freeze", type=none_or_str, default=None, choices=["conv", "None", None])
    parser.add_argument("--cutmix_prob", type=float, default=0)
    parser.add_argument("--aux_loss", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    _args = parser.parse_args()
    
    if _args.params is not None:
        json_args = argparse.Namespace()
        with open(_args.params, "r") as f:
            json_args.__dict__ = json.load(f)

        _args = parser.parse_args(namespace=json_args)
    
    main(_args)
