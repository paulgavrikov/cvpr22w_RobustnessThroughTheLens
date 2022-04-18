import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from scheduler import WarmupCosineLR
from torch.optim.lr_scheduler import StepLR
import numpy as np
import models


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class TrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.myhparams = hparams
        self.save_hyperparameters()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = models.get_model(hparams["classifier"], dataset=hparams["dataset"],
                                      robustbench_model_dir=hparams["robustbench_model_dir"])
        self.acc_max = 0
        self.cutmix_beta = 1

#     def get_progress_bar_dict(self):
#         items = super().get_progress_bar_dict()
#         items["val_acc_max"] = self.acc_max
#         return items

    def forward(self, batch, metric=None):
        images, labels = batch
        if self.myhparams["aux_loss"] == 0 or not self.model.training:
            r = np.random.rand(1)
            if self.cutmix_beta > 0 and r < self.myhparams["cutmix_prob"]:
                lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(ims.size(), lam)
                ims[:, :, bbx1:bbx2, bby1:bby2] = ims[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                predictions = self.model(images)
                loss = self.criterion(predictions, target_a) * lam + self.criterion(predictions, target_b) * (1. - lam)
            else:
                predictions = self.model(images)
                loss = self.criterion(predictions, labels)
        else:
            predictions, aux_outputs = self.model(images)
            loss = self.criterion(predictions, labels) ** 2
            for aux_output in aux_outputs:
                loss += self.criterion(aux_output, labels) ** 2
            loss = torch.sqrt(loss)

        if metric is not None:
            accuracy = metric(predictions, labels)
            return loss, accuracy * 100
        else:
            return loss

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch, self.train_accuracy)
        return loss

    def training_epoch_end(self, outs):
        self.log("loss/train", np.mean([d["loss"].item() for d in outs]))
        self.log("acc/train", self.train_accuracy.compute() * 100)
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch, self.val_accuracy)
        return loss

    def validation_epoch_end(self, outs):
        self.log("loss/val", np.mean([d.item() for d in outs]))

        acc = self.val_accuracy.compute() * 100
        if acc > self.acc_max:
            self.acc_max = acc.item()

        self.log("acc_max/val", self.acc_max)
        self.log("acc/val", acc)
        self.val_accuracy.reset()

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):

        if self.myhparams["freeze"] == "conv":
            for module in self.model.modules():
                if type(module) == torch.nn.Conv2d:
                    for param in module.parameters():
                        param.requires_grad = False

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizers, schedulers = [], []

        if self.myhparams["optimizer"] == "sgd":
            optimizers.append(torch.optim.SGD(
                params,
                lr=self.myhparams["learning_rate"],
                weight_decay=self.myhparams["weight_decay"],
                momentum=self.myhparams["momentum"],
                nesterov=True
            ))
        else:
            optimizers.append(torch.optim.Adam(
                params,
                lr=self.myhparams["learning_rate"],
                weight_decay=self.myhparams["weight_decay"]
            ))

        if self.myhparams["scheduler"] == "WarmupCosine":
            total_steps = self.myhparams["max_epochs"] * len(self.train_dataloader())
            schedulers.append({
                "scheduler": WarmupCosineLR(optimizers[0], warmup_epochs=total_steps * 0.3, max_epochs=total_steps),
                "interval": "step",
                "name": "learning_rate",
            })
        elif self.myhparams["scheduler"] == "Step":
            schedulers.append({
                "scheduler": StepLR(optimizers[0], step_size=30, gamma=0.1),
                "interval": "epoch",
                "name": "learning_rate",
            })

        return optimizers, schedulers
