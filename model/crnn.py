import torchmetrics

from model.backbone.baseline_cnn import BaseCNN
from model.sequence.bilstm import BRNN
import torch
from torch import nn
from preprocessing.vocab import Vocab
import pytorch_lightning as pl


class CRNN(pl.LightningModule):
    def __init__(self, optimizer_hparams):
        super(CRNN, self).__init__()
        self.vocab = Vocab()
        vocab_size = len(self.vocab)

        # Exports the hyper parameters to a YAML file,
        # and create "self.hparams" namespace
        self.save_hyperparameters()

        # Create model
        self.cnn = BaseCNN(input_channels=3)
        self.seq = BRNN(input_size=1024, vocab_size=vocab_size)

        # Create loss module
        self.criterion = nn.CTCLoss()

        # validation metrics
        self.cer = torchmetrics.CharErrorRate()
        self.wer = torchmetrics.WordErrorRate()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            - imgs: (N, C, H, W)
            - outputs: (N, C, T)
        """
        outputs = self.cnn(imgs)
        outputs = torch.permute(outputs, (0,2,1))
        outputs = self.seq(outputs)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        return optimizer

    def training_step(self, batch, batch_idx):
        img, tgt_output= batch['img'], batch['tgt_output']
        # calc the loss
        outputs = self.forward(img)

        # outputs = outputs.view(-1, outputs.size(2))  # flatten(0, 1)
        # tgt_output = tgt_output.view(-1)  # flatten()

        loss = self.criterion(outputs, tgt_output)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, tgt_output= batch['img'], batch['tgt_output']

        outputs = self.forward(img)

        outputs = outputs.flatten(0, 1)
        tgt_output = tgt_output.flatten()

        loss = self.criterion(outputs, tgt_output)
        self.cer(outputs, tgt_output)
        self.wer(outputs, tgt_output)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('cer', self.cer, on_step=True, on_epoch=True)
        self.log('wer', self.wer, on_step=True, on_epoch=True)


if __name__ == "__main__":
    from torchinfo import summary
    a = CRNN({'lr':1e-3})
    summary(a, (64,3,118,2167))
