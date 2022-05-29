import torchmetrics

from model.backbone.baseline_cnn import BaseCNN
from model.sequence.bilstm import BRNN
import torch
from torch import nn
from preprocessing.vocab import Vocab
import pytorch_lightning as pl
from tool.utils import non_zero_length




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
        self.val_cer = torchmetrics.CharErrorRate()
        self.test_cer = torchmetrics.CharErrorRate()
        self.val_wer = torchmetrics.WordErrorRate()
        self.test_wer = torchmetrics.WordErrorRate()

    def _forward(self, imgs: torch.Tensor) -> torch.Tensor:
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

    def _calculate_loss(self, batch):
        img, target = batch['img'], batch['tgt_output']
        # calc the loss
        outputs = self._forward(img)
        batch_size, seq_length = outputs.shape[0:2]
        outputs = torch.permute(outputs, (1, 0, 2)) # (N,T,C) -> (T,N,C)
        output_lengths = torch.full(size=(batch_size,), fill_value=seq_length, dtype=torch.long)
        target_lengths = torch.tensor([non_zero_length(seq) for seq in target])
        loss = self.criterion(outputs, target, output_lengths, target_lengths)
        return outputs, loss

    def _predict(self, outputs):
        """Turn log softmax to string"""
        outputs = torch.max(outputs, -1).indices
        return [self.vocab.decode(out) for out in outputs]

    def forward(self, imgs):
        return self._predict(self._foward(imgs))

    def training_step(self, batch, batch_idx):
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        outputs, loss = self._calculate_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, loss = self._calculate_loss(batch)
        outputs = self._predict(outputs)
        targets = [self.vocab.decode(out) for out in batch['tgt_output']]

        self.val_cer.update(outputs, targets)
        self.val_wer.update(outputs, targets)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('val_cer', self.cer, on_step=True, on_epoch=True)
        self.log('val_wer', self.wer, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch['img'])
        targets = [self.vocab.decode(out) for out in batch['tgt_output']]

        self.test_cer.update(outputs, targets)
        self.test_wer.update(outputs, targets)

        self.log('test_cer', self.test_cer, on_step=True, on_epoch=True)
        self.log('test_wer', self.test_wer, on_step=True, on_epoch=True)


if __name__ == "__main__":
    from torchinfo import summary
    a = CRNN({'lr':1e-3})
    summary(a, (64,3,118,2167))
