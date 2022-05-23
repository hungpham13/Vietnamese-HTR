import torchmetrics

from model.backbone.resnet import ResNet50_FeatureExtractor
from model.sequence.transformer import LanguageTransformer
import torch
from preprocessing.vocab import Vocab
from optim.criterion import LabelSmoothingLoss
import pytorch_lightning as pl

class TransformerOCR(pl.LightningModule):
    def __init__(self, cnn_args, transformer_args,optimizer_hparams):
        super(TransformerOCR, self).__init__()
        self.vocab = Vocab()
        vocab_size = len(self.vocab)

        # Exports the hyper parameters to a YAML file,
        # and create "self.hparams" namespace
        self.save_hyperparameters()

        # Create model
        self.cnn = ResNet50_FeatureExtractor(**cnn_args)
        self.transformer = LanguageTransformer(vocab_size, **transformer_args)

        # Create loss module
        self.criterion = LabelSmoothingLoss(vocab_size,
                                            padding_idx=self.vocab.pad,
                                            smoothing=0.1)
        # validation metrics
        self.cer = torchmetrics.CharErrorRate()
        self.wer = torchmetrics.WordErrorRate()


    def forward(self, img, tgt_input, tgt_key_padding_mask):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        outputs = self.transformer(src, tgt_input,
                                   tgt_key_padding_mask=tgt_key_padding_mask)
        return outputs

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, **self.hparams.optimizer_hparams)
        return optimizer

    def training_step(self, batch, batch_idx):
        # change device
        # batch = batch_to_device(batch, device)
        img, tgt_padding_mask= batch['img'], batch['tgt_padding_mask']
        tgt_output, tgt_input  = batch['tgt_output'], batch['tgt_input']
        # calc the loss
        outputs = self.forward(img, tgt_input,
                        tgt_key_padding_mask=tgt_padding_mask)

        outputs = outputs.view(-1, outputs.size(2))  # flatten(0, 1)
        tgt_output = tgt_output.view(-1)  # flatten()

        loss = self.criterion(outputs, tgt_output)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch[
            'tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

        outputs = self.model(img, tgt_input, tgt_padding_mask)
        #                loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))

        outputs = outputs.flatten(0, 1)
        tgt_output = tgt_output.flatten()
        loss = self.criterion(outputs, tgt_output)
        self.cer(outputs,tgt_output)
        self.wer(outputs,tgt_output)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('cer', self.cer, on_step=True, on_epoch=True)
        self.log('wer', self.wer, on_step=True, on_epoch=True)
