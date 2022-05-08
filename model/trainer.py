import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from transformerocr import TransformerOCR
from preprocessing.loader import create_train_test_loader

CHECKPOINT_PATH = "./saved_models/"


def train_transformer(**kwargs):
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "transformer_ocr")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[
                             ModelCheckpoint(save_weights_only=True, mode="max",
                                             monitor="cer")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=200,
                         gradient_clip_val=2,
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    train_loader, val_loader = create_train_test_loader()

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SetAnomalyTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = TransformerOCR.load_from_checkpoint(pretrained_filename)
    else:
        model = TransformerOCR(max_iters=trainer.max_epochs * len(train_loader),
                               **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = TransformerOCR.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result = trainer.test(model, val_loader, verbose=False)
    # test_result = trainer.test(model, test_anom_loader, verbose=False)
    result = {"val_acc": val_result[0]["test_acc"],
              "train_acc": train_result[0]["test_acc"]}

    model = model.to(device)
    return model, result
