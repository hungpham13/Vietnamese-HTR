import pytorch_lightning as pl
import glob
import os
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from model.transformerocr import TransformerOCR
from model.crnn import CRNN
from preprocessing.loader import create_train_test_loader

CHECKPOINT_PATH = "./saved_models/"
TEST_DIR = "data/Data 1: Handwriting OCR for Vietnamese Address/test_preprocessed"
TRAIN_DIR = ""


def train(**kwargs):
    if kwargs['model_name'] == "transformer":
        Model = TransformerOCR
    elif kwargs['model_name'] == "crnn":
        Model = CRNN

    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    pl.seed_everything(42)  # To be reproducable
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(kwargs['checkpoint_path'], kwargs['model_name'])
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[
                             ModelCheckpoint(save_weights_only=False, mode="max",
                                             monitor="val_cer"),
                             LearningRateMonitor("epoch"),
                         ],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=kwargs['max_epochs'],
                         gradient_clip_val=0.5,
                         progress_bar_refresh_rate=1)

    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    train_loader, val_loader = create_train_test_loader(kwargs['train_dir'],
                                                        kwargs['test_dir'],
                                                        kwargs['batch_size']
                                                        )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_found = glob.glob(os.path.join(root_dir, "./**/*.ckpt"), recursive=True)
    if pretrained_found:
        print("Found pretrained model, resume training...")
        model = Model(**kwargs['model_params'])
        trainer.fit(model, train_loader, val_loader, ckpt_path=pretrained_found[0])
    else:
        model = Model(**kwargs['model_params'])
        trainer.fit(model, train_loader, val_loader)
        model = Model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    # test_result = trainer.test(model, test_anom_loader, verbose=False)
    # result = {"cer_on_val": val_result[0]["test_cer"],
    #           "wer_on_val": val_result[0]["test_wer"]}
    print(val_result)

    model = model.to(device)
    return model, val_result


if __name__ == "__main__":
    data_dir = "data/"
    train_dir = data_dir + "0916_Data_Samples_2/"
    test_dir = data_dir + "1015_Private_Test/"
    train_pre_dir = data_dir + "train_preprocessed/"
    test_pre_dir = data_dir + "test_preprocessed/"
    config = {"batch_size": 32,
              "max_epochs": 200,
              "model_name": "crnn",
              "train_dir": train_dir,
              "test_dir": test_dir,
              "checkpoint_path": "model/saved_models",
              "model_params": {"optimizer_hparams": {"lr":1e-3},
                               },
              }
    model, result = train(**config)