import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    '''
    Ler o config/params
    Inicializar parâmetros do pl.Trainer
    Instanciar o Trainer
    Treinar, validar ou testar 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("configs")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    callbacks = [ModelCheckpoint()]
    logger = WandbLogger()
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         precision="16-mixed",
                         max_epochs=None,
                         fast_dev_run=args.debug,
                         )
    
    model = pl.LightningModule()
    datamodule = pl.LightningDataModule()

    if "training":
        trainer.fit(model, datamodule=datamodule)
    elif "validate": 
        trainer.validate(model, datamodule=datamodule, ckpt_path=None)
    elif "test":
        trainer.test(model, datamodule=datamodule, ckpt_path=None)