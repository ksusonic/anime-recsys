from tempfile import TemporaryDirectory

from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)


class Callbacks:
    def __init__(self):
        self._callbacks = []
        self.checkpoints_dir = None

    def to_list(self) -> list:
        return self._callbacks

    def with_checkpoints(self):
        """
        Model checkpoint callback to save the best weights
        """
        self.checkpoints_dir = TemporaryDirectory()
        self._callbacks.append(
            ModelCheckpoint(
                filepath=self.checkpoints_dir.name,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
            )
        )
        return self

    def with_lr_scheduler(
        self, start_lr=0.00001, min_lr=0.00001, max_lr=0.00005, rampup_epochs=5, sustain_epochs=0, exp_decay=0.8
    ):
        """
        Learning rate scheduler callback
        """
        self._callbacks.append(
            LearningRateScheduler(
                lambda epoch: self.lrfn(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)
            )
        )
        return self

    def with_early_stopping(self):
        """
        Early stopping callback to prevent overfitting
        """
        self._callbacks.append(EarlyStopping(patience=3, monitor='val_loss', mode='min', restore_best_weights=True))
        return self

    def __delete__(self, instance):
        if self.checkpoints_dir is not None:
            self.checkpoints_dir.cleanup()

    @staticmethod
    def lrfn(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        """
        Learning rate schedule function
        """
        if epoch < rampup_epochs:
            return (max_lr - start_lr) / rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            return max_lr
        else:
            return (max_lr - min_lr) * exp_decay ** (epoch - rampup_epochs - sustain_epochs) + min_lr
