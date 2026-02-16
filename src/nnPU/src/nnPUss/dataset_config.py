"""DatasetConfig class - standalone to avoid importing all dataset classes."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset import PUDatasetBase, PULabeler


class DatasetConfig:
    def __init__(
        self,
        name: str,
        DatasetClass: "type[PUDatasetBase]",
        PULabelerClass: "type[PULabeler]",
        num_epochs=50,
        learning_rate=1e-5,
        train_batch_size=512,
        eval_batch_size=128,
    ):
        self.name = name
        self.DatasetClass = DatasetClass
        self.PULabelerClass = PULabelerClass
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
