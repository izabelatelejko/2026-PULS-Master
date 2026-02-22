"""Module for dataset classes and functions."""

import numpy as np
from typing import Optional
import torch

from nnPU.dataset import (
    PUDatasetBase,
    BinaryTargetTransformer,
    PULabeler,
    MNIST_PU,
    FashionMNIST_PU,
    ChestXRay_PU,
)


class Gauss_PULS(PUDatasetBase):

    def __init__(
        self,
        root,
        pu_labeler: PULabeler = None,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=[1, -1], positive_classes=[1]
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
        shifted_prior: Optional[float] = None,
        n_samples: Optional[int] = None,
    ) -> None:
        self.root = root
        self.train = train
        self.download = download
        self.random_seed = random_seed
        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler

        self.data = torch.cat(
            [
                torch.normal(0, 1, (3000, 10)),
                torch.normal(0.8, 1, (3000, 10)),  # change back to 0.5
            ]
        )
        self.targets = torch.cat(
            [
                torch.ones(3000),
                -1 * torch.ones(3000),
            ]
        )

        self._convert_to_shifted_pu_data(shifted_prior, n_samples)


class MNIST_PULS(MNIST_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(10),
            positive_classes=[1, 3, 5, 7, 9],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
        shifted_prior: Optional[float] = None,
        n_samples: Optional[int] = None,
    ):
        super().__init__(
            root=root,
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            download=download,
            random_seed=random_seed,
        )

        self._convert_to_shifted_pu_data(shifted_prior, n_samples)

    def _convert_to_pu_data(self):
        pass


class FashionMNIST_PULS(FashionMNIST_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(10),
            positive_classes=[0, 2, 3, 4, 6],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
        shifted_prior: Optional[float] = None,
        n_samples: Optional[int] = None,
    ):
        super().__init__(
            root=root,
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            download=download,
            random_seed=random_seed,
        )

        self._convert_to_shifted_pu_data(shifted_prior, n_samples)

    def _convert_to_pu_data(self):
        pass


class ChestXRay_PULS(ChestXRay_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
        shifted_prior: Optional[float] = None,
        n_samples: Optional[int] = None,
    ):
        super().__init__(
            root=root,
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            download=download,
            random_seed=random_seed,
        )

        self._convert_to_shifted_pu_data(shifted_prior, n_samples)

    def _convert_to_pu_data(self):
        pass
