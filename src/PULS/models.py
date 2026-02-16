"""Module for the PULS experiment configuration models."""

from typing import Optional
from pydantic import BaseModel


class PiEstimates(BaseModel):

    true: Optional[float] = None
    km1: Optional[float] = None
    km2: Optional[float] = None
    dre: Optional[float] = None
    dre_from_mixed: Optional[float] = None
    mlls_nnpu: Optional[float] = None
    mlls_drpu: Optional[float] = None
    n_iter_mlls_nnpu: Optional[int] = None
    n_iter_mlls_drpu: Optional[int] = None


class LabelShiftConfig(BaseModel):

    train_prior: Optional[float] = None
    train_n_samples: Optional[int] = None
    test_prior: Optional[float] = None
    test_n_samples: Optional[int] = None
    mixed_prior: Optional[float] = None
    mixed_n_samples: Optional[int] = None

