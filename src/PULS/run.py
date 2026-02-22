"""Run PULS experiments."""

from PULS.models import LabelShiftConfig
from PULS.experiment import PULSExperiment

from nnPU.dataset import SCAR_CC_Labeler
from nnPU.dataset_config import DatasetConfig
from nnPU.experiment_config import ExperimentConfig
from nnPU.loss import nnPUccLoss, nnPUccLoss_CE



def run_experiments(dataset_name, dataset_class, n, label_frequency, train_pi_grid, test_pi_grid, K, mean=None, verbose=True):
    """Run all PULS experiments."""
    for train_pi in train_pi_grid:
        for test_pi in test_pi_grid:
            for exp_number in range(0, K):

                label_shift_config = LabelShiftConfig(train_prior=train_pi, train_n_samples=n, test_prior=test_pi, test_n_samples=n)

                dataset_config = DatasetConfig(
                    f"{dataset_name}/{label_shift_config.train_n_samples or 'all'}/{f'{mean}/' if mean is not None else ''}{label_shift_config.train_prior or 'all'}/{label_shift_config.test_prior or 'all'}",
                    DatasetClass=dataset_class,
                    PULabelerClass=SCAR_CC_Labeler,
                )
                
                experiment_config = ExperimentConfig(
                    PULoss=nnPUccLoss,  # sigmoid loss
                    dataset_config=dataset_config,
                    label_frequency=label_frequency,
                    exp_number=exp_number,
                )

                experiment = PULSExperiment(experiment_config=experiment_config, label_shift_config=label_shift_config)
                experiment.train_all()
                experiment.test_shifted()

                if verbose:
                    print(experiment.metrics)
                    print(experiment.test_pis)

                # Test nnPU with CE loss
                label_shift_config = LabelShiftConfig(train_prior=train_pi, train_n_samples=n, test_prior=test_pi, test_n_samples=n)

                dataset_config = DatasetConfig(
                    f"{dataset_name}-CE/{label_shift_config.train_n_samples or 'all'}/0.8/{label_shift_config.train_prior or 'all'}/{label_shift_config.test_prior or 'all'}",
                    DatasetClass=dataset_class,
                    PULabelerClass=SCAR_CC_Labeler,
                )
                
                experiment_config = ExperimentConfig(
                    PULoss=nnPUccLoss_CE,  # cross-entropy loss instead of sigmoid
                    dataset_config=dataset_config,
                    label_frequency=label_frequency,
                    exp_number=exp_number,
                )

                experiment = PULSExperiment(experiment_config=experiment_config, label_shift_config=label_shift_config)
                experiment.train_nnpu()
                experiment.test_nnpu_on_shifted()

                if verbose:
                    print(experiment.metrics)
                    print(experiment.test_pis)