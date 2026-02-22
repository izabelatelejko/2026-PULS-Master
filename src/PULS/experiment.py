"""Experiment class for PULS."""

import json
from typing import TYPE_CHECKING, Optional
from PULS.const import ModelType
import numpy as np
import pkbar
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.optim import Adam

from DRPU.algorithm import priorestimator as ratio_estimator
from DRPU.algorithm import PUsequence, to_ndarray
from DRPU.modules.Kernel_MPE import KM1_KM2_estimate
from nnPU.run_experiment import Experiment, DictJsonEncoder
from nnPU.experiment_config import ExperimentConfig
from nnPU.model import PUModel
from nnPU.loss import DRPUccLoss

from PULS.models import PiEstimates

if TYPE_CHECKING:
    from PULS.models import LabelShiftConfig


class PULSExperiment(Experiment):
    """Experiment class for PULS data."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        label_shift_config: "LabelShiftConfig",
    ) -> None:
        """Initialize the PULS experiment."""
        self.metrics = {}
        self.label_shift_config = label_shift_config

        super().__init__(experiment_config)
        # models
        self.ratio_model = PUModel(self.n_inputs, activate_output=True)
        self.model_from_mixed = PUModel(self.n_inputs, activate_output=False)
        self.ratio_model_from_mixed = PUModel(self.n_inputs, activate_output=True)
        # optimizers
        self.ratio_optimizer = Adam(
            self.ratio_model.parameters(),
            lr=self.experiment_config.dataset_config.learning_rate,
            weight_decay=0.005,
            betas=(0.9, 0.999),
        )
        self.from_mixed_optimizer = Adam(
            self.model_from_mixed.parameters(),
            lr=self.experiment_config.dataset_config.learning_rate,
            weight_decay=0.005,
            betas=(0.9, 0.999),
        )
        self.ratio_from_mixed_optimizer = Adam(
            self.ratio_model_from_mixed.parameters(),
            lr=self.experiment_config.dataset_config.learning_rate,
            weight_decay=0.005,
            betas=(0.9, 0.999),
        )
        self.ratio_train_metrics = []

    def get_model(self, model_type: ModelType) -> PUModel:
        """Get the model from ModelType enum."""
        model_map = {
            ModelType.NNPU: self.model,
            ModelType.DRPU: self.ratio_model, 
            ModelType.MIXED_NNPU: self.model_from_mixed,
            ModelType.MIXED_DRPU: self.ratio_model_from_mixed,
        }
        return model_map[model_type]

    def _prepare_data(self):
        self._set_seed()

        data = {}
        data["train"] = self.experiment_config.dataset_config.DatasetClass(
            self.experiment_config.data_dir,
            self.experiment_config.dataset_config.PULabelerClass(
                label_frequency=self.experiment_config.label_frequency
            ),
            train=True,
            download=True,
            random_seed=self.experiment_config.seed,
            shifted_prior=self.label_shift_config.train_prior,
            n_samples=self.label_shift_config.train_n_samples,
        )
        data["test"] = self.experiment_config.dataset_config.DatasetClass(
            self.experiment_config.data_dir,
            self.experiment_config.dataset_config.PULabelerClass(label_frequency=0),
            train=False,
            download=True,
            random_seed=self.experiment_config.seed,
            shifted_prior=self.label_shift_config.test_prior,
            n_samples=self.label_shift_config.test_n_samples,
        )
        # Mixed dataset: take labeled positive from train and unlabeled from test
        data["mixed_train"] = data["train"].copy()
        data["mixed_train"]._replace_unlabeled_with_target_data(data["test"].data, data["test"].binary_targets)
        self.mixed_set = data["mixed_train"]
        self.train_set = data["train"]
        self.prior = self.train_set.get_prior()
        self.label_shift_config.train_prior = (
            self.label_shift_config.train_prior or self.train_set.get_prior()
        )
        self.label_shift_config.train_n_samples = (
            self.label_shift_config.train_n_samples or len(data["train"])
        )
        self.label_shift_config.test_prior = (
            self.label_shift_config.test_prior or data["test"].pu_labeler.prior
        )
        self.label_shift_config.test_n_samples = (
            self.label_shift_config.test_n_samples or len(data["test"])
        )
        self.label_shift_config.mixed_n_samples = len(data["mixed_train"])
        self.label_shift_config.mixed_prior = data["mixed_train"].get_new_prior()

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.experiment_config.dataset_config.train_batch_size,
            shuffle=True,
        )
        self.n_inputs = len(next(iter(self.train_set))[0].reshape(-1))

        self.test_set = data["test"]
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.experiment_config.dataset_config.eval_batch_size,
            shuffle=False,
        )

        self.mixed_train_set = data["mixed_train"]
        self.mixed_train_loader = DataLoader(
            self.mixed_train_set,
            batch_size=self.experiment_config.dataset_config.train_batch_size,
            shuffle=True,
        )

        self.metrics["dataset_stats"] = {
            "train": data["train"].dataset_stats,
            "mixed_train": data["mixed_train"].dataset_stats,
            "test": data["test"].dataset_stats,
        }

    def _train_step_ratio_estimator(self, epoch: int, kbar: pkbar.Kbar) -> None:
        """Train the ratio estimator model."""
        self.ratio_model.train()
        tr_loss = 0

        loss_fct = DRPUccLoss(prior=self.prior, alpha=None)
        for batch_idx, (data, _, label) in enumerate(self.train_loader):
            data, label = data.to(self.device), label.to(self.device)
            self.ratio_optimizer.zero_grad()
            output = self.ratio_model(data)

            loss = loss_fct(output.view(-1), label.type(torch.float))
            tr_loss += loss.item()
            loss.backward()
            self.ratio_optimizer.step()

            kbar.update(batch_idx + 1, values=[("loss", loss)])

        metric_values = {"tr_loss": tr_loss, "epoch": epoch}
        self.ratio_train_metrics.append(metric_values)

    def train_ratio_estimator(self) -> None:
        """Train the density ratio estimator model."""
        self._set_seed()
        self.ratio_model = self.ratio_model.to(self.device)

        for epoch in range(self.experiment_config.dataset_config.num_epochs):
            kbar = pkbar.Kbar(
                target=len(self.train_loader) + 1,
                epoch=epoch,
                num_epochs=self.experiment_config.dataset_config.num_epochs,
                width=8,
                always_stateful=False,
            )
            self._train_step_ratio_estimator(epoch, kbar)

        kbar = pkbar.Kbar(
            target=1,
            epoch=epoch,
            num_epochs=self.experiment_config.dataset_config.num_epochs,
            width=8,
            always_stateful=False,
        )

        with open(self.experiment_config.drpu_metrics_file, "w") as f:
            json.dump(self.ratio_train_metrics, f, cls=DictJsonEncoder, indent=4)
        print("Metrics saved to", self.experiment_config.drpu_metrics_file)

    def _train_step_from_mixed(self, epoch: int, kbar: pkbar.Kbar, mixed_prior: float) -> None:
        """Train the nnPU model from mixed data."""
        self.model_from_mixed.train()
        tr_loss = 0

        loss_fct = self.experiment_config.PULoss(prior=mixed_prior)
        for batch_idx, (data, _, label) in enumerate(self.mixed_train_loader):
            data, label = data.to(self.device), label.to(self.device)
            self.from_mixed_optimizer.zero_grad()
            output = self.model_from_mixed(data)

            loss = loss_fct(output.view(-1), label.type(torch.float))
            tr_loss += loss.item()
            loss.backward()
            self.from_mixed_optimizer.step()

            kbar.update(batch_idx + 1, values=[("loss", loss)])

        metric_values = {"tr_loss": tr_loss, "epoch": epoch}
        self.from_mixed_train_metrics.append(metric_values)

    def train_from_mixed(self) -> None:
        """Train the nnPU model from mixed data."""
        self._set_seed()
        self.model_from_mixed = self.model_from_mixed.to(self.device)
        self.from_mixed_train_metrics = []
        mixed_prior = self.estimate_mixed_prior()

        for epoch in range(self.experiment_config.dataset_config.num_epochs):
            kbar = pkbar.Kbar(
                target=len(self.mixed_train_loader) + 1,
                epoch=epoch,
                num_epochs=self.experiment_config.dataset_config.num_epochs,
                width=8,
                always_stateful=False,
            )
            self._train_step_from_mixed(epoch, kbar, mixed_prior=mixed_prior)

        print("Mixed-nnPU training complete.")

    def _train_step_ratio_from_mixed(self, epoch: int, kbar: pkbar.Kbar, mixed_prior: float) -> None:
        """Train the DRPU ratio estimator model from mixed data."""
        self.ratio_model_from_mixed.train()
        tr_loss = 0

        loss_fct = DRPUccLoss(prior=mixed_prior, alpha=None)
        for batch_idx, (data, _, label) in enumerate(self.mixed_train_loader):
            data, label = data.to(self.device), label.to(self.device)
            self.ratio_from_mixed_optimizer.zero_grad()
            output = self.ratio_model_from_mixed(data)

            loss = loss_fct(output.view(-1), label.type(torch.float))
            tr_loss += loss.item()
            loss.backward()
            self.ratio_from_mixed_optimizer.step()

            kbar.update(batch_idx + 1, values=[("loss", loss)])

        metric_values = {"tr_loss": tr_loss, "epoch": epoch}
        self.ratio_from_mixed_train_metrics.append(metric_values)

    def train_ratio_from_mixed(self) -> None:
        """Train the DRPU ratio estimator model from mixed data."""
        self._set_seed()
        self.ratio_model_from_mixed = self.ratio_model_from_mixed.to(self.device)
        self.ratio_from_mixed_train_metrics = []
        mixed_prior = self.estimate_mixed_prior()

        for epoch in range(self.experiment_config.dataset_config.num_epochs):
            kbar = pkbar.Kbar(
                target=len(self.mixed_train_loader) + 1,
                epoch=epoch,
                num_epochs=self.experiment_config.dataset_config.num_epochs,
                width=8,
                always_stateful=False,
            )
            self._train_step_ratio_from_mixed(epoch, kbar, mixed_prior=mixed_prior)

        print("Mixed-DRPU training complete.")

    def estimate_mixed_prior(self) -> None:
        """Estimate the prior of the mixed training set using density ratio method."""
        pos = self.train_set.data.clone()[self.train_set.pu_targets == 1].numpy()
        unl = self.mixed_set.data.clone().numpy() 
        _, KM2 = KM1_KM2_estimate(pos, unl)
        return KM2

    def _estimate_test_km_priors(self) -> tuple[float, float]:
        """Estimate the prior of test set with KM1 and KM2 methods."""
        pos = self.train_set.data.clone()[self.train_set.pu_targets == 1].numpy()
        unl = self.test_set.data.clone().numpy()
        KM1, KM2 = KM1_KM2_estimate(pos, unl)

        return KM1, KM2

    def _estimate_test_density_ratio_prior(self, model_type: ModelType = ModelType.DRPU) -> float:
        """Estimate the prior of test set with density ratio method."""
        model = self.get_model(model_type)
        model.eval()
        with torch.no_grad():
            preds_P, preds_U = [], []

            # positive from training set
            for data, target, _ in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                preds = model(data) 
                preds = preds[target == 1]
                preds_P.append(to_ndarray(preds)) 

            # unlabeled from shifted data
            for data, target, _ in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                preds = model(data) 
                preds_U.append(to_ndarray(preds))

            preds_P = np.concatenate(preds_P)
            preds_U = np.concatenate(preds_U)

            prior = ratio_estimator(
                np.concatenate([preds_P, preds_U]),
                PUsequence(len(preds_P), len(preds_U)),
            )

        return prior

    def _estimate_test_pi(self, use_drpu: bool=True) -> None:
        """Estimate the test pi values using KM1, KM2, and DRE methods."""
        # True pi
        true_pi = self.label_shift_config.test_prior

        # KM1, KM2
        KM1, KM2 = self._estimate_test_km_priors()

        # Density ratio
        if use_drpu:
            ratio_pi = self._estimate_test_density_ratio_prior()
            mixed_ratio_pi = self._estimate_test_density_ratio_prior(ModelType.MIXED_DRPU)
        else:
            ratio_pi = None
            mixed_ratio_pi = None

        mixed_prior = self.label_shift_config.mixed_prior
        mixed_prior_km2 = self.estimate_mixed_prior()

        self.test_pis = PiEstimates(
            true=true_pi,
            km1=KM1,
            km2=KM2,
            dre=ratio_pi,
            dre_from_mixed=mixed_ratio_pi,
            mixed_prior=mixed_prior,
            mixed_prior_km2=mixed_prior_km2,
        )

    def _run_mlls(
        self,
        model: torch.nn.Module,
        train_prior: float,
        tol: float = 1e-3,
        max_iter: int = 100,
        factor: Optional[float] = None,
    ) -> tuple[float, torch.Tensor, torch.Tensor, int]:
        """
        Estimate shifted test prior using Maximum Likelihood Label Shift (MLLS)
        with EM algorithm for the binary classification setting.
        """
        new_pi = train_prior
        old_pi = 0.0
        probs = torch.Tensor([])
        targets = torch.Tensor([])
        n_samples = 0

        model.eval()
        with torch.no_grad():
            for data, target, _ in self.test_loader:
                data = data.to(self.device)
                outputs = model(data)
                if factor is not None:
                    outputs = outputs * factor
                    # Clamp outputs to [0, 1]
                    outputs = torch.clamp(outputs, min=0.0, max=1.0)
                else:
                    outputs = torch.sigmoid(outputs)
                batch_probs = outputs.view(-1).cpu()
                probs = torch.cat((probs, batch_probs), dim=0)
                targets = torch.cat((targets, target.view(-1).cpu()), dim=0)
                n_samples += data.size(0)

        for iteration in range(max_iter):
            # E-step: compute expected posteriors for all test samples
            # Apply MLLS formula for adjusted posteriors:
            # p'(y=1|x) = [ (π'/π) * p(y=1|x) ] /
            #             [ (π'/π)*p(y=1|x) + ((1-π')/(1-π))*(1-p(y=1|x)) ]
            adjusted_post_plus = (new_pi / train_prior) * probs
            adjusted_post_minus = ((1 - new_pi) / (1 - train_prior)) * (1 - probs)
            adjusted_post = adjusted_post_plus / (
                adjusted_post_plus + adjusted_post_minus
            )

            # M-step: update π' using mean of adjusted posteriors
            old_pi = new_pi
            new_pi = adjusted_post.mean().item()

            # Check convergence
            if abs(new_pi - old_pi) < tol:
                break

        return new_pi, adjusted_post, targets, iteration + 1


    def _test_with_threshold(
        self,
        estimated_pi: Optional[float] = None,
        model_type: ModelType = ModelType.NNPU,
        calculate_roc_curve: bool = False,
    ):
        """Testing with Threshold Adjustment method."""
        model = self.get_model(model_type)
        model.eval()

        factor = 1
        if model_type == ModelType.DRPU:
            factor = self.prior.item()
        elif model_type == ModelType.MIXED_DRPU:
            assert estimated_pi is not None, "Estimated pi must be provided for Mixed-DRPU"
            factor = estimated_pi

        # assuming train PI is known
        if estimated_pi is not None and model_type in [ModelType.NNPU, ModelType.DRPU]:
            threshold = (self.prior.item() * (1 - estimated_pi)) / (
                (
                    self.prior.item()
                    + estimated_pi
                    - 2 * self.prior.item() * estimated_pi
                )
            )
        else:
            # use default threshold 0.5 if no estimate is provided
            # set threshold to 0.5 for Mixed setup
            threshold = 0.5

        test_loss = 0
        correct = 0
        num_pos = 0

        test_points = []
        targets = []
        preds = []
        outputs = []
        posterior_outputs = []
        y_scores = []

        kbar = pkbar.Kbar(
            target=len(self.test_loader) + 1,
            epoch=0,
            num_epochs=1,
            width=8,
            always_stateful=False,
        )

        with torch.no_grad():
            if model_type in [ModelType.DRPU, ModelType.MIXED_DRPU]:
                test_loss_func = DRPUccLoss(prior=factor, alpha=None)
            elif model_type in [ModelType.NNPU]:
                test_loss_func = self.experiment_config.PULoss(
                    prior=self.prior
                )  # TODO: priors not always known
            else:
                test_loss_func = self.experiment_config.PULoss(
                    prior=self.label_shift_config.mixed_prior
                )  # for mixed model, use mixed prior
            for data, target, _ in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data).view(-1)
                outputs.append(output)

                if model_type in [ModelType.DRPU, ModelType.MIXED_DRPU]:
                    posterior_output = output * factor
                else:
                    posterior_output = torch.sigmoid(output)
                posterior_outputs.append(posterior_output)
                if calculate_roc_curve:
                    y_scores.append(posterior_output)

                test_loss += test_loss_func(
                    output.view(-1), target.type(torch.float)
                ).item()

                pred = torch.where(
                    posterior_output < threshold,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )
                num_pos += torch.sum(pred == 1)
                correct += pred.eq(target.view_as(pred)).sum().item()

                test_points.append(data)
                targets.append(target)
                preds.append(pred)

        test_loss /= len(self.test_loader)
        pos_fraction = float(num_pos) / len(self.test_loader.dataset)

        kbar.add(
            1,
            values=[
                ("test_loss", test_loss),
                ("accuracy", 100.0 * correct / len(self.test_loader.dataset)),
                ("pos_fraction", pos_fraction),
            ],
        )

        targets = torch.cat(targets).detach().cpu().numpy()
        preds = torch.cat(preds).detach().cpu().numpy()

        metric_values = self._calculate_metrics(targets, preds)
        metric_values.n = len(self.test_loader.dataset)
        metric_values.train_pi = self.prior.item()
        metric_values.estimated_test_pi = estimated_pi
        metric_values.threshold = threshold
        metric_values.true_test_pi = self.label_shift_config.test_prior

        if calculate_roc_curve:
            y_scores = torch.cat(y_scores).detach().cpu().numpy()
            fpr, tpr, thres = metrics.roc_curve(targets, y_scores, pos_label=1)
            roc_auc = metrics.auc(fpr, tpr)
            roc_curve = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thres.tolist(),
                "roc_auc": roc_auc,
            }
            return metric_values, roc_curve

        # plot densities
        import matplotlib.pyplot as plt

        outputs = torch.cat(outputs).detach().cpu().numpy()
        posterior_outputs = torch.cat(posterior_outputs).detach().cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.hist(outputs, bins=50, density=True, alpha=0.5, label="Raw outputs")
        plt.axvline(x=threshold, color="r", linestyle="--", label="Threshold")
        plt.title(f'Model outputs distributions with threshold={threshold:.2f} ({model_type.value})')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.hist(posterior_outputs, bins=50, density=True, alpha=0.5, label="Sigmoid outputs")
        plt.axvline(x=threshold, color="r", linestyle="--", label="Threshold")
        plt.title(f'Posterior distributions with threshold={threshold:.2f} ({model_type.value})')
        plt.show()

        return metric_values

    def _test_with_mlls(
        self,
        estimated_pi: float,
        probs: torch.Tensor,
        targets: list[torch.Tensor]
    ):
        """Testing with MLLS method."""
        threshold = 0.5
        preds = torch.where(
            probs < threshold,
            torch.tensor(-1, device=self.device),
            torch.tensor(1, device=self.device),
        )
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        metric_values = self._calculate_metrics(targets, preds)
        metric_values.n = len(self.test_loader.dataset)
        metric_values.train_pi = self.prior.item()
        metric_values.estimated_test_pi = estimated_pi
        metric_values.threshold = threshold
        metric_values.true_test_pi = self.label_shift_config.test_prior
        return metric_values

    def test_nnpu_on_shifted(self) -> None:
        """Test nnPU model on the shifted data."""
        self._estimate_test_pi(use_drpu=False)
        self.metrics["roc_curve"] = {"nnpu": {}}

        (
            self.test_pis.mlls_nnpu,
            mlls_nnpu_preds,
            targets,
            self.test_pis.n_iter_mlls_nnpu,
        ) = self._run_mlls(model=self.model, train_prior=self.prior.item())

        # No adjustment
        self.metrics["nnPU"], self.metrics["roc_curve"]["nnpu"] = (
            self._test_with_threshold(
                estimated_pi=None, model_type=ModelType.NNPU, calculate_roc_curve=True
            )
        )

        # Threshold adjustment with different pi estimates
        self.metrics["nnPU+TA+True"] = self._test_with_threshold(
            estimated_pi=self.test_pis.true, model_type=ModelType.NNPU
        )
        self.metrics["nnPU+TA+KM2"] = self._test_with_threshold(
            estimated_pi=self.test_pis.km2, model_type=ModelType.NNPU
        )
        self.metrics["nnPU+TA+DRE"] = self._test_with_threshold(
            estimated_pi=self.test_pis.dre, model_type=ModelType.NNPU
        )

        # MLLS
        self.metrics["nnPU+MLLS"] = self._test_with_mlls(
            self.test_pis.mlls_nnpu, mlls_nnpu_preds, targets
        )

        # Retrain with target data
        self.metrics["nnPU+Target"] = self._test_with_threshold(
            estimated_pi=None, model_type=ModelType.MIXED_NNPU
        )

        with open(self.experiment_config.metrics_file, "w") as f:
            json.dump(self.metrics, f, cls=DictJsonEncoder, indent=4)
        print("Metrics saved to", self.experiment_config.metrics_file)


    def test_shifted(self) -> None:
        """Test the model on the shifted data."""
        self._estimate_test_pi()
        self.metrics["roc_curve"] = {"nnpu": {}, "drpu": {}}

        (
            self.test_pis.mlls_nnpu,
            mlls_nnpu_preds,
            targets,
            self.test_pis.n_iter_mlls_nnpu,
        ) = self._run_mlls(model=self.model, train_prior=self.prior.item())
        self.test_pis.mlls_drpu, mlls_drpu_preds, _, self.test_pis.n_iter_mlls_drpu = (
            self._run_mlls(
                model=self.ratio_model,
                train_prior=self.prior.item(),
                factor=self.test_pis.dre,
            )
        )
        self.metrics["test_pis"] = self.test_pis.model_dump()

        # No adjustment
        self.metrics["nnPU"], self.metrics["roc_curve"]["nnpu"] = (
            self._test_with_threshold(
                estimated_pi=None, model_type=ModelType.NNPU, calculate_roc_curve=True
            )
        )
        self.metrics["DRPU"], self.metrics["roc_curve"]["drpu"] = (
            self._test_with_threshold(
                estimated_pi=self.test_pis.dre, model_type=ModelType.DRPU, calculate_roc_curve=True
            )
        )

        # Threshold adjustment with different pi estimates
        self.metrics["nnPU+TA+True"] = self._test_with_threshold(
            estimated_pi=self.test_pis.true, model_type=ModelType.NNPU
        )
        self.metrics["nnPU+TA+KM2"] = self._test_with_threshold(
            estimated_pi=self.test_pis.km2, model_type=ModelType.NNPU
        )
        self.metrics["nnPU+TA+DRE"] = self._test_with_threshold(
            estimated_pi=self.test_pis.dre, model_type=ModelType.NNPU
        )
        self.metrics["DRPU+TA+True"] = self._test_with_threshold(
            estimated_pi=self.test_pis.true, model_type=ModelType.DRPU
        )
        self.metrics["DRPU+TA+KM2"] = self._test_with_threshold(
            estimated_pi=self.test_pis.km2, model_type=ModelType.DRPU
        )

        # MLLS
        self.metrics["nnPU+MLLS"] = self._test_with_mlls(
            self.test_pis.mlls_nnpu, mlls_nnpu_preds, targets
        )
        self.metrics["DRPU+MLLS"] = self._test_with_mlls(
            self.test_pis.mlls_drpu, mlls_drpu_preds, targets
        )

        # Retrain with target data
        self.metrics["nnPU+Target"] = self._test_with_threshold(
            estimated_pi=None, model_type=ModelType.MIXED_NNPU
        )
        self.metrics["DRPU+Target"] = self._test_with_threshold(
            estimated_pi=self.test_pis.dre_from_mixed, model_type=ModelType.MIXED_DRPU
        )

        with open(self.experiment_config.metrics_file, "w") as f:
            json.dump(self.metrics, f, cls=DictJsonEncoder, indent=4)
        print("Metrics saved to", self.experiment_config.metrics_file)

    def train_nnpu(self) -> None:
        """Train nnPU models."""
        self.run()
        self.train_from_mixed()

    def train_all(self) -> None:
        """Train all models."""
        self.run()
        self.train_ratio_estimator()
        self.train_from_mixed()
        self.train_ratio_from_mixed()