# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_results

import torch.optim as optim
from torch.optim import lr_scheduler

class PoseTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (tuple): Names of the loss components used in training.

    Methods:
        get_model: Retrieve a pose estimation model with specified configuration.
        set_model_attributes: Set keypoints shape attribute on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_training_samples: Visualize training samples with keypoints.
        plot_metrics: Generate and save training/validation metric plots.
        get_dataset: Retrieve the dataset and ensure it contains required kpt_shape key.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[Dict[str, Any]] = None, _callbacks=None):
        """
        Initialize a PoseTrainer object for training YOLO pose estimation models.

        This initializes a trainer specialized for pose estimation tasks, setting the task to 'pose' and
        handling specific configurations needed for keypoint detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.

        Examples:
            >>> from ultralytics.models.yolo.pose import PoseTrainer
            >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
            >>> trainer = PoseTrainer(overrides=args)
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )
        self.args.K = [self.data["cx"], self.data["cy"], self.data["fx"], self.data["fy"]]

    def get_model(
        self,
        cfg: Optional[Union[str, Path, Dict[str, Any]]] = None,
        weights: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> PoseModel:
        """
        Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = PoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], K=self.args.K, depth=self.args.depth, verbose=verbose
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation."""
        # self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "rot_loss", "depth_loss", "consistency_loss"
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "rot_loss", "depth_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plot training/validation metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png

    def get_dataset(self) -> Dict[str, Any]:
        """
        Retrieve the dataset and ensure it contains the required `kpt_shape` key.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.

        Raises:
            KeyError: If the `kpt_shape` key is not present in the dataset.
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data


class PoseTrainerCustom(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (tuple): Names of the loss components used in training.

    Methods:
        get_model: Retrieve a pose estimation model with specified configuration.
        set_model_attributes: Set keypoints shape attribute on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_training_samples: Visualize training samples with keypoints.
        plot_metrics: Generate and save training/validation metric plots.
        get_dataset: Retrieve the dataset and ensure it contains required kpt_shape key.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[Dict[str, Any]] = None, _callbacks=None):
        """
        Initialize a PoseTrainer object for training YOLO pose estimation models.

        This initializes a trainer specialized for pose estimation tasks, setting the task to 'pose' and
        handling specific configurations needed for keypoint detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.

        Examples:
            >>> from ultralytics.models.yolo.pose import PoseTrainer
            >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
            >>> trainer = PoseTrainer(overrides=args)
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )
        self.args.K = [self.data["cx"], self.data["cy"], self.data["fx"], self.data["fy"]]

    def get_model(
        self,
        cfg: Optional[Union[str, Path, Dict[str, Any]]] = None,
        weights: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> PoseModel:
        """
        Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = PoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], K=self.args.K, depth=self.args.depth, verbose=verbose
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation."""
        # self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "rot_loss", "depth_loss", "consistency_loss"
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "rot_loss", "depth_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plot training/validation metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png

    def get_dataset(self) -> Dict[str, Any]:
        """
        Retrieve the dataset and ensure it contains the required `kpt_shape` key.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.

        Raises:
            KeyError: If the `kpt_shape` key is not present in the dataset.
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data
    
    
    def one_cycle(self, y1=0.0, y2=1.0, steps=100):
        """
        Return a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf.

        Args:
            y1 (float, optional): Initial value.
            y2 (float, optional): Final value.
            steps (int, optional): Number of steps.

        Returns:
            (function): Lambda function for computing the sinusoidal ramp.
        """
        return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1
    
    def build_optimizer(self, model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5, **kwargs):
        # instead it has betas = (beta1, beta2)
        return optim.Adam(
            model.parameters(),
            lr=self.args.lr0,                # learning rate
            betas=(0.9, 0.999),   # (beta1, beta2) similar to momentum in spirit
            eps=1e-5,             # numerical stability
            weight_decay=self.args.weight_decay,  # L2 regularization
            **kwargs
        )
        
    def _setup_scheduler(self):
        """Initialize training learning rate scheduler (override)."""
        if getattr(self.args, "scheduler_type", None) == "StepLR":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        elif getattr(self.args, "scheduler_type", None) == "CosineAnnealingLR":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        elif getattr(self.args, "scheduler_type", None) == "ReduceLROnPlateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=5)

        else:
            # fallback to base behavior (cosine or linear with LambdaLR)
            if self.args.cos_lr:
                self.lf = self.one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
            else:
                self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)