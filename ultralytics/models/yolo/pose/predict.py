# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
import torch

class PosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a pose model.

    This class specializes in pose estimation, handling keypoints detection alongside standard object detection
    capabilities inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO pose model with keypoint detection capabilities.

    Methods:
        construct_result: Construct the result object from the prediction, including keypoints.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize PosePredictor for pose estimation tasks.

        Sets up a PosePredictor instance, configuring it for pose detection tasks and handling device-specific
        warnings for Apple MPS.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.

        Examples:
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.pose import PosePredictor
            >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
            >>> predictor = PosePredictor(overrides=args)
            >>> predictor.predict_cli()
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct the result object from the prediction, including keypoints.

        Extends the parent class implementation by extracting keypoint data from predictions and adding them to the
        result object.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints with shape (N, 6+K*D) where N is
                the number of detections, K is the number of keypoints, and D is the keypoint dimension.
            img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
            orig_img (np.ndarray): The original unprocessed image as a numpy array.
            img_path (str): The path to the original image file.

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and
                keypoints.
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        pred_kpts = pred[:, 6:33].view(len(pred), *self.model.kpt_shape)
        pred_rot = pred[:, 33:42].view(len(pred), 3, 3)
        pred_depth = pred[:, 42:43].view(len(pred), 1)
        pred_translation = self.obtain_translation_vector(pred_kpts, pred_depth, self.model.K)
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        result.update(rotation_matrix=pred_rot)
        result.update(translation_vector=pred_translation)
        return result
    
    @staticmethod
    def obtain_translation_vector(pred_kpts: torch.Tensor, tz: torch.Tensor, K: list) -> torch.Tensor:
        """
        Obtain the translation vector from the predicted keypoints.

        Args:
            pred_kpts (torch.Tensor): Predicted keypoints of shape (N, K, 2 or 3).
            tz (torch.Tensor): Depth (z translation) of shape (N, 1).
            K (list or tuple): Camera intrinsics [cx, cy, fx, fy].

        Returns:
            torch.Tensor: Translation vector of shape (N, 3).
        """
        cx, cy, fx, fy = K

        # Assume the first keypoint is the object center in image coords
        center_kpt = pred_kpts[:, 0, :2]                      # (N, 2)
        obj_center_h = torch.cat([center_kpt, torch.ones_like(center_kpt[:, :1])], dim=-1)  # (N, 3)

        # Normalize by intrinsics
        t = torch.empty_like(obj_center_h)
        t[:, 0] = (obj_center_h[:, 0] - cx) / fx
        t[:, 1] = (obj_center_h[:, 1] - cy) / fy
        t[:, 2] = 1.0

        # Scale by depth
        t = tz * t  # (N, 3)
        return t

