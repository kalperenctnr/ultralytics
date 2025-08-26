# pose_metric.py
import numpy as np
from scipy.spatial import cKDTree
import torch
from scipy.spatial.distance import cdist

class PoseMetric:
    def __init__(self, model_points: np.ndarray, diameter: float, threshold: float):
        """
        model_points: (B, N, 3)
        diameter: float
        """
        self.model_points = model_points
        self.diameter = diameter
        self.threshold = threshold
        self.add_scores = []
        self.adds_scores = []

    def compute_add(self, R_pred, t_pred, R_gt, t_gt):
        """
        Computes ADD (Average Distance of points) for a single input.

        Args:
            R_pred (torch.Tensor): Predicted rotation matrix (3, 3).
            t_pred (torch.Tensor): Predicted translation vector (3,).
            R_gt (torch.Tensor): Ground truth rotation matrix (3, 3).
            t_gt (torch.Tensor): Ground truth translation vector (3,).
            model_points (torch.Tensor): The model's point cloud (N, 3).

        Returns:
            torch.Tensor: The ADD score as a scalar tensor.
        """
        # Transform the model points using the predicted and ground truth poses.
        # The @ operator performs matrix multiplication.
        # We use .T for transpose, which works for 2D tensors.
        pts_pred = (R_pred @ self.model_points.T).T + t_pred
        pts_gt = (R_gt @ self.model_points.T).T + t_gt

        # Calculate the mean L2 distance between the transformed point clouds.
        # torch.linalg.norm computes the Euclidean distance along the specified dimension.
        return torch.linalg.norm(pts_pred - pts_gt, dim=1).mean()

    def compute_adds(self, R_pred, t_pred, R_gt, t_gt):
        """
        Computes ADD-S (Average Distance of Nearest Surface Points) for a single input.
        This metric is used for symmetric objects where different poses might be
        indistinguishable.

        Args:
            R_pred (torch.Tensor): Predicted rotation matrix (3, 3).
            t_pred (torch.Tensor): Predicted translation vector (3,).
            R_gt (torch.Tensor): Ground truth rotation matrix (3, 3).
            t_gt (torch.Tensor): Ground truth translation vector (3,).
            model_points (torch.Tensor): The model's point cloud (N, 3).

        Returns:
            torch.Tensor: The ADD-S score as a scalar tensor.
        """
        # Ensure float precision
        model_points = self.model_points.to(R_pred.dtype)

        # Transform model points
        pts_pred = (R_pred @ model_points.T).T + t_pred  # (N, 3)
        pts_gt = (R_gt @ model_points.T).T + t_gt        # (N, 3)

        # Pairwise distances between predicted and GT points
        dists = torch.cdist(pts_pred, pts_gt, p=2)  # (N, N)

        # For each predicted point, take nearest GT point
        nearest_dists, _ = dists.min(dim=1)

        return nearest_dists.mean()


    def update(self, R_pred, t_pred, R_gt, t_gt, iou, symmetric=False):
        """
        Updates scores by matching each ground truth to the best prediction based on IoU.

        For each ground truth object, this method finds the prediction with the highest
        IoU. If this IoU value is above a given threshold (self.threshold), it then 
        computes the ADD or ADD-S score for that specific ground truth/prediction pair.

        Args:
            R_pred (torch.Tensor): Predicted rotation matrices, shape (M, 3, 3).
            t_pred (torch.Tensor): Predicted translation vectors, shape (M, 3).
            R_gt (torch.Tensor): Ground truth rotation matrices, shape (N, 3, 3).
            t_gt (torch.Tensor): Ground truth translation vectors, shape (N, 3).
            iou (torch.Tensor): An IoU matrix of shape (N, M), where N is the number of
                                ground truths and M is the number of predictions.
            symmetric (bool): Flag to indicate if the object is symmetric, determining
                            whether to use ADD-S or ADD score.
        """
        # Get the number of ground truth (N) and predicted (M) objects
        num_gt, num_pred = iou.shape

        # If there are no predictions, there's nothing to match.
        if num_pred == 0:
            return

        # Iterate through each ground truth object
        for i in range(num_gt):
            # Find the index of the prediction with the highest IoU for the current GT
            best_pred_idx = torch.argmax(iou[i])
            
            # Check if the highest IoU score for this GT is above the threshold
            if iou[i, best_pred_idx] > self.threshold:
                # We have a valid match, so we select the corresponding poses.
                R_p = R_pred[best_pred_idx]
                t_p = t_pred[best_pred_idx]
                R_g = R_gt[i]
                t_g = t_gt[i]

                # Compute the appropriate score based on whether the object is symmetric
                if symmetric:
                    score = self.compute_adds(R_p, t_p, R_g, t_g)
                    # The result is a scalar tensor, so we extract the value
                    self.adds_scores.append(score.item())
                else:
                    score = self.compute_add(R_p, t_p, R_g, t_g)
                    self.add_scores.append(score.item())
                    
        return self.add_scores

    def mean_results(self):
        add = np.mean(self.add_scores) if self.add_scores else torch.nan
        adds = np.mean(self.adds_scores) if self.adds_scores else torch.nan
        return {"ADD": add, "ADD-S": adds}

    def accuracy(self, threshold_ratio=0.1, symmetric=False):
        threshold = self.diameter * threshold_ratio
        scores = self.adds_scores if symmetric else self.add_scores
        if not scores:
            return 0.0
        scores = np.array(scores)
        return np.mean(scores < threshold)
