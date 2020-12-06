#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    remove_correspondences_with_ids,
    eye3x4
)
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose


class CameraTracker:
    MAX_REPR_ERROR = 5.0
    MIN_ANGLE = 1.1
    RANSAC_PROB = 0.99999
    THRESHOLD = 0.2

    SEED = 1642832

    def __init__(self,
                 intrinsic_mat,
                 corners: CornerStorage,
                 known_view_1: Optional[Tuple[int, Pose]] = None,
                 known_view_2: Optional[Tuple[int, Pose]] = None):
        self.intrinsic_mat = intrinsic_mat
        self.corners = corners
        self.frame_count = len(corners)
        print("Found {} frames".format(self.frame_count))
        self.pc_builder = PointCloudBuilder()
        if known_view_1 is None or known_view_2 is None:
            frame_1, frame_2, view_mat_2 = self._find_initial_view_mats()
            view_mat_1 = eye3x4()
        else:
            frame_1 = known_view_1[0]
            frame_2 = known_view_2[0]
            view_mat_1 = pose_to_view_mat3x4(known_view_2[1])
            view_mat_2 = pose_to_view_mat3x4(known_view_2[1])

        self.is_known = np.full(self.frame_count, False)
        self.is_known[frame_1] = True
        self.is_known[frame_2] = True
        self.view_mats = [view_mat_1] * self.frame_count
        self.view_mats[frame_2] = view_mat_2
        self._extend_point_cloud(frame_1, frame_2, max_reprojection_error=10)

    def _generate_frame_pairs(self):
        np.random.seed(self.SEED)
        n = self.frame_count

        def neighb(x, rad=(n // 4 + 5)):
            range_1 = np.arange(max(0, x - rad), x - 4)
            x_1 = np.full_like(range_1, x)
            range_2 = np.arange(x + 5, min(n, x + rad))
            x_2 = np.full_like(range_2, x)
            return np.column_stack((range_1, x_1)), np.column_stack((range_2, x_2))

        rand_pairs = np.random.choice(n, (n // 2, 2), replace=False)
        a = np.random.choice(n, replace=False)
        b = (a + n // 2) % n
        return np.concatenate((rand_pairs,
                               *neighb(a),
                               *neighb(b)),
                               axis=0)

    def _find_initial_view_mats(self):
        max_points_count = 0
        best_frame_1, best_frame_2 = 0, 1
        best_pose = None
        for frame_1, frame_2 in self._generate_frame_pairs():
            print('Initializing: trying frames {} and {}'.format(frame_1,
                                                                 frame_2))
            pose, points_count = self._find_pose(frame_1, frame_2)
            if points_count > max_points_count:
                max_points_count = points_count
                best_pose = pose
                best_frame_1, best_frame_2 = frame_1, frame_2

        print('Initialization complete. Selected frames: {} and {}'.format(best_frame_1,
                                                                           best_frame_2))
        return best_frame_1, best_frame_2, pose_to_view_mat3x4(best_pose)

    def _find_pose(self, frame_1, frame_2):
        corners_1 = self.corners[frame_1]
        corners_2 = self.corners[frame_2]
        correspondences = build_correspondences(corners_1, corners_2)

        if len(correspondences.ids) < 5:
            print('Not enough correspondences, aborting')
            return None, 0
        mat, mask = cv2.findEssentialMat(correspondences.points_1,
                                         correspondences.points_2,
                                         self.intrinsic_mat,
                                         cv2.RANSAC,
                                         self.RANSAC_PROB,
                                         self.THRESHOLD)
        correspondences = remove_correspondences_with_ids(correspondences,
                                                          correspondences.ids[mask.flatten() == 0])
        R1, R2, t = cv2.decomposeEssentialMat(mat)
        best_pose = None
        max_points_count = 0
        for R in (R1.T, R2.T):
            for tr in (t, -t):
                pose = Pose(R, R @ tr)
                points, _, _ = triangulate_correspondences(correspondences,
                                                           eye3x4(),
                                                           pose_to_view_mat3x4(pose),
                                                           self.intrinsic_mat,
                                                           TriangulationParameters(self.MAX_REPR_ERROR,
                                                                                   self.MIN_ANGLE,
                                                                                   min_depth=0))
                if len(points) > max_points_count:
                    best_pose = pose
                    max_points_count = len(points)
        print('Found {} points'.format(max_points_count))
        return best_pose, max_points_count

    def _extend_point_cloud(self, frame_1, frame_2, max_reprojection_error=MAX_REPR_ERROR, min_angle=MIN_ANGLE):
        print("Calculating new 3D points using frames {} and {}".format(frame_1 + 1, frame_2 + 1))
        corners_1 = self.corners[frame_1]
        corners_2 = self.corners[frame_2]
        correspondences = build_correspondences(corners_1, corners_2, self.pc_builder.ids)
        if len(correspondences.ids) == 0:
            print("No correspondences found")
            return

        print("Found {} correspondences".format(len(correspondences.ids)))

        triangulation_parameters = TriangulationParameters(max_reprojection_error,
                                                           min_angle,
                                                           0)

        points3d, ids, _ = triangulate_correspondences(correspondences,
                                                       self.view_mats[frame_1],
                                                       self.view_mats[frame_2],
                                                       self.intrinsic_mat,
                                                       triangulation_parameters)
        print("Triangulation successful, found {} new points".format(len(ids)))
        print("Point cloud currently contains {} 3D points".format(len(self.pc_builder.ids)))

        self.pc_builder.add_points(ids, points3d)

    def _calc_camera_position(self, frame):
        print("Processing frame #{}".format(frame + 1))
        corners = self.corners[frame]
        points3d = self.pc_builder.points
        _, corners_ids, points_ids = np.intersect1d(corners.ids,
                                                    self.pc_builder.ids,
                                                    assume_unique=True,
                                                    return_indices=True)

        print("Found {} points".format(len(corners_ids)))
        if len(points_ids) < 4:
            print("Not enough points to calculate position\n")
            return

        points3d = points3d[points_ids]
        corners_points = corners.points[corners_ids]

        success, R, t, inliers = cv2.solvePnPRansac(objectPoints=points3d,
                                                    imagePoints=corners_points,
                                                    cameraMatrix=self.intrinsic_mat,
                                                    reprojectionError=self.MAX_REPR_ERROR,
                                                    confidence=self.RANSAC_PROB,
                                                    distCoeffs=None,
                                                    flags=cv2.SOLVEPNP_EPNP)

        if not success:
            print("Unable to solve PnP with RANSAC\n")
            return
        else:
            print("Found {} inliers".format(len(inliers)))
        if len(inliers) < 4:
            print("Not enough inliers")
            return
        points3d = points3d[inliers]
        corners_points = corners_points[inliers]

        success, R, t = cv2.solvePnP(objectPoints=points3d,
                                     imagePoints=corners_points,
                                     cameraMatrix=self.intrinsic_mat,
                                     distCoeffs=None,
                                     useExtrinsicGuess=True,
                                     rvec=R,
                                     tvec=t,
                                     flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            print("Unable to solve PnP\n")
            return
        else:
            print("Position calculated successfully\n".format(len(inliers)))

        self.view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(R, t)
        return R, t

    def track(self):
        while not self.is_known.all():
            unknown_frames = np.arange(self.frame_count)[self.is_known == False]
            is_updated = False
            for frame in unknown_frames:
                if self._calc_camera_position(frame) is not None:
                    is_updated = True
                    updated_frame = frame
                    break

            if not is_updated:
                print("Unable to calculate remaining camera positions")
                break

            old_frames = np.arange(self.frame_count)[self.is_known]
            self.is_known[updated_frame] = True

            for old_frame in old_frames:
                self._extend_point_cloud(updated_frame, old_frame)
        
        return self.view_mats, self.pc_builder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats, point_cloud_builder = CameraTracker(intrinsic_mat,
                                                   corner_storage,
                                                   known_view_1,
                                                   known_view_2).track()

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )

    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
