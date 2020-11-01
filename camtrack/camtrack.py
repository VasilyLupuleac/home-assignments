#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import numpy as np
import cv2
from typing import List, Optional, Tuple

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
    rodrigues_and_translation_to_view_mat3x4
)
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose


class CameraTracker:

    MAX_REPR_ERROR = 3.0

    def __init__(self,
                 intrinsic_mat,
                 corners: CornerStorage,
                 known_view_1: Optional[Tuple[int, Pose]] = None,
                 known_view_2: Optional[Tuple[int, Pose]] = None):
        if known_view_1 is None or known_view_2 is None:
            raise NotImplementedError()
        self.intrinsic_mat = intrinsic_mat
        self.corners = corners
        self.frame_count = len(corners)
        self.pc_builder = PointCloudBuilder()
        frame_1 = known_view_1[0]
        frame_2 = known_view_2[0]
        self.is_known = np.full(self.frame_count, False)
        self.is_known[frame_1] = True
        self.is_known[frame_2] = True
        self.view_mats = [pose_to_view_mat3x4(known_view_1[1])] * self.frame_count
        self.view_mats[frame_2] = pose_to_view_mat3x4(known_view_2[1])
        self._extend_point_cloud(frame_1, frame_2)

    def _extend_point_cloud(self, frame_1, frame_2, max_reprojection_error=MAX_REPR_ERROR, min_angle=2.0):
        corners_1 = self.corners[frame_1]
        corners_2 = self.corners[frame_2]
        correspondences = build_correspondences(corners_1, corners_2, self.pc_builder.ids)
        triangulation_parameters = TriangulationParameters(max_reprojection_error,
                                                           min_angle,
                                                           0)
        points3d, ids, _ = triangulate_correspondences(correspondences,
                                                       self.view_mats[frame_1],
                                                       self.view_mats[frame_2],
                                                       self.intrinsic_mat,
                                                       triangulation_parameters)
        self.pc_builder.add_points(ids, points3d)

    def _get_pose(self, frame):
        corners = self.corners[frame]
        points3d = self.pc_builder.points
        _, corners_ids, points_ids = np.intersect1d(corners.ids,
                                                    self.pc_builder.ids,
                                                    assume_unique=True,
                                                    return_indices=True)
        points3d = points3d[points_ids]
        corners_points = corners.points[corners_ids]
        if len(points3d) < 4:
            return

        success, R, t, inliers = cv2.solvePnPRansac(objectPoints=points3d,
                                                    imagePoints=corners_points,
                                                    cameraMatrix=self.intrinsic_mat,
                                                    reprojectionError=self.MAX_REPR_ERROR,
                                                    confidence=0.999,
                                                    distCoeffs=None,
                                                    flags=cv2.SOLVEPNP_EPNP)

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

        self.view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(R, t)
        return R, t

    def track(self):
        while not self.is_known.all():
            unknown_frames = np.arange(self.frame_count)[self.is_known == False]
            update_results = [(frame, self._get_pose(frame)) for frame in unknown_frames]
            updated = [frame for frame, res in update_results if res is not None]
            if len(updated) == 0:
                break
            self.is_known[updated] = True

            old_frames = np.arange(self.frame_count)[self.is_known]
            for frame_1 in updated:
                for old_frame in old_frames:
                    self._extend_point_cloud(frame_1, old_frame)
                for new_frame in updated:
                    if new_frame < frame_1:
                        self._extend_point_cloud(frame_1, new_frame)

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
