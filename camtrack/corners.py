#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli, filter_frame_corners


def to_uint8(img):
    img = 255.0 * img
    return img.astype(np.uint8)


class CornerTracker:
    MAX_CORNERS = 1000
    QUALITY_LEVEL = 0.01
    MIN_DISTANCE = 2
    WIN_SIZE = (15, 15)
    BLOCK_SIZE = 10
    PYR_ITERS = 2
    TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)

    def __init__(self, img):
        self.next_id = 0
        self.frame_corners = FrameCorners(ids=np.empty((0,), dtype=np.int64),
                                          points=np.empty((0, 2), dtype=np.float32),
                                          sizes=np.empty((0,), dtype=np.int))
        self.prev_img = img
        self.frame_corners = FrameCorners(*self._detect_new_corners(img))

    def _get_circles_mask(self, points=np.empty((0, 2)), radii=np.empty((0, 1))):
        mask = np.full(self.prev_img.shape, 255).astype(np.uint8)
        points = np.concatenate((self.frame_corners.points, points), axis=0)
        radii = np.concatenate((self.frame_corners.sizes, radii))
        for (x, y), radius in zip(points, radii):
            cv2.circle(img=mask,
                       center=(np.round(x).astype(np.int),
                               np.round(y).astype(np.int)),
                       radius=int(radius),
                       thickness=-1,
                       color=0)
        return mask

    def _select_distributed_points(self, points):
        taken = np.empty((0, 2))
        for i, point in enumerate(points):
            if taken.shape[0] > self.MAX_CORNERS - self.frame_corners.ids.shape[0]:
                break
            if np.all(np.linalg.norm(taken - point, axis=1) >= self.MIN_DISTANCE):
                taken = np.append(taken, point.reshape(1, 2), axis=0)
        return taken

    def _detect_new_corners(self, img):
        feature_params = dict(qualityLevel=self.QUALITY_LEVEL,
                              blockSize=self.BLOCK_SIZE,
                              minDistance=self.MIN_DISTANCE,
                              useHarrisDetector=False)

        pyr_img = img.copy()
        coeff = 1
        points = np.empty((0, 2))
        sizes = np.empty((0,), dtype=np.int)
        mask = self._get_circles_mask()
        for _ in range(self.PYR_ITERS):
            remaining_corners = self.MAX_CORNERS - self.frame_corners.points.shape[0]
            if remaining_corners <= 0:
                break
            size = self.BLOCK_SIZE * coeff

            corners_found = cv2.goodFeaturesToTrack(image=pyr_img,
                                                    mask=mask[::coeff, ::coeff],
                                                    maxCorners=remaining_corners,
                                                    **feature_params)

            if corners_found is None:
                continue
            points_found = corners_found.reshape(-1, 2).astype(np.float32) * coeff
            new_points = self._select_distributed_points(points_found)
            new_sizes = np.full(new_points.shape[0], size, dtype=np.int)
            mask = self._get_circles_mask(new_points, new_sizes.reshape(-1, 1))
            new_sizes = np.full(new_points.shape[0], size, dtype=np.int)
            sizes = np.concatenate((sizes, new_sizes), axis=0)
            points = np.concatenate((points, new_points), axis=0)

            pyr_img = cv2.pyrDown(pyr_img)
            coeff *= 2

        ids = np.arange(self.next_id, self.next_id + points.shape[0], dtype=np.int64)
        self.next_id += points.shape[0]
        return ids, points, sizes

    def detect_and_track(self, new_img=None):
        if new_img is None:
            return self.frame_corners

        lk_params = dict(
            winSize=self.WIN_SIZE,
            maxLevel=self.PYR_ITERS,
            criteria=self.TERM_CRITERIA)
        new_points, st, _ = cv2.calcOpticalFlowPyrLK(to_uint8(self.prev_img),
                                                     to_uint8(new_img),
                                                     self.frame_corners.points.astype(np.float32).reshape((-1, 1, 2)),
                                                     None,
                                                     **lk_params)
        st = st.ravel()
        tracked = filter_frame_corners(self.frame_corners, st == 1)

        new_tracked = FrameCorners(tracked.ids, new_points[st == 1], tracked.sizes)
        self.frame_corners = new_tracked
        new_ids, new_points, new_sizes = self._detect_new_corners(new_img)

        ids = np.concatenate((self.frame_corners.ids.ravel(), new_ids), axis=0)
        points = np.concatenate((self.frame_corners.points, new_points), axis=0)
        sizes = np.concatenate((self.frame_corners.sizes.ravel(), new_sizes), axis=0)
        self.frame_corners = FrameCorners(ids, points, sizes)
        self.prev_img = new_img
        return tracked


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    corner_tracker = CornerTracker(frame_sequence[0])

    for ind, img in enumerate(frame_sequence[1:]):
        corners = corner_tracker.detect_and_track(img)
        builder.set_corners_at_frame(ind, corners)

    last_corners = corner_tracker.detect_and_track()
    builder.set_corners_at_frame(len(frame_sequence) - 1, last_corners)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.
    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)

    corner_storage = builder.build_corner_storage()
    final_storage = without_short_tracks(corner_storage, min_len=20)

    return final_storage


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
