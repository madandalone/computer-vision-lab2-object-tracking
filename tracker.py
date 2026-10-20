
import cv2
import numpy as np


class ObjectTracker:
    def __init__(self, reference_image):
        """
        reference_image: эталонное изображение
        в оттенках серого.
        """

        self.reference_image = reference_image

        height, width = reference_image.shape

        self.reference_corners = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]).reshape(-1, 1, 2)

        # SIFT и сопоставление дескрипторов.
        self.detector = cv2.SIFT_create(nfeatures=2000)

        self.keypoints, self.descriptors = (
            self.detector.detectAndCompute(
                reference_image, None
            )
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_L2,
            crossCheck=False,
        )

        # Параметры SIFT + RANSAC.
        self.ratio_threshold = 0.8
        self.ransac_threshold = 5.0
        self.min_good_matches = 10
        self.min_inliers = 8

        # Параметры оптического потока.
        self.min_flow_inliers = 8

        self.lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        }

        # Состояние предыдущего кадра.
        self.prev_gray = None

        # Соответствующие друг другу точки:
        # эталон и предыдущий кадр.
        self.ref_points = None
        self.prev_points = None

        # Статистика текущего кадра.
        self.last_good_matches = 0
        self.last_inliers = 0
        self.last_inlier_ratio = 0.0
        self.last_failure_reason = None

        # Метод, который обнаружил объект.
        self.last_method = None

        # Данные для визуализации SIFT.
        self.last_keypoints = []
        self.last_matches = []
        self.last_inlier_mask = None

        print(
            f"Найдено ключевых точек: "
            f"{len(self.keypoints)}"
        )

    def track(self, frame):
        """
        Находит объект на текущем кадре.

        Сначала SIFT + RANSAC.
        При неудаче пробуем Lucas-Kanade.

        Возвращает:
            corners или None.
        """

        self.last_good_matches = 0
        self.last_inliers = 0
        self.last_inlier_ratio = 0.0
        self.last_failure_reason = None
        self.last_method = None

        self.last_keypoints = []
        self.last_matches = []
        self.last_inlier_mask = None

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # 1. Пробуем обнаружение через SIFT.
        corners = self._track_sift(frame, gray)

        # 2. Если SIFT не справился,
        # пробуем оптический поток.
        if corners is None:
            corners = self._track_flow(gray, frame.shape)

        if corners is None:
            # Оба метода не сработали.
            # Сбрасываем старые соответствия.
            self.ref_points = None
            self.prev_points = None
        else:
            self.last_failure_reason = "detected"

        # Обновляем предыдущий кадр
        # после всех попыток трекинга.
        self.prev_gray = gray

        return corners

    def _track_sift(self, frame, gray):
        """Поиск объекта через SIFT и RANSAC."""

        keypoints, descriptors = (
            self.detector.detectAndCompute(
                gray, None
            )
        )

        self.last_keypoints = keypoints

        if (
            descriptors is None
            or self.descriptors is None
            or len(descriptors) < 2
        ):
            self.last_failure_reason = "no_descriptors"
            return None

        matches = self.matcher.knnMatch(
            self.descriptors,
            descriptors,
            k=2,
        )

        # Тест Лоу.
        good_matches = [
            pair[0]
            for pair in matches
            if len(pair) == 2
            and pair[0].distance
            < self.ratio_threshold * pair[1].distance
        ]

        self.last_matches = good_matches
        self.last_good_matches = len(good_matches)

        if len(good_matches) < self.min_good_matches:
            self.last_failure_reason = "not_enough_matches"
            return None

        src_points = np.float32([
            self.keypoints[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_points = np.float32([
            keypoints[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            self.ransac_threshold,
        )

        if H is None or mask is None:
            self.last_failure_reason = "homography_failed"
            return None

        inlier_mask = mask.ravel().astype(bool)

        sift_inliers = int(inlier_mask.sum())

        self.last_inlier_mask = inlier_mask
        self.last_inliers = sift_inliers

        self.last_inlier_ratio = (
            sift_inliers / len(good_matches)
        )

        if sift_inliers < self.min_inliers:
            self.last_failure_reason = "not_enough_inliers"
            return None

        projected = cv2.perspectiveTransform(
            self.reference_corners,
            H,
        )

        if not self.valid_corners(
            projected,
            frame.shape,
        ):
            self.last_failure_reason = "invalid_geometry"
            return None

        # Сохраняем согласованные точки.
        # Они понадобятся Lucas-Kanade
        # на следующем кадре.
        self.ref_points = src_points[inlier_mask]
        self.prev_points = dst_points[inlier_mask]

        self.last_method = "sift"

        return projected.astype(np.int32)

    def _track_flow(self, gray, frame_shape):
        """
        Продолжает трекинг через Lucas-Kanade,
        используя предыдущий кадр.
        """

        if (
            self.prev_gray is None
            or self.prev_points is None
            or self.ref_points is None
            or len(self.prev_points) < self.min_flow_inliers
        ):
            return None

        next_points, status, _ = (
            cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_points,
                None,
                **self.lk_params,
            )
        )

        if next_points is None or status is None:
            return None

        status = status.ravel().astype(bool)

        # Исключаем невалидные координаты.
        finite = np.isfinite(
            next_points.reshape(-1, 2)
        ).all(axis=1)

        status &= finite

        if int(status.sum()) < self.min_flow_inliers:
            return None

        src_points = self.ref_points[status]
        dst_points = next_points[status]

        H, mask = cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            self.ransac_threshold,
        )

        if H is None or mask is None:
            return None

        inlier_mask = mask.ravel().astype(bool)
        flow_inliers = int(inlier_mask.sum())

        if flow_inliers < self.min_flow_inliers:
            return None

        projected = cv2.perspectiveTransform(
            self.reference_corners,
            H,
        )

        if not self.valid_corners(
            projected,
            frame_shape,
        ):
            return None

        # Обновляем точки для следующего кадра.
        self.ref_points = src_points[inlier_mask]
        self.prev_points = dst_points[inlier_mask]

        self.last_inliers = flow_inliers
        self.last_inlier_ratio = (
            flow_inliers / len(src_points)
        )

        self.last_method = "flow"

        return projected.astype(np.int32)

    def draw_matches(self, frame, max_matches=80):
        """
        Визуализация соответствий SIFT.

        Зелёные линии: inliers.
        Красные линии: outliers.
        """

        reference_bgr = cv2.cvtColor(
            self.reference_image,
            cv2.COLOR_GRAY2BGR,
        )

        if not self.last_matches:
            return np.hstack([
                cv2.resize(
                    reference_bgr,
                    (frame.shape[1], frame.shape[0]),
                ),
                frame.copy(),
            ])

        selected_indices = np.linspace(
            0,
            len(self.last_matches) - 1,
            min(
                max_matches,
                len(self.last_matches),
            ),
            dtype=int,
        )

        selected_matches = [
            self.last_matches[i]
            for i in selected_indices
        ]

        selected_mask = None

        if self.last_inlier_mask is not None:
            selected_mask = [
                int(self.last_inlier_mask[i])
                for i in selected_indices
            ]

        # Рисуем все соответствия красным.
        visualization = cv2.drawMatches(
            reference_bgr,
            self.keypoints,
            frame,
            self.last_keypoints,
            selected_matches,
            None,
            matchColor=(0, 0, 255),
            flags=(
                cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            ),
        )

        # Поверх рисуем inliers зелёным.
        if selected_mask is not None:
            inlier_matches = [
                match
                for match, is_inlier in zip(
                    selected_matches,
                    selected_mask,
                )
                if is_inlier
            ]

            if inlier_matches:
                visualization = cv2.drawMatches(
                    reference_bgr,
                    self.keypoints,
                    frame,
                    self.last_keypoints,
                    inlier_matches,
                    visualization,
                    matchColor=(0, 255, 0),
                    flags=(
                        cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
                    ),
                )

        return visualization

    @staticmethod
    def valid_corners(corners, frame_shape):
        """
        Проверяет геометрию четырёхугольника.
        """

        height, width = frame_shape[:2]

        points = corners.reshape(4, 2).astype(
            np.float32
        )

        if not np.isfinite(points).all():
            return False

        if not cv2.isContourConvex(
            points.reshape(-1, 1, 2)
        ):
            return False

        area = cv2.contourArea(points)
        frame_area = height * width

        if area < frame_area * 0.001:
            return False

        if area > frame_area * 2.0:
            return False

        margin_x = width * 0.2
        margin_y = height * 0.2

        if (
            np.any(points[:, 0] < -margin_x)
            or np.any(points[:, 0] > width + margin_x)
            or np.any(points[:, 1] < -margin_y)
            or np.any(points[:, 1] > height + margin_y)
        ):
            return False

        return True
