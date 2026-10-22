

from collections import Counter

from pathlib import Path

import csv

import json

import argparse

import cv2

import numpy as np



from tracker import ObjectTracker





BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"

RESULTS_DIR = BASE_DIR / "results"

ROI_DIR = RESULTS_DIR / "roi"

DEBUG_DIR = RESULTS_DIR / "debug"



VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}



# Кадры, на которых сохраняем визуализацию соответствий.

DEBUG_FRAMES = {100, 200, 300, 400}



# Максимальное количество проблемных кадров на видео.

MAX_FAILURE_IMAGES = 5



FAILURE_REASONS = [

    "detected",

    "no_descriptors",

    "not_enough_matches",

    "homography_failed",

    "not_enough_inliers",

    "invalid_geometry",

]


def order_corners(points):
    """Упорядочивает углы: левый верхний, правый верхний, правый нижний, левый нижний."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Нужно выбрать ровно четыре точки")

    # Центр и полярный угол позволяют принимать точки в произвольном порядке.
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = np.argmin(pts[:, 0] + pts[:, 1])
    pts = np.roll(pts, -start, axis=0)

    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        raise ValueError("Углы должны образовывать выпуклый четырёхугольник")
    if abs(cv2.contourArea(pts)) < 100:
        raise ValueError("Выбранная область слишком мала")
    return pts


def extract_reference(frame, points):
    """Выпрямляет четырёхугольник в прямоугольное изображение эталона."""
    pts = order_corners(points)
    tl, tr, br, bl = pts
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 16 or height < 16:
        raise ValueError("Эталон слишком маленький")
    destination = np.float32([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1],
    ])
    matrix = cv2.getPerspectiveTransform(pts, destination)
    return cv2.warpPerspective(frame, matrix, (width, height))


def select_four_corners(frame, video_name):
    """Интерактивно выбирает четыре угла, возвращает точки либо None."""
    window = f"Select 4 corners: {video_name}"
    points = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    print("Нажми на четыре угла объекта. ENTER — подтвердить, R — сбросить, ESC — отменить.")
    try:
        while True:
            preview = frame.copy()
            for i, (x, y) in enumerate(points):
                cv2.circle(preview, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(preview, str(i + 1), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(points) == 4:
                try:
                    corners = order_corners(points)
                    cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 0), 2)
                except ValueError:
                    pass
            cv2.imshow(window, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (13, 10, 32) and len(points) == 4:
                try:
                    extract_reference(frame, points)
                    return points
                except ValueError as exc:
                    print(f"Некорректный выбор: {exc}. Нажми R и попробуй снова.")
    finally:
        cv2.destroyWindow(window)


def get_reference_roi(video_path, first_frame, reselect=False):
    """Выбирает кадр и 4 угла. Возвращает выпрямленное изображение эталона или None."""
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    roi_path = ROI_DIR / f"{video_path.stem}_roi.json"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError("Не удалось определить количество кадров")

        def read_frame(index):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            return frame if ok else None

        if roi_path.exists() and not reselect:
            try:
                data = json.loads(roi_path.read_text(encoding="utf-8"))
                index = int(data.get("frame_idx", 0))
                if not 0 <= index < frame_count or "points" not in data:
                    raise ValueError("Сохранённый формат устарел или номер кадра некорректен")
                frame = read_frame(index)
                if frame is None:
                    raise ValueError("Не удалось прочитать эталонный кадр")
                h, w = frame.shape[:2]
                points = np.asarray(data["points"], dtype=np.float32)
                if points.shape != (4, 2) or not np.isfinite(points).all():
                    raise ValueError("Некорректные координаты")
                if (points[:, 0] < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] < 0).any() or (points[:, 1] >= h).any():
                    raise ValueError("Точки за пределами кадра")
                reference = extract_reference(frame, points)
                print(f"Загружен эталон: {roi_path}, кадр {index}")
                return reference
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"Нужно выбрать новый эталон: {exc}")

        window = f"Select reference frame: {video_path.name}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        selected_idx = 0
        selected_frame = first_frame.copy()

        def on_trackbar(value):
            nonlocal selected_idx, selected_frame
            frame = read_frame(value)
            if frame is not None:
                selected_idx = value
                selected_frame = frame
                preview = frame.copy()
                cv2.putText(preview, f"Frame {value}/{frame_count - 1} | ENTER: select | ESC: cancel",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(window, preview)

        try:
            cv2.createTrackbar("Frame", window, 0, frame_count - 1, on_trackbar)
            on_trackbar(0)
            print("Выбери кадр ползунком и нажми ENTER.")
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (13, 10, 32):
                    break
                if key == 27:
                    return None
        finally:
            cv2.destroyWindow(window)

        points = select_four_corners(selected_frame, video_path.name)
        if points is None:
            return None
        reference = extract_reference(selected_frame, points)
        data = {
            "frame_idx": selected_idx,
            "points": [[int(x), int(y)] for x, y in points],
            "frame_width": int(selected_frame.shape[1]),
            "frame_height": int(selected_frame.shape[0]),
        }
        roi_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"Эталон сохранён: {roi_path}; кадр {selected_idx}; размер {reference.shape[1]}x{reference.shape[0]}")
        return reference
    finally:
        cap.release()


def save_debug_image(tracker, frame, video_name, frame_number, suffix):

    """Сохраняет визуализацию соответствий SIFT."""



    output_dir = DEBUG_DIR / video_name

    output_dir.mkdir(parents=True, exist_ok=True)



    visualization = tracker.draw_matches(

        frame,

        max_matches=80,

    )



    output_path = (

        output_dir / f"frame_{frame_number:04d}_{suffix}.jpg"

    )



    success = cv2.imwrite(

        str(output_path),

        visualization,

    )



    if not success:

        print(f"Не удалось сохранить: {output_path}")



    return output_path





def process_video(video_path, reselect_roi=False):

    """Обрабатывает видео и возвращает статистику."""



    print(f"\nОбрабатываем: {video_path.name}")



    cap = cv2.VideoCapture(str(video_path))



    if not cap.isOpened():

        print(f"Не удалось открыть видео: {video_path}")

        return None



    success, first_frame = cap.read()



    if not success:

        cap.release()

        print("Не удалось прочитать первый кадр")

        return None



    reference = get_reference_roi(
        video_path, first_frame, reselect=reselect_roi
    )

    if reference is None:
        cap.release()
        print("Объект не выбран, видео пропущено")
        return None

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    tracker = ObjectTracker(reference_gray)

    # Эталон может быть взят из любого кадра, но обработка начинается с нулевого.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)



    fps = cap.get(cv2.CAP_PROP_FPS)



    if fps <= 0 or not np.isfinite(fps):

        fps = 30.0



    height, width = first_frame.shape[:2]



    RESULTS_DIR.mkdir(parents=True, exist_ok=True)



    output_path = (

        RESULTS_DIR / f"{video_path.stem}_tracked.mp4"

    )



    writer = cv2.VideoWriter(

        str(output_path),

        cv2.VideoWriter_fourcc(*"mp4v"),

        fps,

        (width, height),

    )



    if not writer.isOpened():

        cap.release()

        raise RuntimeError(

            f"Не удалось создать видео: {output_path}"

        )



    frame_count = 0

    detected_count = 0

    failure_images_saved = 0



    good_matches_history = []

    inliers_history = []

    inlier_ratio_history = []



    failure_stats = Counter()
    method_stats = Counter()
    sift_inliers_history = []
    flow_inliers_history = []
    sift_ratio_history = []
    flow_ratio_history = []



    try:

        while True:

            success, frame = cap.read()



            if not success:

                break



            frame_count += 1



            # Запускаем трекинг.

            corners = tracker.track(frame)

            if (
                video_path.stem == "simple_track"
                and corners is None
                and tracker.last_failure_reason == "invalid_geometry"
            ):
                print(
                    f"Кадр {frame_count}: "
                    f"{tracker.last_geometry_failure}"
                )

            reason = tracker.last_failure_reason
            method = tracker.last_method
            if corners is not None:
                if method not in ("sift", "flow"):
                    raise RuntimeError(f"Неизвестный метод: {method}")
                method_stats[method] += 1
                if method == "sift":
                    sift_inliers_history.append(tracker.last_inliers)
                    sift_ratio_history.append(tracker.last_inlier_ratio)
                else:
                    flow_inliers_history.append(tracker.last_inliers)
                    flow_ratio_history.append(tracker.last_inlier_ratio)



            if reason not in FAILURE_REASONS:

                raise RuntimeError(

                    f"Неизвестный результат трекинга: {reason}"

                )



            failure_stats[reason] += 1



            good_matches_history.append(

                tracker.last_good_matches

            )



            inliers_history.append(

                tracker.last_inliers

            )



            inlier_ratio_history.append(

                tracker.last_inlier_ratio

            )



            # Сохраняем соответствия на выбранных кадрах.

            if frame_count in DEBUG_FRAMES:

                save_debug_image(

                    tracker,

                    frame,

                    video_path.stem,

                    frame_count,

                    reason,

                )



            # Сохраняем примеры проблем с RANSAC.

            if (

                reason == "not_enough_inliers"

                and failure_images_saved < MAX_FAILURE_IMAGES

            ):

                save_debug_image(

                    tracker,

                    frame,

                    video_path.stem,

                    frame_count,

                    "few_inliers",

                )



                failure_images_saved += 1



            if corners is not None:

                detected_count += 1



                # Рисуем рамку вокруг объекта.

                cv2.polylines(

                    frame,

                    [corners],

                    isClosed=True,

                    color=(0, 255, 0),

                    thickness=3,

                )



                label = f"Tracked object ({method.upper() if method == 'sift' else 'LK'})"



                (text_width, text_height), _ = cv2.getTextSize(

                    label,

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    2,

                )



                x_label = int(corners[:, 0, 0].min())

                y_label = int(corners[:, 0, 1].min()) - 10



                x_label = max(

                    0,

                    min(

                        x_label,

                        max(0, width - text_width - 1),

                    ),

                )



                y_label = max(

                    text_height + 5,

                    min(y_label, height - 5),

                )



                cv2.putText(

                    frame,

                    label,

                    (x_label, y_label),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 255, 0),

                    2,

                )



            writer.write(frame)



    finally:

        cap.release()

        writer.release()



    # Проверяем согласованность статистики.

    assert sum(failure_stats.values()) == frame_count



    assert failure_stats["detected"] == detected_count
    assert sum(method_stats.values()) == detected_count



    detection_rate = (

        detected_count / frame_count * 100

        if frame_count else 0.0

    )



    result = {

        "video": video_path.name,

        "frames": frame_count,

        "detected": detected_count,

        "detection_rate": detection_rate,

        "good_matches": (

            float(np.mean(good_matches_history))

            if good_matches_history else 0.0

        ),

        "inliers": (

            float(np.mean(inliers_history))

            if inliers_history else 0.0

        ),

        "inlier_ratio": (

            float(np.mean(inlier_ratio_history))

            if inlier_ratio_history else 0.0

        ),

        "failure_stats": dict(failure_stats),
        "sift_detected": method_stats["sift"],
        "flow_detected": method_stats["flow"],
        "sift_inliers": float(np.mean(sift_inliers_history)) if sift_inliers_history else 0.0,
        "flow_inliers": float(np.mean(flow_inliers_history)) if flow_inliers_history else 0.0,
        "sift_inlier_ratio": float(np.mean(sift_ratio_history)) if sift_ratio_history else 0.0,
        "flow_inlier_ratio": float(np.mean(flow_ratio_history)) if flow_ratio_history else 0.0,

    }



    print(f"Обработано кадров: {frame_count}")

    print(f"Объект найден: {detected_count}")

    print(f"Доля обнаружений: {detection_rate:.1f}%")
    print(f"Обнаружено SIFT: {method_stats['sift']}")
    print(f"Обнаружено Lucas-Kanade: {method_stats['flow']}")



    print(

        f"Среднее good matches: "

        f"{result['good_matches']:.1f}"

    )



    print(

        f"Среднее inliers (SIFT и LK вместе): "

        f"{result['inliers']:.1f}"

    )



    print(

        f"Средняя доля inliers: "

        f"{result['inlier_ratio']:.1%}"

    )



    print(f"Среднее SIFT inliers при успешном SIFT: {result['sift_inliers']:.1f}")
    print(f"Среднее LK inliers при успешном LK: {result['flow_inliers']:.1f}")
    print("\nПричины результатов трекинга:")



    for reason in FAILURE_REASONS:

        count = failure_stats[reason]



        percentage = (

            count / frame_count * 100

            if frame_count else 0.0

        )



        print(

            f"  {reason:<22} "

            f"{count:>4} кадров "

            f"({percentage:>5.1f}%)"

        )



    print(f"\nРезультат: {output_path}")

    print(f"Диагностика: {DEBUG_DIR / video_path.stem}")



    return result





def save_metrics(results):

    """Сохраняет статистику всех видео в CSV."""



    RESULTS_DIR.mkdir(parents=True, exist_ok=True)



    csv_path = RESULTS_DIR / "tracking_metrics.csv"



    fieldnames = [

        "video",

        "frames",

        "detected",

        "detection_rate",

        "good_matches",

        "inliers",

        "inlier_ratio",

        *FAILURE_REASONS,

    ]



    with open(

        csv_path,

        "w",

        newline="",

        encoding="utf-8",

    ) as file:



        writer = csv.DictWriter(

            file,

            fieldnames=fieldnames,

        )



        writer.writeheader()



        for result in results:

            row = {

                "video": result["video"],

                "frames": result["frames"],

                "detected": result["detected"],

                "detection_rate": round(

                    result["detection_rate"], 2

                ),

                "good_matches": round(

                    result["good_matches"], 2

                ),

                "inliers": round(

                    result["inliers"], 2

                ),

                "inlier_ratio": round(

                    result["inlier_ratio"], 4

                ),

            }



            for reason in FAILURE_REASONS:

                row[reason] = (

                    result["failure_stats"].get(reason, 0)

                )



            writer.writerow(row)



    print(f"\nСтатистика сохранена: {csv_path}")





def main():

    if not VIDEO_DIR.exists():

        print("Папка videos не найдена")

        return



    video_paths = sorted(

        path

        for path in VIDEO_DIR.iterdir()

        if path.is_file()

        and path.suffix.lower() in VIDEO_EXTENSIONS

    )



    if not video_paths:

        print("В папке videos нет видеофайлов")

        return



    print(f"Найдено видео: {len(video_paths)}")



    results = []



    for video_path in video_paths:

        result = process_video(video_path, reselect_roi=args.reselect_roi)



        if result is not None:

            results.append(result)



    if not results:

        print("Нет успешно обработанных видео")

        return



    print("\n" + "=" * 75)

    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")

    print("=" * 75)



    for result in results:

        print(

            f"{result['video']:<38} "

            f"{result['detected']:>4}/"

            f"{result['frames']:<4} "

            f"{result['detection_rate']:>6.1f}% "
            f"(SIFT: {result['sift_detected']}, LK: {result['flow_detected']})"

        )



    total_frames = sum(

        result["frames"] for result in results

    )



    total_detected = sum(

        result["detected"] for result in results

    )



    overall_rate = (

        total_detected / total_frames * 100

        if total_frames else 0.0

    )



    print("-" * 75)



    print(

        f"ВСЕГО: {total_detected}/{total_frames} "

        f"({overall_rate:.1f}%)"

    )



    print(f"SIFT всего: {sum(r['sift_detected'] for r in results)}")
    print(f"Lucas-Kanade всего: {sum(r['flow_detected'] for r in results)}")
    save_metrics(results)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Трекинг объекта: SIFT + Lucas-Kanade"
    )

    parser.add_argument(
        "--reselect-roi",
        action="store_true",
        help="Выбрать эталон заново",
    )

    args = parser.parse_args()

    main()
