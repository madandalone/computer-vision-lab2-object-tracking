#!/usr/bin/env python3
"""Генерация README.md для ЛР2 по компьютерному зрению.

Запуск из корня проекта: python generate_report.py
Опционально: python generate_report.py --no-images
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REASONS = ["no_descriptors", "not_enough_matches", "homography_failed", "not_enough_inliers", "invalid_geometry"]
REASON_NAMES = {
    "no_descriptors": "Не найдены дескрипторы",
    "not_enough_matches": "Недостаточно соответствий",
    "homography_failed": "Не удалось оценить гомографию",
    "not_enough_inliers": "Недостаточно inliers",
    "invalid_geometry": "Некорректная геометрия рамки",
}


def load_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден CSV со статистикой: {path}. Сначала запусти python main.py")
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Файл {path} пуст")
    for row in rows:
        for key in ("video", "frames", "detected"):
            if key not in row:
                raise ValueError(f"В {path} отсутствует столбец {key}")
        row["frames"] = int(row["frames"])
        row["detected"] = int(row["detected"])
        for key in REASONS:
            row[key] = int(row.get(key) or 0)
    return rows


def load_methods(path: Path) -> dict[str, tuple[int, int]]:
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            result[row["video"]] = (int(row["sift_detected"]), int(row["flow_detected"]))
    return result


def fmt_pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "0.0%"


def extract_examples(rows: list[dict], results_dir: Path, output_dir: Path,
                     selected_frames: dict[str, int] | None = None) -> dict[str, str]:
    """Сохранить кадры по выбранным номерам и листы предпросмотра.

    Если номер не задан, выбирается первый из кандидатов с зелёной рамкой.
    Это эвристика: корректность геометрии проверяется вручную по contact sheets.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Для иллюстраций нужны opencv-python и numpy")
        return {}

    selected_frames = selected_frames or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    for row in rows:
        name = Path(row["video"]).stem
        candidates = [results_dir / f"{name}_tracked.mp4", results_dir / f"{name}_sift_tracked.mp4"]
        video_path = next((p for p in candidates if p.exists()), None)
        if video_path is None:
            print(f"Нет обработанного видео: {row['video']}")
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video_path}")
            continue
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Контактный лист позволяет оценить кадры до экспорта в PDF.
            indices = sorted(set(min(count - 1, max(0, round((count - 1) * r)))
                                 for r in (0.08, 0.16, 0.24, 0.32, 0.40,
                                           0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.96)))
            thumbs = []
            candidates_with_box = []
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    continue
                # В видео зелёная рамка. Маска используется только для наличия рамки,
                # не для проверки того, что рамка правильно обводит объект.
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (35, 100, 110), (90, 255, 255))
                green_pixels = int(cv2.countNonZero(mask))
                if green_pixels > max(80, frame.shape[0] * frame.shape[1] * 0.0007):
                    candidates_with_box.append(index)
                thumb = cv2.resize(frame, (320, 180))
                cv2.rectangle(thumb, (0, 0), (320, 25), (0, 0, 0), -1)
                cv2.putText(thumb, f"Frame {index}", (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                thumbs.append(thumb)
            if thumbs:
                canvas = np.full((180 * 3, 320 * 4, 3), 245, dtype=np.uint8)
                for i, thumb in enumerate(thumbs):
                    canvas[(i // 4) * 180:(i // 4 + 1) * 180,
                           (i % 4) * 320:(i % 4 + 1) * 320] = thumb
                cv2.imwrite(str(output_dir / f"{name}_contact_sheet.jpg"), canvas)

            chosen = selected_frames.get(row["video"])
            if chosen is None:
                # Для первичной версии берём кадр с видимой рамкой ближе к началу.
                # После просмотра contact sheet задаём точный индекс в report_frames.json.
                chosen = candidates_with_box[0] if candidates_with_box else None
            if chosen is None or not 0 <= chosen < count:
                print(f"Для {row['video']} не выбран корректный номер кадра")
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, chosen)
            ok, selected = cap.read()
            if not ok:
                continue
            image_path = output_dir / f"{name}_example.jpg"
            if cv2.imwrite(str(image_path), selected):
                found[row["video"]] = image_path.relative_to(ROOT).as_posix()
                print(f"Иллюстрация {row['video']}: кадр {chosen}")
        finally:
            cap.release()
    return found


def generate_readme(rows: list[dict], methods: dict[str, tuple[int, int]], images: dict[str, str]) -> str:
    total_frames = sum(row["frames"] for row in rows)
    total_detected = sum(row["detected"] for row in rows)
    own = [row for row in rows if Path(row["video"]).stem.upper().startswith("IMG_")]
    methods_complete = bool(methods) and all(
        row["video"] in methods and sum(methods[row["video"]]) == row["detected"] for row in rows
    )
    if methods and not methods_complete:
        print("Внимание: статистика методов не соответствует общему числу обнаружений. Разбивка SIFT/LK не будет показана.")

    lines = [
        "# Лабораторная работа №2. Трекинг плоского объекта на видео",
        "",
        "**Дисциплина:** Компьютерное зрение  ",
        "**Задача:** отслеживание прямоугольного плоского объекта на видеопоследовательности по ключевым точкам.",
        "",
        "Программа позволяет выбрать любой кадр видео и указать четыре угла эталонного объекта. "
        "На следующих кадрах она пытается определить положение объекта, построить четырёхугольную рамку и подписать объект.",
        "",
        "## 1. Теоретическая база",
        "",
        "**Ключевые точки и дескрипторы.** Для поиска объекта применяется SIFT (Scale-Invariant Feature Transform). "
        "Алгоритм выделяет характерные точки изображения и вычисляет для них дескрипторы, "
        "которые можно сопоставлять между эталоном и текущим кадром. SIFT устойчив к изменению масштаба "
        "и повороту в определённых пределах.",
        "",
        "**Сопоставление признаков.** Дескрипторы сравниваются с помощью BFMatcher и евклидова расстояния (L2). "
        "Для каждой точки ищутся два ближайших соседа, затем применяется тест Лоу (ratio test), "
        "уменьшающий число неоднозначных соответствий.",
        "",
        "**Гомография и RANSAC.** Для плоского объекта перспективное преобразование между эталоном "
        "и кадром приближённо описывается матрицей гомографии $H$ размерности $3\\times3$. "
        "Алгоритм RANSAC оценивает $H$ по согласованным соответствиям, отбрасывая выбросы. "
        "Точки, согласующиеся с найденной моделью, называются *inliers*. "
        "Углы эталона преобразуются через $H$, формируя рамку объекта.",
        "",
        "**Оптический поток Лукаса–Канаде.** Если на текущем кадре SIFT не смог построить "
        "допустимую рамку, используется разреженный оптический поток. Он оценивает перемещение "
        "ранее отслеженных точек между соседними кадрами. По сохранившимся соответствиям "
        "между эталоном и текущим кадром повторно вычисляется гомография.",
        "",
        "## 2. Описание разработанной системы",
        "",
        "### Архитектура",
        "",
        "- `main.py`: чтение видео, выбор и кэширование ROI, запуск трекера, рисование рамки, сохранение результатов и метрик.",
        "- `tracker.py`: поиск и сопоставление SIFT-признаков, оценка гомографии, резервный трекинг Lucas–Kanade и проверка геометрии.",
        "- `generate_report.py`: формирование настоящего отчёта по CSV и обработанным видео.",
        "- `videos/`: входные видео, включая самостоятельно снятый ролик.",
        "- `results/roi/`: сохранённые координаты четырёх углов и индекс выбранного кадра.",
        "- `results/debug/`: изображения соответствий для анализа проблемных кадров.",
        "- `results/`: размеченные видео, `tracking_metrics.csv` и при наличии `tracking_methods.csv`.",
        "",
        "### Алгоритм обработки",
        "",
        "1. Пользователь выбирает кадр ползунком и отмечает четыре угла объекта; номер кадра и координаты сохраняются в JSON.",
        "2. Эталон выпрямляется перспективным преобразованием в прямоугольное изображение.",
        "3. На эталоне один раз вычисляются ключевые точки и дескрипторы SIFT.",
        "4. На очередном кадре находятся SIFT-соответствия и оценивается гомография методом RANSAC.",
        "5. Проверяются выпуклость и площадь рамки, допустимость её положения, распределение inliers и ошибка перепроецирования; согласованные точки сохраняются для следующего кадра.",
        "6. Если SIFT не дал допустимой рамки, выполняется Lucas–Kanade по точкам предыдущего кадра, затем повторная оценка гомографии и проверка рамки.",
        "7. Если оба метода не сработали, рамка не рисуется; на следующем кадре выполняется новая попытка обнаружения.",
        "8. Для успешных кадров рисуются четырёхугольник и подпись, а результаты сохраняются в MP4 и CSV.",
        "",
        "### Запуск",
        "",
        "Требуются Python, `opencv-python` и `numpy`. Запуск из корня проекта:",
        "",
        "```bash",
        "python -m pip install opencv-python numpy",
        "python main.py",
        "python generate_report.py",
        "```",
        "",
        "Для повторного выбора эталонной области: `python main.py --reselect-roi`. После выбора кадра нажмите Enter, затем отметьте четыре угла объекта и подтвердите Enter; R сбрасывает точки, Esc отменяет выбор.",
        "",
        "## 3. Результаты работы и тестирования системы",
        "",
        f"Обработано **{len(rows)} видео**, всего **{total_frames} кадров**. "
        f"Рамка построена на **{total_detected} кадрах ({fmt_pct(total_detected, total_frames)})**.",
        "",
        "Доля обнаружений определяется как число кадров с построенной рамкой, делённое на число "
        "обрабатываемых кадров. Эта метрика не равна точности геометрической локализации, "
        "поскольку разметка истинных границ объекта отсутствует.",
        "",
    ]
    if methods_complete:
        lines += [
            "| Видео | Кадров | SIFT | Lucas–Kanade | Обнаружено | Доля обнаружений |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            sift, flow = methods[row["video"]]
            lines.append(f"| `{row['video']}` | {row['frames']} | {sift} | {flow} | {row['detected']} | {fmt_pct(row['detected'], row['frames'])} |")
        sift_total = sum(methods[row["video"]][0] for row in rows)
        flow_total = sum(methods[row["video"]][1] for row in rows)
        lines += [
            f"| **Итого** | **{total_frames}** | **{sift_total}** | **{flow_total}** | **{total_detected}** | **{fmt_pct(total_detected, total_frames)}** |",
            "",
            f"Резервный трекинг Lucas–Kanade обеспечил **{flow_total} дополнительных обнаружений** "
            f"({fmt_pct(flow_total, total_detected)} от всех обнаружений). "
            "В данной реализации метод LK запускается только после неуспешного SIFT.",
            "",
        ]
    else:
        lines += [
            "| Видео | Кадров | Обнаружено | Доля обнаружений |",
            "|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(f"| `{row['video']}` | {row['frames']} | {row['detected']} | {fmt_pct(row['detected'], row['frames'])} |")
        lines += [
            f"| **Итого** | **{total_frames}** | **{total_detected}** | **{fmt_pct(total_detected, total_frames)}** |",
            "",
        ]

    lines += ["### Анализ причин пропусков", "", "| Причина | Число кадров |", "|---|---:|",]
    for reason in REASONS:
        count = sum(row[reason] for row in rows)
        lines.append(f"| {REASON_NAMES[reason]} | {count} |")
    lines += ["", "### Примеры результатов", ""]
    if images:
        for row in rows:
            path = images.get(row["video"])
            if path:
                lines += [f"**{row['video']}**", "", f"![Пример трекинга: {row['video']}]({path})", ""]
    else:
        lines += ["Примеры кадров не добавлены.", ""]

    lines += ["## 4. Выводы по работе", ""]
    lines += [
        "Реализован алгоритм трекинга плоского объекта на основе ключевых точек SIFT, "
        "оценки гомографии методом RANSAC и резервного оптического потока Lucas–Kanade. "
        "Система обрабатывает набор видео, позволяет выбирать эталон по четырём углам на произвольном кадре и сохраняет рамки, диагностические изображения и количественные показатели.",
        "",
        f"На {len(rows)} видео получена общая доля обнаружений **{fmt_pct(total_detected, total_frames)}**. "
        "Доля обнаружений отражает наличие принятой рамки и не является измерением точности локализации. Качество зависит от текстуры объекта, масштаба, смазывания, числа корректных соответствий "
        "и допустимости найденной геометрии.",
        "",
    ]
    if own:
        row = own[0]
        lines += [
            f"На самостоятельно снятом видео `{row['video']}` рамка построена на "
            f"**{row['detected']} из {row['frames']} кадров ({fmt_pct(row['detected'], row['frames'])})**.",
            "",
        ]
    if methods_complete:
        lines += [
            "Добавление Lucas–Kanade позволило продолжать отслеживание на кадрах, где SIFT не прошёл проверки. "
            "Наличие рамки само по себе не гарантирует точности её положения; "
            "для количественной оценки ошибки локализации потребовалась бы покадровая разметка эталонных границ.",
            "",
        ]
    lines += [
        "### Итог последнего улучшения",
        "",
        "В последнем эксперименте улучшение геометрической проверки повысило долю обнаружений с 84,7% до 93,6% (2554/2728 кадров). Для `simple_track.avi` результат вырос с 39,3% до 81,7%, а число пропусков с причиной `invalid_geometry` снизилось с 314 до 78. Это сравнение зафиксированных запусков; актуальные показатели в таблице выше берутся из CSV.",
        "",
        "## 5. Использованные источники",
        "",
        "1. Lowe D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision*, 60, 91–110. https://doi.org/10.1023/B:VISI.0000029664.99615.94",
        "2. Fischler M. A., Bolles R. C. (1981). Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. *Communications of the ACM*, 24(6), 381–395. https://doi.org/10.1145/358669.358692",
        "3. Lucas B. D., Kanade T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. *IJCAI 1981*, 674–679.",
        "4. OpenCV documentation: Feature Matching, Homography, Optical Flow. https://docs.opencv.org/4.x/",
        "5. Методические указания к лабораторной работе №2 по компьютерному зрению (`LW2_25.pdf`).",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Генерация README для ЛР2")
    parser.add_argument("--no-images", action="store_true", help="Не извлекать кадры из видео")
    parser.add_argument("--frames", default="report_frames.json",
                        help="JSON с номерами выбранных кадров")
    args = parser.parse_args()
    frame_config = ROOT / args.frames
    selected_frames = json.loads(frame_config.read_text(encoding="utf-8")) if frame_config.exists() else {}
    metrics_path = ROOT / "results" / "tracking_metrics.csv"
    rows = load_metrics(metrics_path)
    methods = load_methods(ROOT / "results" / "tracking_methods.csv")
    # Если main.py уже сохраняет раздельные столбцы, отдельный файл не нужен.
    for row in rows:
        if "sift_detected" in row and "flow_detected" in row:
            methods[row["video"]] = (int(row["sift_detected"]), int(row["flow_detected"]))
    images = {} if args.no_images else extract_examples(rows, ROOT / "results", ROOT / "results" / "report_images", selected_frames)
    readme_path = ROOT / "README.md"
    readme_path.write_text(generate_readme(rows, methods, images), encoding="utf-8")
    print(f"README создан: {readme_path}")
    print(f"Видео в отчёте: {len(rows)}; иллюстраций: {len(images)}")


if __name__ == "__main__":
    main()
