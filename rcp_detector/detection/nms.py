"""Non-maximum suppression for post-reconstruction deduplication.

Extracted from Reconstructiing_less.py — pure NumPy, no framework dependency.
"""

import numpy as np


def non_max_suppression(
    boxes: np.ndarray,
    overlap_thresh: float = 0.3,
) -> np.ndarray:
    """Greedy NMS on boxes shaped (N, 6): [x1, y1, x2, y2, score, class_id].

    Returns the kept rows as an integer array.
    """
    if len(boxes) == 0:
        return np.empty((0, 6), dtype=int)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    pick: list[int] = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs,
            np.concatenate(([last], np.where(overlap > overlap_thresh)[0])),
        )

    return boxes[pick].astype("int")
