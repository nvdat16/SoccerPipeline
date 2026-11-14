"""Gộp dự đoán số áo theo tracklet — giống helpers.find_best_prediction (repo gốc):
lọc xác suất thấp, cộng xác suất theo từng số (có bias ưu tiên số 2 chữ số), lấy argmax,
và chỉ chốt nếu tổng trọng số > SUM_THRESHOLD; ngược lại = -1 (không đọc được).
"""
from collections import defaultdict


def is_valid_number(s: str) -> bool:
    return s.isdigit() and len(s) <= 2 and 0 < int(s) < 100


def bias(number: int) -> float:
    return 0.61 if number > 9 else 0.39


class TrackletVoter:
    def __init__(self, filter_threshold=0.2, sum_threshold=1.0, use_bias=True):
        self.filter_threshold = filter_threshold
        self.sum_threshold = sum_threshold
        self.use_bias = use_bias
        self.reads = defaultdict(list)          # track_id -> [(number, prob)]
        self.n_attempts = defaultdict(int)      # số lần đã thử đọc (kể cả illegible)

    def add(self, track_id: int, label: str, prob: float):
        if is_valid_number(label):
            self.reads[track_id].append((int(label), prob))

    def weights(self, track_id: int) -> dict:
        w = defaultdict(float)
        for number, prob in self.reads.get(track_id, []):
            if prob < self.filter_threshold:
                continue
            w[number] += prob * (bias(number) if self.use_bias else 1.0)
        return dict(w)

    def best(self, track_id: int):
        """(number | -1, weight)."""
        w = self.weights(track_id)
        if not w:
            return -1, 0.0
        number, weight = max(w.items(), key=lambda kv: kv[1])
        return (number if weight > self.sum_threshold else -1), weight
