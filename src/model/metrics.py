from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, log_loss


@dataclass
# 대회용 평가 지표를 한 번에 다루기 위한 컨테이너.
class CompetitionMetrics:
    average_precision: float
    weighted_logloss: float
    competition_score: float

    # 메트릭을 wandb나 CSV 기록과 호환되는 딕셔너리로 만든다.
    def to_dict(self) -> Dict[str, float]:
        return {
            "ap": self.average_precision,
            "wll": self.weighted_logloss,
            "competition_score": self.competition_score,
        }


# 클래스 불균형을 고려한 가중 로그로스를 계산한다.
def compute_weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_pos = np.sum(y_true)
    n = y_true.shape[0]
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return log_loss(y_true, y_pred)
    pos_weight = 0.5 / (n_pos / n)
    neg_weight = 0.5 / (n_neg / n)
    sample_weight = np.where(y_true == 1, pos_weight, neg_weight)
    return log_loss(y_true, y_pred, sample_weight=sample_weight)


# Average Precision과 가중 로그로스를 합산한 대회 스코어를 반환한다.
def compute_competition_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> CompetitionMetrics:
    ap = average_precision_score(y_true, y_pred)
    wll = compute_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1.0 - wll)
    return CompetitionMetrics(ap, wll, score)
