"""
Template pipeline for the offline contextual bandit challenge.

Key entry points:
    * load_data             — read train/test CSVs.
    * build_preprocessor    — ColumnTransformer for numeric/binary/categorical blocks.
    * fit_reward_models     — train action-conditional reward estimators q̂(x, a).
    * make_policy           — convert q̂ scores into a stochastic policy via softmax+ε.
    * snips / best_static_ips — offline evaluation helpers.
    * predict_policy        — produce action probabilities for the test set.
    * save_submission       — persist predictions in the required format.

The main() function wires everything together and exposes useful CLI arguments.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Core columns/constants used across the pipeline
ID_COL = "id"
ACTION_COL = "segment"
REWARD_COL = "visit"
ACTIONS: Tuple[str, ...] = ("Mens E-Mail", "Womens E-Mail", "No E-Mail")
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}

NUMERIC_FEATURES = ["recency", "history"]
BINARY_FEATURES = ["mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["zip_code", "channel", "history_segment"]


@dataclass
class Config:
    train_path: str
    test_path: str
    submission_path: str
    model_type: str
    temperature_grid: List[float]
    epsilon_grid: List[float]
    seed: int
    mu: float
    log_level: str


MODEL_REGISTRY = {
    "logistic": lambda seed: LogisticRegression(max_iter=2_000, random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    ),
}


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Contextual bandit baseline template")
    parser.add_argument("--train-path", default="data/train.csv", help="Path to train CSV")
    parser.add_argument("--test-path", default="data/test.csv", help="Path to test CSV")
    parser.add_argument(
        "--submission-path",
        default="results/submission.csv",
        help="Where to write submission.csv",
    )
    parser.add_argument(
        "--model-type",
        choices=sorted(MODEL_REGISTRY),
        default="logistic",
        help="Reward model family per action",
    )
    parser.add_argument(
        "--temperature-grid",
        default="1.0",
        help="Comma-separated list of temperatures T for softmax policy",
    )
    parser.add_argument(
        "--epsilon-grid",
        default="0.05",
        help="Comma-separated list of epsilon-mix values for exploration",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--mu",
        type=float,
        default=1 / 3,
        help="Logging policy propensity for observed actions",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of logger",
    )
    args = parser.parse_args()
    return Config(
        train_path=args.train_path,
        test_path=args.test_path,
        submission_path=args.submission_path,
        model_type=args.model_type,
        temperature_grid=_parse_float_grid(args.temperature_grid),
        epsilon_grid=_parse_float_grid(args.epsilon_grid),
        seed=args.seed,
        mu=args.mu,
        log_level=args.log_level,
    )


def _parse_float_grid(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Temperature/epsilon grid cannot be empty")
    return values


def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Loading data from %s and %s", train_path, test_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _make_estimator(model_type: str, seed: int) -> BaseEstimator:
    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model type {model_type}. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_type](seed)


def fit_reward_models(
    train_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    model_type: str,
    seed: int,
) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    for action in ACTIONS:
        mask = train_df[ACTION_COL] == action
        if not mask.any():
            raise ValueError(f"No training rows for action {action}")
        y = train_df.loc[mask, REWARD_COL]
        if y.nunique() == 1:
            logging.warning(
                "Action %s has a single reward class; falling back to DummyClassifier.",
                action,
            )
            estimator: BaseEstimator = DummyClassifier(strategy="constant", constant=y.iloc[0])
        else:
            estimator = _make_estimator(model_type, seed)
        pipeline = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(train_df.loc[mask], y)
        models[action] = pipeline
        logging.info("Fitted reward model for %s on %d rows", action, mask.sum())
    return models


def _predict_q_values(df: pd.DataFrame, models: Dict[str, Pipeline]) -> np.ndarray:
    q_columns = []
    for action in ACTIONS:
        model = models[action]
        probs = model.predict_proba(df)[:, 1]
        q_columns.append(probs)
    return np.column_stack(q_columns)


def make_policy(q_hat: np.ndarray, temperature: float, epsilon: float) -> np.ndarray:
    if q_hat.ndim != 2 or q_hat.shape[1] != len(ACTIONS):
        raise ValueError("q_hat must be of shape [n_samples, n_actions]")
    temp = max(temperature, 1e-6)
    logits = q_hat / temp
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    eps = float(np.clip(epsilon, 0.0, 1.0))
    if eps > 0:
        probs = (1 - eps) * probs + eps / probs.shape[1]
    return probs


def snips(
    pi_probs: np.ndarray,
    a_logged: Iterable[str],
    rewards: Iterable[float],
    mu: float | np.ndarray = 1 / 3,
) -> float:
    a_logged = np.asarray(list(a_logged))
    rewards = np.asarray(list(rewards))
    idx = np.array([ACTION_TO_INDEX.get(action, -1) for action in a_logged], dtype=int)
    valid = idx >= 0
    if not np.all(valid):
        logging.warning("Some logged actions are unknown and will be ignored in SNIPS.")
    if not valid.any():
        return 0.0
    rows = np.where(valid)[0]
    pi_selected = pi_probs[rows, idx[valid]]
    mu_vec = (
        np.full(rows.shape[0], float(mu))
        if np.isscalar(mu)
        else np.asarray(mu, dtype=float)[valid]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.divide(pi_selected, mu_vec, out=np.zeros_like(pi_selected), where=mu_vec != 0)
    denom = weights.sum()
    if denom == 0:
        return 0.0
    return float(np.dot(weights, rewards[valid]) / denom)


def best_static_ips(
    a_logged: Iterable[str],
    rewards: Iterable[float],
    mu: float | np.ndarray = 1 / 3,
) -> float:
    a_logged = np.asarray(list(a_logged))
    rewards = np.asarray(list(rewards), dtype=float)
    best = -np.inf
    for action in ACTIONS:
        mask = a_logged == action
        if not mask.any():
            continue
        if np.isscalar(mu):
            mu_vec = float(mu)
            numerator = rewards[mask].sum() / mu_vec
            denominator = mask.sum() / mu_vec
        else:
            mu_vals = np.asarray(mu, dtype=float)[mask]
            numerator = np.sum(rewards[mask] / mu_vals)
            denominator = np.sum(1.0 / mu_vals)
        value = numerator / denominator if denominator else 0.0
        best = max(best, value)
    return best if np.isfinite(best) else 0.0


def predict_policy(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    temperature: float,
    epsilon: float,
) -> np.ndarray:
    q_hat = _predict_q_values(df, models)
    return make_policy(q_hat, temperature, epsilon)


def save_submission(submission_df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    submission_df.to_csv(path, index=False)
    logging.info("Submission saved to %s", path)
    return path


def select_best_policy(
    q_hat: np.ndarray,
    train_df: pd.DataFrame,
    temperatures: List[float],
    epsilons: List[float],
    mu: float,
) -> Tuple[float, float, float, float]:
    best_score = -np.inf
    best_metrics = (temperatures[0], epsilons[0], -np.inf, -np.inf)
    rewards = train_df[REWARD_COL]
    actions = train_df[ACTION_COL]
    static_reference = best_static_ips(actions, rewards, mu)
    for temp, eps in product(temperatures, epsilons):
        policy = make_policy(q_hat, temp, eps)
        value = snips(policy, actions, rewards, mu)
        score = value - static_reference
        logging.info(
            "T=%.3f, eps=%.3f => SNIPS=%.5f, best_static=%.5f, score=%.5f",
            temp,
            eps,
            value,
            static_reference,
            score,
        )
        if score > best_score:
            best_score = score
            best_metrics = (temp, eps, value, static_reference)
    return (*best_metrics, best_score)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=getattr(logging, level.upper()),
    )


def create_submission(pred_matrix: np.ndarray, ids: Iterable[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: ids,
            "p_mens_email": pred_matrix[:, ACTION_TO_INDEX["Mens E-Mail"]],
            "p_womens_email": pred_matrix[:, ACTION_TO_INDEX["Womens E-Mail"]],
            "p_no_email": pred_matrix[:, ACTION_TO_INDEX["No E-Mail"]],
        }
    )


def main() -> None:
    config = parse_args()
    setup_logging(config.log_level)
    set_global_determinism(config.seed)

    train_df, test_df = load_data(config.train_path, config.test_path)
    preprocessor = build_preprocessor()
    models = fit_reward_models(train_df, preprocessor, config.model_type, config.seed)

    train_q_hat = _predict_q_values(train_df, models)
    temp, eps, snips_value, best_static_value, final_score = select_best_policy(
        train_q_hat,
        train_df,
        config.temperature_grid,
        config.epsilon_grid,
        config.mu,
    )
    logging.info(
        "Selected policy: T=%.3f, eps=%.3f (SNIPS=%.5f, best_static=%.5f, score=%.5f)",
        temp,
        eps,
        snips_value,
        best_static_value,
        final_score,
    )

    test_policy = predict_policy(test_df, models, temp, eps)
    submission_df = create_submission(test_policy, test_df[ID_COL])
    save_submission(submission_df, config.submission_path)


if __name__ == "__main__":
    main()
