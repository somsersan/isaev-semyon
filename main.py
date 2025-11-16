"""
Упрощённое решение для Contextual Bandit задачи.

Основные принципы:
1. Минималистичный подход - только проверенные методы
2. Конфигурация через config.yaml
3. Простая логистическая регрессия как baseline
4. Без переусложнения
"""
import logging
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Константы
ID_COL = "id"
ACTION_COL = "segment"
REWARD_COL = "visit"
ACTIONS = ("Mens E-Mail", "Womens E-Mail", "No E-Mail")
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}

# Базовые признаки
NUMERIC_FEATURES = ["recency", "history"]
BINARY_FEATURES = ["mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["zip_code", "channel", "history_segment"]


def load_config(config_path: str = "config.yaml") -> dict:
    """Загрузка конфигурации из YAML файла."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Установка random seed для воспроизводимости."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(level: str = "INFO") -> None:
    """Настройка логирования."""
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=getattr(logging, level.upper()),
    )


def load_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка train и test данных."""
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df = pd.read_csv(config["data"]["test_path"])
    logging.info(f"Загружено: train={len(train_df)}, test={len(test_df)}")
    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """
    Создание препроцессора для признаков.
    Простой подход без feature engineering.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def create_model(config: dict):
    """Создание модели на основе конфига."""
    model_type = config["model"]["type"]
    seed = config["seed"]
    
    if model_type == "logistic":
        params = config["model"]["logistic"]
        return LogisticRegression(
            max_iter=params["max_iter"],
            C=params["C"],
            solver=params["solver"],
            random_state=seed,
        )
    elif model_type == "random_forest":
        params = config["model"]["random_forest"]
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "extra_trees":
        params = config["model"]["extra_trees"]
        return ExtraTreesClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            bootstrap=False,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def fit_reward_models(
    train_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    config: dict,
) -> Dict[str, Pipeline]:
    """
    Обучение отдельной модели для каждого действия.
    Direct Method: P(reward=1 | x, action)
    """
    models = {}
    
    for action in ACTIONS:
        # Данные для этого действия
        mask = train_df[ACTION_COL] == action
        X = train_df.loc[mask]
        y = train_df.loc[mask, REWARD_COL]
        
        # Создание пайплайна
        model = create_model(config)
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        
        # Обучение
        pipeline.fit(X, y)
        models[action] = pipeline
        
        logging.info(f"Обучена модель для '{action}': {mask.sum()} примеров, "
                    f"reward rate = {y.mean():.3f}")
    
    return models


def predict_q_values(df: pd.DataFrame, models: Dict[str, Pipeline]) -> np.ndarray:
    """
    Предсказание Q-значений (вероятностей reward=1) для каждого действия.
    
    Returns:
        Array shape [n_samples, n_actions] с Q(x, a)
    """
    q_columns = []
    for action in ACTIONS:
        model = models[action]
        # Вероятность reward=1
        probs = model.predict_proba(df)[:, 1]
        q_columns.append(probs)
    
    return np.column_stack(q_columns)


def make_policy_greedy(
    q_values: np.ndarray,
    epsilon: float = 0.05,
) -> np.ndarray:
    """
    Жадная (почти детерминистическая) политика с epsilon-greedy.
    
    π(a|x) = 1 - ε,  если a = argmax Q(x,a')
             ε/n,    иначе
    
    Args:
        q_values: Q-значения shape [n_samples, n_actions]
        epsilon: Доля для non-greedy действий
    
    Returns:
        Policy probabilities shape [n_samples, n_actions]
    """
    n_samples, n_actions = q_values.shape
    
    # Жадное действие
    greedy_actions = np.argmax(q_values, axis=1)
    
    # Инициализация с uniform epsilon
    policy = np.full((n_samples, n_actions), epsilon / n_actions)
    
    # Основная вероятность на жадное действие
    policy[np.arange(n_samples), greedy_actions] += (1.0 - epsilon)
    
    # Ренормализация
    policy = policy / policy.sum(axis=1, keepdims=True)
    
    return policy


def make_policy_softmax(
    q_values: np.ndarray,
    temperature: float = 1.0,
    min_prob: float = 0.01,
) -> np.ndarray:
    """
    Softmax политика с температурой.
    
    π(a|x) = softmax(Q/T)
    
    Args:
        q_values: Q-значения shape [n_samples, n_actions]
        temperature: Температура для softmax (ниже = более детерминистично)
        min_prob: Минимальная вероятность для стабильности
    
    Returns:
        Policy probabilities shape [n_samples, n_actions]
    """
    # Softmax с температурой
    logits = q_values / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    # Клиппинг для стабильности
    n_actions = probs.shape[1]
    max_prob = 1.0 - (n_actions - 1) * min_prob
    probs = np.clip(probs, min_prob, max_prob)
    
    # Ренормализация
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    return probs


def make_policy(q_values: np.ndarray, config: dict) -> np.ndarray:
    """
    Создание политики на основе конфигурации.
    
    Args:
        q_values: Q-значения
        config: Конфигурация
    
    Returns:
        Policy probabilities
    """
    policy_type = config["policy"]["type"]
    
    if policy_type == "greedy":
        return make_policy_greedy(q_values, epsilon=config["policy"]["epsilon"])
    elif policy_type == "softmax":
        return make_policy_softmax(
            q_values,
            temperature=config["policy"]["temperature"],
            min_prob=config["policy"]["min_prob"],
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def snips_score(
    policy_probs: np.ndarray,
    actions: pd.Series,
    rewards: pd.Series,
    mu: float = 1/3,
) -> float:
    """
    Вычисление SNIPS (Self-Normalized Importance Sampling) метрики.
    
    SNIPS = Σ(π(a|x)/μ * r) / Σ(π(a|x)/μ)
    """
    actions_arr = actions.values
    rewards_arr = rewards.values
    
    # Индексы действий
    action_indices = np.array([ACTION_TO_INDEX[a] for a in actions_arr])
    
    # π(a_logged | x)
    pi_logged = policy_probs[np.arange(len(policy_probs)), action_indices]
    
    # Важность (importance weights)
    weights = pi_logged / mu
    
    # SNIPS
    numerator = np.sum(weights * rewards_arr)
    denominator = np.sum(weights)
    
    return numerator / denominator if denominator > 0 else 0.0


def best_static_policy_value(actions: pd.Series, rewards: pd.Series, mu: float = 1/3) -> float:
    """
    Значение лучшей статической политики (выбирает одно действие всегда).
    Это baseline для сравнения.
    
    Best Static IPS = max_a [ E[r | a] ] = max_a [ mean(rewards where action=a) ]
    """
    best_value = -np.inf
    
    for action in ACTIONS:
        mask = actions == action
        if mask.sum() == 0:
            continue
        
        # Среднее вознаграждение для этого действия
        # IPS оценка: (sum(r)/mu) / (count/mu) = sum(r) / count = mean(r)
        value = rewards[mask].mean()
        best_value = max(best_value, value)
    
    return best_value


def create_submission(policy_probs: np.ndarray, ids: pd.Series) -> pd.DataFrame:
    """Создание submission файла."""
    return pd.DataFrame({
        ID_COL: ids,
        "p_mens_email": policy_probs[:, ACTION_TO_INDEX["Mens E-Mail"]],
        "p_womens_email": policy_probs[:, ACTION_TO_INDEX["Womens E-Mail"]],
        "p_no_email": policy_probs[:, ACTION_TO_INDEX["No E-Mail"]],
    })


def save_submission(submission_df: pd.DataFrame, path: str) -> None:
    """Сохранение submission файла."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    submission_df.to_csv(path, index=False)
    logging.info(f"✅ Submission сохранён: {path}")


def main():
    """Главная функция."""
    # Загрузка конфига
    config = load_config("config.yaml")
    
    # Настройка
    set_seed(config["seed"])
    setup_logging(config["logging"]["level"])
    
    logging.info("=" * 60)
    logging.info("CONTEXTUAL BANDIT - SIMPLIFIED BASELINE")
    logging.info("=" * 60)
    logging.info(f"Модель: {config['model']['type']}")
    logging.info(f"Политика: {config['policy']['type']}")
    if config['policy']['type'] == 'greedy':
        logging.info(f"Epsilon: {config['policy']['epsilon']}")
    elif config['policy']['type'] == 'softmax':
        logging.info(f"Temperature: {config['policy']['temperature']}")
    logging.info(f"Seed: {config['seed']}")
    logging.info("=" * 60)
    
    # Загрузка данных
    train_df, test_df = load_data(config)
    
    # Препроцессор
    preprocessor = build_preprocessor()
    
    # Обучение моделей (Direct Method)
    logging.info("\n🔧 Обучение reward моделей...")
    models = fit_reward_models(train_df, preprocessor, config)
    
    # Предсказание Q-значений на train для оценки
    logging.info("\n📊 Оценка на train данных...")
    train_q_values = predict_q_values(train_df, models)
    
    # Создание политики
    train_policy = make_policy(train_q_values, config)
    
    # Оценка политики
    snips_value = snips_score(train_policy, train_df[ACTION_COL], train_df[REWARD_COL], config["mu"])
    best_static = best_static_policy_value(train_df[ACTION_COL], train_df[REWARD_COL], config["mu"])
    score = snips_value - best_static
    
    logging.info(f"\n📈 МЕТРИКИ:")
    logging.info(f"  SNIPS: {snips_value:.5f}")
    logging.info(f"  Best Static: {best_static:.5f}")
    logging.info(f"  Score (SNIPS - Best Static): {score:.5f}")
    
    # Предсказание на test
    logging.info("\n🎯 Генерация submission...")
    test_q_values = predict_q_values(test_df, models)
    test_policy = make_policy(test_q_values, config)
    
    # Сохранение submission
    submission_df = create_submission(test_policy, test_df[ID_COL])
    save_submission(submission_df, config["data"]["submission_path"])
    
    # Статистика политики
    logging.info(f"\n📊 Статистика политики (test):")
    for i, action in enumerate(ACTIONS):
        mean_prob = test_policy[:, i].mean()
        logging.info(f"  {action}: {mean_prob:.3f} (среднее)")
    
    logging.info("\n✅ Готово!")


if __name__ == "__main__":
    main()

