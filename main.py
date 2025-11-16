"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
import logging
import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

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


class CTRTargetEncoder(BaseEstimator, TransformerMixin):
    """
    CTR (target) энкодер для категориальных признаков.
    На выходе для каждого категориального признака формируется числовая колонка
    с априорно-сглаженной оценкой P(reward=1 | category).
    """
    def __init__(self, columns, alpha: float = 5.0, handle_unknown: str = "global_mean"):
        # ВАЖНО: не модифицировать параметры в __init__, чтобы поддержать sklearn.clone
        self.columns = columns
        self.alpha = float(alpha)
        self.handle_unknown = handle_unknown
        self.global_mean_ = None
        self.mapping_ = {}
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("CTRTargetEncoder требует y (целевую переменную) при fit.")
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X
        y_arr = pd.Series(y).astype(float)
        columns = list(self.columns)
        
        # Глобальный CTR
        self.global_mean_ = float(y_arr.mean()) if len(y_arr) > 0 else 0.0
        self.mapping_ = {}
        feature_names_out = []
        
        for col in columns:
            # Групповая статистика
            stats = (
                X_df[[col]]
                .assign(target=y_arr.values)
                .groupby(col)["target"]
                .agg(["sum", "count"])
            )
            # Сглаженная оценка
            stats["ctr"] = (stats["sum"] + self.alpha * self.global_mean_) / (stats["count"] + self.alpha)
            # Сохраняем маппинг
            self.mapping_[col] = stats["ctr"].to_dict()
            feature_names_out.append(f"{col}_ctr")
        
        self.feature_names_out_ = np.array(feature_names_out, dtype=object)
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X
        encoded_cols = []
        for col in list(self.columns):
            mapping = self.mapping_.get(col, {})
            col_encoded = X_df[col].map(mapping)
            if self.handle_unknown == "global_mean":
                col_encoded = col_encoded.fillna(self.global_mean_)
            else:
                # fallback на 0.0 если политика иная; по умолчанию используем global_mean
                col_encoded = col_encoded.fillna(0.0)
            encoded_cols.append(col_encoded.astype(float).values.reshape(-1, 1))
        if not encoded_cols:
            return np.empty((len(X_df), 0))
        return np.hstack(encoded_cols)
    
    def get_feature_names_out(self, input_features=None):
        cols = list(self.columns)
        return self.feature_names_out_ if self.feature_names_out_ is not None else np.array([f"{c}_ctr" for c in cols])


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
    Обновлено: для категориальных фич используется CTR target encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat_ctr", CTRTargetEncoder(CATEGORICAL_FEATURES, alpha=5.0, handle_unknown="global_mean"), CATEGORICAL_FEATURES),
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


def compute_action_stats(train_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, int], str]:
    """
    Подсчёт средних наград и количества примеров по каждому действию (segment).
    Возвращает:
      - средние награды по действиям
      - количества по действиям
      - лучшая рука (действие) по среднему reward
    """
    reward_means: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    best_action: str = ACTIONS[0]
    best_mean = -np.inf
    for action in ACTIONS:
        mask = train_df[ACTION_COL] == action
        cnt = int(mask.sum())
        counts[action] = cnt
        if cnt > 0:
            mean_reward = float(train_df.loc[mask, REWARD_COL].mean())
        else:
            mean_reward = -np.inf
        reward_means[action] = mean_reward
        if mean_reward > best_mean:
            best_mean = mean_reward
            best_action = action
    return reward_means, counts, best_action


def make_rl_wrapped_policy(
    q_values: np.ndarray,
    baseline_best_action_idx: int,
    trained_action_mask: np.ndarray,
    epsilon: float = 0.1,
    override_delta: float = 0.15,
) -> np.ndarray:
    """
    Лёгкая RL-обёртка над политикой:
      1) Начинать с one-hot на лучшую руку (по train среднему reward)
      2) Разрешать ML переопределять, только если разница вероятностей
         (max_q - q_best) > override_delta и модель для выбранной руки обучена
      3) Внести ε-эксплорацию: смешиваем с uniform с вероятностью ε
    """
    n_samples, n_actions = q_values.shape
    assert n_actions == len(ACTIONS)
    uniform = np.full((n_samples, n_actions), 1.0 / n_actions, dtype=float)
    # Базовый one-hot на лучшую руку
    base = np.zeros((n_samples, n_actions), dtype=float)
    base[:, baseline_best_action_idx] = 1.0
    # Кандидат от ML
    argmax_actions = np.argmax(q_values, axis=1)
    best_q = q_values[:, baseline_best_action_idx]
    max_q = q_values[np.arange(n_samples), argmax_actions]
    allow_override = (max_q - best_q) > override_delta
    # Учитываем, что переопределять можно только на обученную руку
    trained_idx = np.array(trained_action_mask, dtype=bool)
    can_override_to_trained = trained_idx[argmax_actions]
    do_override = allow_override & can_override_to_trained
    # Построим итоговый one-hot до смешивания с uniform
    final_one_hot = base.copy()
    rows_to_override = np.where(do_override)[0]
    if rows_to_override.size > 0:
        final_one_hot[rows_to_override, baseline_best_action_idx] = 0.0
        final_one_hot[rows_to_override, argmax_actions[rows_to_override]] = 1.0
    # ε-эксплорация: смешивание с uniform
    policy = (1.0 - epsilon) * final_one_hot + epsilon * uniform
    # Численно стабилизируем и ренормализуем
    policy = np.clip(policy, 1e-9, 1.0)
    policy = policy / policy.sum(axis=1, keepdims=True)
    return policy


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


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    # predictions - это кортеж (policy_probs, ids)
    policy_probs, ids = predictions
    
    # Создать пандас таблицу submission
    submission = pd.DataFrame({
        ID_COL: ids,
        "p_mens_email": policy_probs[:, ACTION_TO_INDEX["Mens E-Mail"]],
        "p_womens_email": policy_probs[:, ACTION_TO_INDEX["Womens E-Mail"]],
        "p_no_email": policy_probs[:, ACTION_TO_INDEX["No E-Mail"]],
    })
    
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    logging.info(f"✅ Submission сохранён: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Конфигурация (все параметры заданы здесь)
    config = {
        # Пути к данным
        "data": {
            "train_path": "data/train.csv",
            "test_path": "data/test.csv",
            "submission_path": "results/submission.csv",
        },
        # Random seed для воспроизводимости
        "seed": 42,
        # Logging policy propensity (uniform random = 1/3)
        "mu": 0.3333333333,
        # Основные параметры модели
        "model": {
            # Тип базового алгоритма: 'logistic', 'random_forest', 'extra_trees'
            "type": "logistic",
            # Использовать feature engineering (False = только базовые признаки)
            "use_feature_engineering": False,
            # Параметры для логистической регрессии
            "logistic": {
                "max_iter": 2000,
                "C": 1.0,
                "solver": "lbfgs",
            },
            # Параметры для Random Forest
            "random_forest": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 5,
                "min_samples_split": 10,
            },
            # Параметры для Extra Trees
            "extra_trees": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 5,
                "min_samples_split": 10,
            },
        },
        # Политика (policy)
        "policy": {
            # Тип политики: "greedy" или "softmax"
            "type": "greedy",
            # Температура для softmax (только если type="softmax")
            # T < 1: более детерминистично, T = 1: стандартный softmax, T > 1: больше exploration
            "temperature": 0.1,
            # Epsilon для epsilon-greedy (только если type="greedy")
            # Жадная политика: π(a*) = 1 - ε, π(other) = ε / (n_actions - 1)
            "epsilon": 0.05,
            # Минимальная вероятность действия (для стабильности SNIPS)
            "min_prob": 0.01,
            # Delta для override в RL-обёртке
            "override_delta": 0.15,
        },
        # Логирование
        "logging": {
            "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
            "save_experiment_logs": True,
            "experiment_dir": "experiments",
        },
    }
    
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
    
    # RL-статистика по рукам (действиям) на train
    reward_means, counts, best_action = compute_action_stats(train_df)
    best_action_idx = ACTION_TO_INDEX[best_action]
    trained_action_mask = np.array([counts[a] > 0 for a in ACTIONS], dtype=bool)
    logging.info("\n🧠 Статистика по действиям (train):")
    for a in ACTIONS:
        logging.info(f"  {a}: count={counts[a]}, mean_reward={reward_means[a]:.5f}")
    logging.info(f"  → Лучшая рука (baseline): {best_action}")
    
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
    # RL-обёртка перед политикой на инференсе
    override_delta = config["policy"]["override_delta"]
    epsilon = config["policy"]["epsilon"]
    test_policy = make_rl_wrapped_policy(
        test_q_values,
        baseline_best_action_idx=best_action_idx,
        trained_action_mask=trained_action_mask,
        epsilon=epsilon,
        override_delta=override_delta,
    )
    
    # Выведем предикт: аргмакс действий и первые строки
    pred_actions_idx = np.argmax(test_policy, axis=1)
    pred_actions = [ACTIONS[i] for i in pred_actions_idx]
    unique, counts_arr = np.unique(pred_actions, return_counts=True)
    logging.info("\n🖨️ Предикт (распределение выбранных действий на test):")
    for a, c in zip(unique, counts_arr):
        logging.info(f"  {a}: {int(c)}")
    logging.info("Примеры первых 5 предсказаний (id, action, probs):")
    for i in range(min(5, len(test_df))):
        probs_i = test_policy[i]
        logging.info(
            f"  id={test_df.iloc[i][ID_COL]} → {pred_actions[i]} | "
            f"[mens={probs_i[ACTION_TO_INDEX['Mens E-Mail']]:.3f}, "
            f"womens={probs_i[ACTION_TO_INDEX['Womens E-Mail']]:.3f}, "
            f"no={probs_i[ACTION_TO_INDEX['No E-Mail']]:.3f}]"
        )
    
    # Статистика политики
    logging.info(f"\n📊 Статистика политики (test):")
    for i, action in enumerate(ACTIONS):
        mean_prob = test_policy[:, i].mean()
        logging.info(f"  {action}: {mean_prob:.3f} (среднее)")
    
    logging.info("\n✅ Готово!")
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    predictions = (test_policy, test_df[ID_COL])
    create_submission(predictions)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()