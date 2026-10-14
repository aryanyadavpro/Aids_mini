"""
src/baseline.py — Random Forest Baseline for ResoGenre.

Extracts handcrafted audio features and trains a Random Forest
classifier as a baseline benchmark against the 2D CNN.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from config import (
    RAW_DATA_DIR,
    MODELS_DIR,
    GENRES,
    GENRE_TO_IDX,
    RF_N_ESTIMATORS,
    RANDOM_SEED,
    BASELINE_MODEL_FILENAME,
)
from src.features import load_audio, segment_audio, extract_handcrafted_features
from src.dataset import get_song_paths, split_songs

logger = logging.getLogger(__name__)


def extract_features_from_songs(
    songs: list[tuple[Path, str]],
    use_segments: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract handcrafted features from a list of songs.

    Args:
        songs: List of (filepath, genre) tuples.
        use_segments: If True, extract features per 3s segment.
                      If False, extract from full 30s track.

    Returns:
        Tuple of (feature_matrix, label_array).
    """
    all_features = []
    all_labels = []

    for filepath, genre in songs:
        label = GENRE_TO_IDX[genre]
        try:
            y, sr = load_audio(str(filepath))
            if use_segments:
                segments = segment_audio(y)
                for seg in segments:
                    features = extract_handcrafted_features(seg)
                    all_features.append(features)
                    all_labels.append(label)
            else:
                features = extract_handcrafted_features(y)
                all_features.append(features)
                all_labels.append(label)
        except Exception as e:
            logger.warning(f"Skipping {filepath}: {e}")

    X = np.array(all_features)
    y = np.array(all_labels)
    logger.info(f"Extracted features: X={X.shape}, y={y.shape}")
    return X, y


def train_baseline(
    data_dir: Path = RAW_DATA_DIR,
    n_estimators: int = RF_N_ESTIMATORS,
    seed: int = RANDOM_SEED,
    save_model: bool = True,
) -> tuple[RandomForestClassifier, dict]:
    """Train Random Forest baseline model.

    Args:
        data_dir: Directory with genre subdirectories.
        n_estimators: Number of trees in the forest.
        seed: Random seed.
        save_model: Whether to save the trained model.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    csv_path = Path("Music_data/Data/features_3_sec.csv")
    if not csv_path.exists():
        csv_path = Path("data/features_3_sec.csv")

    if csv_path.exists():
        import pandas as pd
        logger.info(f"Loading pre-extracted features from {csv_path}...")
        df = pd.read_csv(csv_path)
        # Extract song_id from filename (e.g. blues.00000.0.wav -> blues.00000)
        df["song_id"] = df["filename"].apply(lambda f: ".".join(f.split(".")[:2]))
        unique_songs = df[["song_id", "label"]].drop_duplicates()

        train_songs, temp_songs = train_test_split(
            unique_songs, test_size=0.2, random_state=seed, stratify=unique_songs["label"]
        )
        val_songs, test_songs = train_test_split(
            temp_songs, test_size=0.5, random_state=seed, stratify=temp_songs["label"]
        )

        train_df = df[df["song_id"].isin(train_songs["song_id"])]
        val_df = df[df["song_id"].isin(val_songs["song_id"])]
        test_df = df[df["song_id"].isin(test_songs["song_id"])]

        feature_cols = [c for c in df.columns if c not in ["filename", "length", "label", "song_id"]]
        X_train, y_train = train_df[feature_cols].values, train_df["label"].map(GENRE_TO_IDX).values
        X_val, y_val = val_df[feature_cols].values, val_df["label"].map(GENRE_TO_IDX).values
        X_test, y_test = test_df[feature_cols].values, test_df["label"].map(GENRE_TO_IDX).values
        logger.info(f"Loaded {len(X_train)} train, {len(X_val)} val, {len(X_test)} test segments")
    else:
        # Get and split songs (leakage-safe)
        songs = get_song_paths(data_dir)
        train_songs, val_songs, test_songs = split_songs(songs)

        # Extract features
        logger.info("Extracting training features...")
        X_train, y_train = extract_features_from_songs(train_songs)
        logger.info("Extracting validation features...")
        X_val, y_val = extract_features_from_songs(val_songs)
        logger.info("Extracting test features...")
        X_test, y_test = extract_features_from_songs(test_songs)

    # Train Random Forest
    logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    val_pred = rf.predict(X_val)
    test_pred = rf.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    report = classification_report(
        y_test, test_pred, target_names=GENRES
    )

    metrics = {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "classification_report": report,
    }

    logger.info(f"Baseline Val Accuracy: {val_acc:.4f}")
    logger.info(f"Baseline Test Accuracy: {test_acc:.4f}")
    logger.info(f"\n{report}")

    # Save model
    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / BASELINE_MODEL_FILENAME
        with open(model_path, "wb") as f:
            pickle.dump(rf, f)
        logger.info(f"Baseline model saved to {model_path}")

    return rf, metrics


if __name__ == "__main__":
    from src.utils import setup_logging, ensure_dirs, set_seed
    setup_logging()
    ensure_dirs()
    set_seed()
    logger.info("Starting Baseline Model Training...")
    train_baseline()

