"""Tien xu ly MovieLens -> watch-history dataset (Muc 4.1.2 cua bai bao).

Moi mau: (lich su toi da WINDOW phim truoc do, phim tiep theo).
Moi nguoi dung = 1 client (giong thiet lap 162k clients cua tac gia).
"""
import os
import numpy as np
import pandas as pd

WINDOW = 7  # window size toi uu theo bai bao (Hinh 8)


def load_clients(data_dir, max_users=None, seed=0, test_frac=0.1):
    """Tra ve (clients, test_set, n_movies).

    clients: list cac (X, y); X: [n, WINDOW] (0 = padding), y: [n]
    test_set: (X, y) gop tu phan cuoi (theo thoi gian) cua moi user.
    """
    path = os.path.join(data_dir, "ratings.csv")
    df = pd.read_csv(path, usecols=["userId", "movieId", "timestamp"])

    rng = np.random.RandomState(seed)
    users = df["userId"].unique()
    if max_users is not None and max_users < len(users):
        users = rng.choice(users, max_users, replace=False)
        df = df[df["userId"].isin(users)]

    # movieId -> index 1..N (0 danh cho padding)
    movie_ids = np.sort(df["movieId"].unique())
    mapping = {m: i + 1 for i, m in enumerate(movie_ids)}
    df = df.assign(m=df["movieId"].map(mapping))
    n_movies = len(movie_ids) + 1

    clients, test_X, test_y = [], [], []
    for _, g in df.sort_values("timestamp", kind="stable").groupby("userId"):
        seq = g["m"].to_numpy()
        if len(seq) < 5:
            continue
        X, y = [], []
        for t in range(1, len(seq)):  # sliding window
            h = seq[max(0, t - WINDOW):t]
            X.append(np.pad(h, (WINDOW - len(h), 0)))  # pad ben trai
            y.append(seq[t])
        X, y = np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)
        n_test = max(1, int(len(y) * test_frac))
        clients.append((X[:-n_test], y[:-n_test]))
        test_X.append(X[-n_test:])
        test_y.append(y[-n_test:])

    clients = [c for c in clients if len(c[1]) > 0]
    test = (np.concatenate(test_X), np.concatenate(test_y))
    print(f"[data] {len(clients)} clients | {n_movies-1} phim | "
          f"{sum(len(c[1]) for c in clients)} mau train | {len(test[1])} mau test")
    return clients, test, n_movies
