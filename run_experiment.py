"""Chay thuc nghiem: central / fedavg / fedq (+ nen NNC tuy chon).

Vi du:
  python run_experiment.py --data_dir ml-latest-small --mode central
  python run_experiment.py --data_dir ml-latest-small --mode fedavg --clients_per_round 100
  python run_experiment.py --data_dir ml-latest-small --mode fedq --clients_per_round 100 --queue 10
  python run_experiment.py --data_dir ml-latest-small --mode fedq --clients_per_round 100 --queue 10 --qp -30
"""
import argparse, csv, os, time
import numpy as np
import torch
import torch.nn as nn

from data import load_clients
from model import CandidateGenerator, top_k_accuracy
from fl import run_round, NNCCodec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--mode", choices=["central", "fedavg", "fedq"], default="fedavg")
    p.add_argument("--max_users", type=int, default=None)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--clients_per_round", type=int, default=100)  # N
    p.add_argument("--queue", type=int, default=10)                # L (chi cho fedq)
    p.add_argument("--epochs", type=int, default=1)                # E
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)               # bai bao: giam 10x so voi central
    p.add_argument("--qp", type=int, default=None)                 # None = khong nen; vd -30
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clients, (tx, ty), n_movies = load_clients(a.data_dir, a.max_users, a.seed)

    norm = "batch" if a.mode == "central" else "group"
    model = CandidateGenerator(n_movies, norm=norm).to(device)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.mode}_N{a.clients_per_round}_L{a.queue if a.mode=='fedq' else 1}_qp{a.qp}"
    log = open(os.path.join(a.out, tag + ".csv"), "w", newline="")
    w = csv.writer(log); w.writerow(["round", "top100_acc", "MB_sent", "MB_raw", "sec"])

    t0 = time.time()
    if a.mode == "central":  # baseline khong-FL
        X = np.concatenate([c[0] for c in clients]); y = np.concatenate([c[1] for c in clients])
        opt = torch.optim.SGD(model.parameters(), lr=a.lr * 10, momentum=0.9)
        for ep in range(1, a.rounds + 1):
            model.train(); perm = np.random.permutation(len(y))
            for i in range(0, len(y), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                nn.functional.cross_entropy(
                    model(torch.from_numpy(X[idx]).to(device)),
                    torch.from_numpy(y[idx]).to(device)).backward()
                opt.step()
            acc = top_k_accuracy(model, tx, ty, device=device)
            print(f"epoch {ep} top100={acc:.4f}"); w.writerow([ep, acc, 0, 0, time.time() - t0])
        return

    codec = NNCCodec(a.qp)
    L = a.queue if a.mode == "fedq" else 1
    N = (min(a.clients_per_round, len(clients)) // L) * L
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for r in range(1, a.rounds + 1):
        selected = list(np.random.choice(len(clients), N, replace=False))
        state = run_round(model, state, clients, selected, L, a.epochs,
                          a.batch_size, a.lr, device, codec)
        model.load_state_dict(state)
        acc = top_k_accuracy(model, tx, ty, device=device)
        print(f"round {r}/{a.rounds} top100={acc:.4f} sent={codec.bytes_sent/1e6:.1f}MB")
        w.writerow([r, acc, codec.bytes_sent / 1e6, codec.bytes_raw / 1e6, time.time() - t0])
        log.flush()
    if a.qp is not None:
        print(f"Tiet kiem bang thong: {100*(1-codec.bytes_sent/codec.bytes_raw):.2f}%")


if __name__ == "__main__":
    main()
