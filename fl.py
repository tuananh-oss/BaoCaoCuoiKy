"""FedAvg (Alg.1), FedQ (Alg.2) va nen NNC (Muc 3.7) - mo phong tuan tu nhu Phu luc A.

Bo nho thap: chi giu (global, trung binh tich luy, 1 client hien tai).
"""
import copy
import numpy as np
import torch
import torch.nn as nn


# ---------------- NNCodec ----------------
class NNCCodec:
    """Boc nnc.compress / nnc.decompress (thu vien trong nncodec-master.zip).

    qp=None -> khong nen (de so sanh). Dem tong so byte truyen.
    """

    def __init__(self, qp=None):
        self.qp = qp
        self.bytes_sent = 0
        self.bytes_raw = 0
        if qp is not None:
            import nnc  # pip install ./nncodec-master
            self.nnc = nnc

    def roundtrip(self, state):
        """state: OrderedDict tensor -> (state sau nen/giai nen, so byte)."""
        raw = sum(v.numel() * 4 for v in state.values())
        self.bytes_raw += raw
        if self.qp is None:
            self.bytes_sent += raw
            return state
        arrs = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items()}
        bs = self.nnc.compress(arrs, bitstream_path=None, qp=self.qp,
                               return_bitstream=True, verbose=False)
        rec = self.nnc.decompress(bs, verbose=False)
        self.bytes_sent += len(bs)
        out = {}
        for k, v in state.items():
            out[k] = torch.from_numpy(np.asarray(rec[k], dtype=np.float32)).reshape(v.shape)
        return out


# ---------------- Local training ----------------
def update_client(model, state, data, epochs, batch_size, lr, device):
    """UpdateClient(theta, i): SGD thuan (stateless) nhu tac gia."""
    model.load_state_dict(state)
    model.train()
    X, y = data
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(y)
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _accumulate(acc, state, w):
    if acc is None:
        return {k: v.float() * w for k, v in state.items()}
    for k, v in state.items():
        acc[k] += v.float() * w
    return acc


# ---------------- One communication round ----------------
def run_round(model, global_state, clients, selected, queue_len, epochs,
              batch_size, lr, device, codec):
    """queue_len = 1 -> FedAvg; queue_len = L > 1 -> FedQ (N phai chia het L)."""
    assert len(selected) % queue_len == 0, "N phai chia het cho L"
    total = sum(len(clients[i][1]) for i in selected)
    acc = None
    for k in range(len(selected) // queue_len):
        queue = selected[k * queue_len:(k + 1) * queue_len]
        # downstream: server -> client dau hang doi (nen neu co)
        state = codec.roundtrip(global_state)
        s_k = 0
        for ci in queue:
            state = update_client(model, state, clients[ci], epochs, batch_size, lr, device)
            s_k += len(clients[ci][1])
            if ci != queue[-1]:
                state = codec.roundtrip(state)   # client -> client ke tiep
        # upstream: client cuoi hang doi -> server
        state = codec.roundtrip(state)
        acc = _accumulate(acc, state, s_k / total)  # trung binh co trong so
    return acc
