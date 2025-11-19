# [文件名: dataset.py]
import torch
import h5py
from torch.utils.data import Dataset

class BasecallingDataset(Dataset):
    """
    一个更健壮的 Dataset：
    - 在 __init__ 时只读取数据集长度（打开后立即关闭文件），避免把 h5py.File 长期保留在父进程中。
    - 在每次 __getitem__ 调用时确保本进程/线程有自己的打开句柄（惰性打开），从而与 DataLoader 的多进程工作模式兼容。
    """
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._h5 = None       # 在需要时打开
        # 只短暂打开以获取长度，然后关闭
        with h5py.File(h5_path, 'r') as f:
            self.length = f['event'].shape[0]
        # debugging counters
        self._getitem_calls = 0

    def _ensure_open(self):
        """在当前进程/线程中打开 HDF5（如果还没打开）。
        这会为每个 worker/进程创建独立句柄，避免跨进程复用文件对象导致的死锁。
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
            self.events = self._h5['event']
            self.labels = self._h5['label']
            self.label_lens = self._h5['label_len']

    def __len__(self):
        return int(self.length)

    def __getitem__(self, idx):
        # 确保文件在当前进程/线程打开
        self._ensure_open()

        # 支持批量索引：如果传入的是 list/tuple/ndarray，则一次性读取一个批次以减少 IO 开销
        import numpy as _np
        is_batch = isinstance(idx, (list, tuple, _np.ndarray))
        if is_batch:
            # 将 idx 转为 numpy array
            idx_arr = _np.array(idx, dtype=int)
            # debug: batch read
            import time as _time, os as _os
            print(f"DEBUG-DATASET PID={_os.getpid()} batch __getitem__ start len={len(idx_arr)} first={int(idx_arr[0])}", flush=True)

            # h5py advanced indexing requires indices to be in increasing order.
            # If not sorted, fall back to per-index reads while preserving original order.
            idx_sorted = _np.sort(idx_arr)
            if _np.all(idx_sorted == idx_arr):
                # fast path: contiguous/sorted indexing supported by h5py
                signals_np = _np.array(self.events[idx_arr])
                labels_np = _np.array(self.labels[idx_arr])
                label_lens_np = _np.array(self.label_lens[idx_arr])
            else:
                # faster fallback: read using sorted indices (h5py accepts increasing order),
                # then reorder to the original requested order to preserve semantics.
                order = _np.argsort(idx_arr)
                sorted_idx = idx_arr[order]
                signals_sorted = _np.array(self.events[sorted_idx])
                labels_sorted = _np.array(self.labels[sorted_idx])
                label_lens_sorted = _np.array(self.label_lens[sorted_idx])

                # compute inverse permutation to restore original order
                inv = _np.argsort(order)
                signals_np = signals_sorted[inv]
                labels_np = labels_sorted[inv]
                label_lens_np = label_lens_sorted[inv]

            signals = torch.tensor(signals_np, dtype=torch.float)
            labels = torch.tensor(labels_np, dtype=torch.long)
            label_lens = torch.tensor(label_lens_np.astype(int), dtype=torch.long)

            print(f"DEBUG-DATASET PID={_os.getpid()} batch __getitem__ done len={len(idx_arr)}", flush=True)
            return signals, labels, label_lens

        # debug: log first few __getitem__ calls to see where DataLoader blocks
        self._getitem_calls += 1
        log_this = (self._getitem_calls <= 20) or (self._getitem_calls % 1000 == 0)
        if log_this:
            import time as _time, os as _os
            _start = _time.time()
            print(f"DEBUG-DATASET PID={_os.getpid()} __getitem__ start idx={idx} call#{self._getitem_calls}", flush=True)

        # 从 HDF5 读取并立即复制为 numpy，然后转换为 torch.tensor，避免引用底层缓冲区
        signal_np = self.events[idx]
        label_np = self.labels[idx]
        label_len_np = self.label_lens[idx]

        # 显式拷贝为 numpy（有时 h5py 返回的数组是延迟对象）
        import numpy as _np, time as _time, os as _os
        signal = torch.tensor(_np.array(signal_np), dtype=torch.float)
        label = torch.tensor(_np.array(label_np), dtype=torch.long)
        label_len = torch.tensor(int(_np.array(label_len_np)), dtype=torch.long)

        if log_this:
            _end = _time.time()
            print(f"DEBUG-DATASET PID={_os.getpid()} __getitem__ done idx={idx} call#{self._getitem_calls} elapsed={_end-_start:.4f}s", flush=True)

        return signal, label, label_len

    def close(self):
        """关闭当前打开的 HDF5 句柄（如果有）。"""
        if getattr(self, '_h5', None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass