import os
import struct
import pickle
import zipfile
import io

INCOMPLETE = r"C:\Users\gsart\Downloads\cod_models_github\.git\lfs\incomplete"
CKPT_DIR = r"C:\Users\gsart\Downloads\cod_models_github\IFBO_NET_V3_checkpoints"


def find_sources():
    srcs = []
    if os.path.isdir(INCOMPLETE):
        for name in os.listdir(INCOMPLETE):
            srcs.append(os.path.join(INCOMPLETE, name))
    if os.path.isdir(CKPT_DIR):
        for name in os.listdir(CKPT_DIR):
            path = os.path.join(CKPT_DIR, name)
            if os.path.getsize(path) > 1000:
                srcs.append(path)
    return srcs


def extract_datapkl(path, max_read=32 * 1024 * 1024):
    size = os.path.getsize(path)
    n = min(size, max_read)
    with open(path, "rb") as f:
        data = f.read(n)
    if data[:4] != b"PK\x03\x04":
        raise RuntimeError(f"not a zip: {path} magic={data[:8]!r}")

    off = 0
    while off + 30 <= len(data):
        if data[off : off + 4] != b"PK\x03\x04":
            nxt = data.find(b"PK\x03\x04", off + 1)
            if nxt < 0:
                break
            off = nxt
            continue
        nlen = struct.unpack_from("<H", data, off + 26)[0]
        elen = struct.unpack_from("<H", data, off + 28)[0]
        name = data[off + 30 : off + 30 + nlen].decode("utf-8", "replace")
        payload = off + 30 + nlen + elen
        nxt = data.find(b"PK\x03\x04", payload)
        if name.endswith("data.pkl"):
            blob = data[payload:nxt] if nxt > 0 else data[payload:]
            return name, blob, size
        if nxt < 0:
            break
        off = nxt
    raise RuntimeError(f"data.pkl not found in first {n} bytes of {path}")


class _Stub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.state = state

    def __repr__(self):
        return f"<stub {self.args!r}>"

    def append(self, *args, **kwargs):
        return None


class SkipUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch") or module.startswith("numpy") or module == "collections":
            if module == "collections" and name in ("OrderedDict", "defaultdict"):
                return super().find_class(module, name)
            return _Stub
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub

    def persistent_load(self, pid):
        return _Stub(pid)


def summarize(obj, prefix=""):
    if isinstance(obj, dict):
        skip = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
        }
        for k, v in obj.items():
            if k in skip:
                n = len(v) if hasattr(v, "__len__") else "?"
                print(f"{prefix}{k}: <weights, {n} entries>")
            else:
                print(f"{prefix}{k}:")
                summarize(v, prefix + "  ")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} len={len(obj)}")
        if obj and not isinstance(obj[0], (bytes, bytearray)):
            summarize(obj[0], prefix + "  [0] ")
    else:
        text = repr(obj)
        if len(text) > 400:
            text = text[:400] + "..."
        print(f"{prefix}{text}")


def main():
    srcs = find_sources()
    if not srcs:
        print("no sources")
        return
    for path in srcs:
        print("=" * 60)
        print("FILE", path)
        print("SIZE", os.path.getsize(path))
        try:
            name, blob, total = extract_datapkl(path)
            print("ENTRY", name, "pkl_bytes", len(blob), "file_size", total)
            obj = SkipUnpickler(io.BytesIO(blob)).load()
            print("TYPE", type(obj))
            summarize(obj)
        except Exception as e:
            print("FAIL", type(e).__name__, e)


if __name__ == "__main__":
    main()
