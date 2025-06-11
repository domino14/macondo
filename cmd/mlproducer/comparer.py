import struct
import numpy as np


def read_vector(f):
    len_hdr = f.read(4)
    if not len_hdr or len(len_hdr) < 4:
        return None
    (n_bytes,) = struct.unpack("<I", len_hdr)
    payload = f.read(n_bytes)
    if len(payload) != n_bytes:
        return None
    vec = np.frombuffer(payload, dtype=np.float32)
    return vec


file1 = "out-10000-new.dat"
file2 = "out-10000-old.dat"

with open(file1, "rb") as f1, open(file2, "rb") as f2:
    idx = 0
    while True:
        v1 = read_vector(f1)
        v2 = read_vector(f2)
        if v1 is None or v2 is None:
            print(f"End of file reached at vector {idx}.")
            break
        if len(v1) != len(v2):
            print(f"Vector {idx}: lengths differ: {len(v1)} vs {len(v2)}")
            break
        diffs = []
        for i, (a, b) in enumerate(zip(v1, v2)):
            if not np.isclose(a, b, atol=1e-6):
                diffs.append((i, a, b))
        if diffs:
            print(f"Vector {idx} differs at {len(diffs)} indices:")
            print(f"  Lengths: {len(v1)} vs {len(v2)}")
            for i, a, b in diffs:
                print(f"  Index {i}: new={a}  old={b}")
            break
        idx += 1
    else:
        print("All vectors compared are identical (within tolerance).")
