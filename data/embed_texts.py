"""Sentence embeddings of text files with all-MiniLM-L6-v2 (384 dimensions).

Paper: Section VI-A, Datasets ("embedded with all-MiniLM-L6-v2 into 384 dimensions").

The model is loaded with plain transformers: mean pooling over the attention mask, then L2 normalization (the model's
own pipeline), at most 256 tokens, batches of 512 texts formed after sorting the texts of a file by length, float16
autocast on the device. Each input file is one call: its batches, and so the float16 rounding of a text, depend on the
other texts of the file. Output: float32 (n, 384), next to each input with the prefix texts_ / titles_ replaced by emb_
unless --out is given.

Usage:
    python data/embed_texts.py --model RAW/models/all-MiniLM-L6-v2 --texts PREP/amazon/texts_items.parquet
"""
import argparse
import os
import time

import numpy as np


def embed(texts, tok, model, device, bs=512, max_len=256, log=print):
    import torch
    N = len(texts)
    # sort by length for throughput, restore order afterwards
    order = np.argsort([len(t) for t in texts], kind="stable")
    E = np.zeros((N, 384), np.float32)
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device.split(":")[0], dtype=torch.float16):
        for s in range(0, N, bs):
            idx = order[s:s + bs]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(v, dim=1)
            E[idx] = v.cpu().numpy()
            if (s // bs) % 200 == 0:
                log(f"{s}/{N} {time.time()-t0:.0f}s", flush=True)
    return E


def default_out(path):
    d, f = os.path.split(path); stem = os.path.splitext(f)[0]
    for pre in ("texts_", "titles_"):
        if stem.startswith(pre):
            stem = stem[len(pre):]
    return os.path.join(d, "emb_" + stem + ".npy")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="directory with the files of sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--texts", nargs="+", required=True, help="parquet files with a column `text`; each file is one call")
    ap.add_argument("--out", nargs="*", default=None, help="output .npy per input file")
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    a = parse_args()
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModel, AutoTokenizer
    outs = a.out or [default_out(f) for f in a.texts]; assert len(outs) == len(a.texts)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(a.device).eval()
    for f, o in zip(a.texts, outs):
        if os.path.exists(o):
            print("exists", o, flush=True); continue
        texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
        E = embed(texts, tok, model, a.device, a.bs, a.max_len)
        np.save(o, E); print("saved", o, E.shape, flush=True)


if __name__ == "__main__":
    main()
