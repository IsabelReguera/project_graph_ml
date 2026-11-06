"""Check that the refactored models are identical to the LIRE-2026 ones (same init, same output)."""
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent / "lire_ref" / "src"))   # old code (bare imports)

from graphmatchingnetwork import GraphMatchingNetwork as OldGMN
from baseline import BaselineGNN as OldBaseline
from gmnbind.models.gmn import GraphMatchingNetwork as NewGMN
from gmnbind.models.baseline import BaselineGNN as NewBaseline
from gmnbind.data.complex_dataset import ComplexDataset, complex_collate_fn

# same hyperparameters for old and new (take them from configs/gmn.yaml)
kw = dict(token_dim=64, node_embedding_dims=[32, 16], edge_embedding_dims=[32, 16],
          num_layers=2, dropout_p=0.12, normalization=True)

# a small real batch from the processed data
ds = ComplexDataset(REPO_ROOT / "data/processed/v1", ["1a0t", "1a1b", "1a1c"])
prot, lig = complex_collate_fn([ds[i] for i in range(len(ds))])


def compare(name, OldCls, NewCls):
    torch.manual_seed(0); old = OldCls(**kw)
    torch.manual_seed(0); new = NewCls(**kw)
    sd_old, sd_new = old.state_dict(), new.state_dict()

    same_keys = list(sd_old.keys()) == list(sd_new.keys())
    same_weights = same_keys and all(torch.equal(sd_old[k], sd_new[k]) for k in sd_old)

    old.eval(); new.eval()                       # no dropout
    with torch.no_grad():
        out_old = old(prot, lig)
        out_new = new(prot, lig)
    same_output = torch.equal(out_old, out_new)

    print(f"{name:9s} | keys: {same_keys} | weights: {same_weights} | output: {same_output}")
    if not same_keys:
        print("  only in old:", set(sd_old) - set(sd_new))
        print("  only in new:", set(sd_new) - set(sd_old))


compare("GMN", OldGMN, NewGMN)
compare("Baseline", OldBaseline, NewBaseline)