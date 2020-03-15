
"""
Motivation: I have been able to reproduce the unconditional generation
model's loss numbers (SketchRNN decoder only) on the pig category when
training on the npz dataset (originally used in ndjson). However,
the loss numbers are much higher (0.1 vs -0.6) when training on the ndjson
dataset. This investigates that.

Turns out, the ndjson dataset is a simplified version of the npz drawings, as
the average length is shorter for ndjson than npz. This despite the
website saying that they both use (RDP with epsilon=2 to simplify).
https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm.
(I had originally used the ndjson dataset because it contains extra metadata,
such as whether or not the drawing was recognized by a trained classifier
during QuickDraw.)

Nevertheless, the generated drawings for a npz model vs a ndjson model seem
similar though.... If anything the ndjson is slightly better? I'm not sure
why the loss numbers vary so much though. It's okay.
"""

from src.models.base.stroke_models import NpzStrokeDataset, NdjsonStrokeDataset
from src.data_manager.quickdraw import normalize_strokes
import numpy as np

def compare():

    # cat = 'pig'
    cat = 'bear'
    npz_ds = NpzStrokeDataset(cat, 'train')
    n = 10000
    ndjson_ds = NdjsonStrokeDataset(cat, 'train', max_per_category=n)

    print(len(npz_ds.data))
    print(len(ndjson_ds.data))

    z_strokes = [d['stroke3'] for d in npz_ds.data]
    j_strokes = [d['stroke3'] for d in ndjson_ds.data]

    # Average deta x / delta y: these are about the same
    print(np.mean([z.mean() for z in z_strokes]))
    print(np.mean([j.mean() for j in j_strokes]))
    print()

    # number of penups: these are about the same
    print(np.mean([z[:,2].sum() for z in z_strokes]))
    print(np.mean([j[:,2].sum() for j in j_strokes]))
    print()

    # Length of stroke: these differ!
    print(np.mean([len(z) for z in z_strokes]))
    print(np.mean([len(j) for j in j_strokes]))
    print()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    compare()