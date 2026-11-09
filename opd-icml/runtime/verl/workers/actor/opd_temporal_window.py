"""Exact fixed-bank temporal accumulation. CPU only, no model calls or RNG."""
import numpy as np


class TemporalWindow:
    def __init__(self):
        self.start = self.last = None
        self.anchor = self.previous = None
        self.path = dict(raw=0., unit=0.)

    @staticmethod
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)

    def observe(self, step, states):
        # References to immutable CPU tensors retained by the existing replay.
        xs = [x.detach().numpy() if hasattr(x, 'detach') else np.asarray(x) for x in states]
        if self.start is None:
            if step % 10 != 0:
                raise ValueError('exact temporal window must resume at a ten-step checkpoint')
            self.start = self.last = step
            self.anchor = self.previous = xs
            return {}
        if step != self.last + 1:
            raise ValueError('noncontiguous temporal window')
        n = sum(len(x) for x in xs)
        net = dict(raw=0., unit=0.)
        velocity = dict(raw=0., unit=0.)
        for x, before, anchor in zip(xs, self.previous, self.anchor):
            if x.shape != before.shape or x.shape != anchor.shape:
                raise ValueError('fixed diagnostic bank changed')
            for name, a, b, c in [('raw', x, before, anchor),
                                  ('unit', self.unit(x), self.unit(before), self.unit(anchor))]:
                velocity[name] += np.square(a-b, dtype=np.float64).sum()
                net[name] += np.square(a-c, dtype=np.float64).sum()
        out = dict(temporal_start=np.asarray(self.start), temporal_end=np.asarray(step),
                   temporal_steps=np.asarray(step-self.start), temporal_tokens=np.asarray(n))
        for name in ('raw', 'unit'):
            self.path[name] += float(np.sqrt(velocity[name]/n))
            net_rms = float(np.sqrt(net[name]/n))
            out['temporal_'+name+'_net_rms'] = np.asarray(net_rms)
            out['temporal_'+name+'_path_rms'] = np.asarray(self.path[name])
            out['temporal_'+name+'_accumulation'] = np.asarray(
                net_rms/self.path[name] if self.path[name] > 0 else np.nan)
        self.previous = xs
        self.last = step
        if step % 10 == 0:
            self.start = step
            self.anchor = xs
            self.path = dict(raw=0., unit=0.)
        return out
