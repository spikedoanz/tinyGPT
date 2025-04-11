def invert_dict(d): return {v: k for k, v in reversed(d.items())}
def dedup_dict(d): return invert_dict(invert_dict(d))
