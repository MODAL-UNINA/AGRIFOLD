
# %%

import sys

# %%

_interactive_mode = "ipykernel_launcher" in sys.argv[0] or (
    len(sys.argv) == 1 and sys.argv[0] == ""
)

if _interactive_mode:
    from tqdm.auto import tqdm, trange
else:
    from tqdm import tqdm, trange


def is_interactive():
    return _interactive_mode


# %%

__all__ = ["is_interactive"]
