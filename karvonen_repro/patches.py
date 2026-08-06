"""The only four places our harness reaches into Karvonen's pipeline.

`full_pipeline.analyze_sae_groups` runs verbatim. It assumes four things about its
environment that are not true for us, and each rebind below fixes exactly one:

  load_dictionary     he instantiates the SAE himself from ae.pt keyed on
                      config.json's dict_class. Ours can be any Dictionary subclass,
                      so the loader owns instantiation.
  get_nested_folders  he globs a group directory for every checkpoint under it. We
                      evaluate an explicit list, one checkpoint at a time, so a crash
                      costs one checkpoint instead of the group.
  deque               `analyze_sae_groups` builds its device stack as
                      `deque(f"cuda:{i}" ...)`. Rebinding the name is the only way to
                      run his code unmodified on CPU. On GPU with N_GPUS=1 the result
                      is identical to his.
  evaluate            `get_evals` calls it without `n_batches`, so every published
                      SAE-quality column is one batch of 25 activation vectors
                      (PROTOCOL.md section 2). eval_n_batches=1 is his protocol
                      exactly; anything larger is measurement mode.

Nothing else in circuits/ is touched. Each rebind is a divergence risk; there are
four, and this docstring is the list.
"""

import collections
import functools
import json
from typing import Callable

import circuits.dictionary_learning.utils as dictionary_utils
import circuits.eval_sae_as_classifier as eval_sae
import circuits.evaluation as evaluation
import circuits.full_pipeline as full_pipeline
import circuits.get_eval_results as get_eval_results

LoadSAE = Callable[[str, str], tuple]


def bind(load_sae: LoadSAE, checkpoint_dirs: list[str], device: str, eval_n_batches: int) -> None:
    """checkpoint_dirs must be absolute and end in "/" — his string concatenation
    (`f"{autoencoder_path}config.json"`) supplies no separator of its own."""
    assert all(d.endswith("/") for d in checkpoint_dirs), checkpoint_dirs

    def load_dictionary(base_path: str, load_device: str) -> tuple:
        ae, meta = load_sae(base_path, load_device)
        with open(f"{base_path}config.json") as f:
            d_submodule = json.load(f)["buffer"]["d_submodule"]
        assert ae.activation_dim == d_submodule, (
            f"{base_path}: loader returned activation_dim={ae.activation_dim} but the "
            f"pipeline will feed it d_submodule={d_submodule} activations"
        )
        print(f"loaded {base_path} -> {meta}")
        return ae, meta

    dictionary_utils.load_dictionary = load_dictionary
    eval_sae.get_nested_folders = lambda _path: list(checkpoint_dirs)
    full_pipeline.deque = lambda _devices: collections.deque([device])
    get_eval_results.evaluate = functools.partial(evaluation.evaluate, n_batches=eval_n_batches)
