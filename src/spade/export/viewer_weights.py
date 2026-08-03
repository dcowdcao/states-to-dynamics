"""Export the model and each SAE to the flat fp16 format the browser viewer reads.

The viewer runs its own forward pass in JavaScript, so this is the only seam between the
research code and the deployed page: it reads the registry and the checkpoints, and writes
weights plus a manifest and a reference output for verification.

It is research code — it dispatches on trainer class, walks state dicts per SAE kind, and
depends on the standardization contract — which is why it lives in the package rather than
next to the JavaScript.

Two invariants. The per-kind tensor-name maps key on the autoencoder *class names*, so a
class rename breaks the export with a KeyError rather than a wrong number. And
standardization is baked in here and re-applied in the browser: three independent
implementations of scale*(x-mean) exist (adapter, exporter, JS), and
viewer/web/test_inference.mjs against the exported reference.json is the only thing
keeping them in agreement. Run it before every upload.
"""

# KIND_BY_CLASS = {}   # autoencoder class name -> encoder kind understood by the JS
# def export_model(...), export_sae(...), write_manifest(...): ...
# def main(): ...      # -> viewer/web/weights/{model.bin, sae_*.bin, manifest.json, reference.json}
