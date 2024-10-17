In this example we:
1. define a data format standard for ML models and save the schema to `standard/schema.yaml`. See `petab_sciml_standard.py`.
2. create an ML model in pytorch
3. convert that pytorch model into a PEtab SciML ML model and store it to disk (see `data/models0.yaml`)
4. read the model from disk, reconstruct the pytorch model, then convert that reconstructed pytorch model back into PEtab SciML, and store it to disk once more (see `data/models1.yaml`)

In total, this means we do:
```
pytorch model
-> petab sciml model
-> petab sciml yaml
-> petab sciml model
-> pytorch model
-> petab sciml model
-> petab sciml yaml
```
and then verify that the two YAML files match.


TODO: check that the original pytorch forward call provides that same output as the reconstructed pytorch forward call, for some different inputs.
