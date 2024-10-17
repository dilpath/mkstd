In this example we:
1. define a data format standard for ML models and save the schema to `standard/schema.yaml`. See `petab_sciml_standard.py`.
2. create an ML model in pytorch
3. convert that pytorch model into a PEtab SciML ML model and store it to disk (see `data/models0.yaml`)
4. read the model from disk, reconstruct the pytorch model, then convert that reconstructed pytorch model back into PEtab SciML, and store it to disk once more (see `data/models1.yaml`)
5. verify that the round-trip including both the PEtab SciML ML model standard and pytorch worked
