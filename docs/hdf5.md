# HDF5 notes
## Data layout

A `dict` (or a nested model) is a group; everything else is a dataset.
Stored natively are booleans, strings and numeric lists/arrays.
Other types (`None`, a list of dicts, a list of mixed types) are stored as a YAML string.

`dict` keys are percent-encoded in HDF5, e.g. so that `/` in a key is not interpreted as a HDF5 nested group.

Strings are stored as UTF-8.
