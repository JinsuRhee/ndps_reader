# ndps_reader

Reads the catalog files in an NDPS catalog download (`<job ID>.zip`).

| File | Contents | Reader |
|---|---|---|
| `galaxy_NNNN.pkl` / `.txt` (`halo_` for halos) | One row per object, properties as columns | `read_catalog` |
| `galaxy_pid_NNNN.pkl` / `.txt` | Member particle IDs per object | `read_members` |
| `snapinf_NNNN.txt` | Snapshot metadata (e.g. `aexp`) | `read_snapinfo` |

## Install

`ndps_reader.py` is a single file that needs only Python 3.7+ and NumPy.
Copy it next to your script, or install it:

```bash
pip install .            # from this directory
```

## Usage

```python
import ndps_reader as nr

cat = nr.read_catalog('galaxy_0680.pkl')    # or galaxy_0680.txt
print(cat.dtype.names)                      # ('ID', 'Mass_tot', 'SFR', ...)
massive = cat[cat['Mass_tot'] > 1e11]
sfr = cat['SFR']                            # vector properties keep their shape, (N, 15)

pid = nr.read_members('galaxy_pid_0680.txt')
pid['P_ID'][0]                              # particle IDs of the first object

info = nr.read_snapinfo('snapinf_0680.txt') # {'aexp': 0.5838..., ...}

nr.read('halo_0100.txt')                    # picks the reader from the file name
```

## Examples

[`examples/01_catalog.ipynb`](examples/01_catalog.ipynb) is a step-by-step notebook
covering every file in a download:

- `galaxy_NNNN` / `halo_NNNN` catalogs (sections 2–8)
- `galaxy_pid_NNNN` member particle IDs (section 4-(2))
- `snapinf_NNNN.txt` and `error.txt` (section 9)

## Notes

- Pickle and ASCII files hold the same values. Pickle keeps the original dtypes
  (e.g. float32). ASCII stores no dtypes, so a column whose values are all
  integers is read as int64 and any other column as float64. A float property
  whose values all happen to be whole numbers (often `Mass_tot`) is therefore
  read as int64. The values are still exact, but cast with `.astype(float)` if
  you need floats.
- Pickle files are loaded with a restricted unpickler that only rebuilds NumPy
  arrays, so a tampered file cannot run code. Pass `trusted=True` to use plain
  `pickle.load`.
- Property names are the HDF5 `G_Prop` dataset names without the `G_` prefix.
  In ASCII, vector properties are written as `SFR[0]`, `SFR[1]`, ... and are
  joined back into one field when read.
