# PyDeepDriving

### Setup

We recommend using anaconda and using the provided `environment.yml`. Fastai library is required.

### Visual Studio Code

Some tips for easier development of Python C extensions on Visual Studio Code:

- `.vscode/settings.json`
    - set conda python as interpreter, e.g. `"python.pythonPath": ".../anaconda/envs/PyDeepDriving/bin/python"`
    - exclude system python, e.g. `"files.exclude: { ..., /usr/include/python2.7": true }`, to avoid looking for `<Python.h>` on two places
- `c_cpp_properties.json`
    - add conda Python to include and browse path, e.g. `".../anaconda/envs/PyDeepDriving/include/python3.6m",`