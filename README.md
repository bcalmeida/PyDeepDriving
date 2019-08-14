# PyDeepDriving

This project is an extension of Princeton [DeepDriving](http://deepdriving.cs.princeton.edu/). This an implementation of a driving agent using vision with innovative direct perception technique and an extension of the original system. You may now use Python and PyTorch to interface with [TORCS](http://torcs.sourceforge.net/).

The paper for this project, [Autonomous Driving on a Direct Perception System with Deep Recurrent Layers](https://dl.acm.org/ft_gateway.cfm?id=3309790&ftid=2071587&dwn=1&CFID=149835448&CFTOKEN=ea96ef396449373b-A54CD359-E1CE-E7B7-DA198E3B1306F2B2), was presented at the 2nd International Conference on Applications of Intelligent Systems (APPIS '19) in Las Palmas de Gran Canaria, Spain.

### Demo

![](demo.gif)


### Setup

We recommend using anaconda and using the provided `environment.yml`. Fastai library is required.

### Visual Studio Code

Some tips for easier development of Python C extensions on Visual Studio Code:

- `.vscode/settings.json`
    - set conda python as interpreter, e.g. `"python.pythonPath": ".../anaconda/envs/PyDeepDriving/bin/python"`
    - exclude system python, e.g. `"files.exclude: { ..., /usr/include/python2.7": true }`, to avoid looking for `<Python.h>` on two places
- `c_cpp_properties.json`
    - add conda Python to include and browse path, e.g. `".../anaconda/envs/PyDeepDriving/include/python3.6m",`

