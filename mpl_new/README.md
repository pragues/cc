## Notations on Pybindings

pybind11 tutorial [link](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#arrays)


## Python vs. cpp class and methods lookup table


## Exmaple Usage

Requirements:
1. Conda env. See `requirements.txt`. Generally uses 3.10 python version here to fit pybind11.
```
conda create -n mpl python=3.10
pip install -r requirements
```
2. Build the docker env from the docker folder. 
3. motion promitive library from this [link](https://github.com/sikang/motion_primitive_library/tree/master) 
4. Generate corresponding dataset. 

Compile by: 
```bash
set -e

rm -rf build
mkdir build && cd build
cmake ..
make -j1
cd ..

./build/test_planner_2d ./data/corridor.yaml test_planner_2d_circle2

cd test
python test.py
cd ..
```


Instead of using `MapReader` class, directly use `pyyaml` to tread the `corridor.yaml`

By adding `bindings.cpp` under `motion_primitive_library/src/python/bindings`
