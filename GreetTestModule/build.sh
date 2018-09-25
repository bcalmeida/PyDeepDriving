#gcc -fpic --shared $(python-config --includes) greetmodule.cpp -o greetmodule.so
python3 setup.py build_ext --inplace

