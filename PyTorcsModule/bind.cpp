#include <Python.h>
#include "libmypy.h"

PyObject * hello(PyObject * self) {
    ////////
    // allocate space for 3 pointers to strings
    char **strings = (char**)malloc(3*sizeof(char*));
    int i = 0;
    // allocate space for each string
    // here allocate 50 bytes, which is more than enough for the strings
    for(i = 0; i < 3; i++){
        printf("%d\n", i);
        strings[i] = (char*)malloc(150*sizeof(char));
    }
    //assign them all something
    sprintf(strings[0], "/home/bcalmeida/dev/tg/DeepDriving/Caffe_driving/torcs/pre_trained/driving_run_1F.prototxt");
    sprintf(strings[1], "/home/bcalmeida/dev/tg/DeepDriving/Caffe_driving/torcs/pre_trained/driving_train_1F_iter_140000.caffemodel");
    sprintf(strings[2], "GPU");

    torcs_run(1, strings);
    ////////

    Py_RETURN_NONE;
}

char hellofunc_docs[] = "Hello world description.";

PyMethodDef helloworld_funcs[] = {
	{	"hello",
		(PyCFunction)hello,
		METH_NOARGS,
		hellofunc_docs},
	{	NULL}
};

char helloworldmod_docs[] = "This is hello world module.";

PyModuleDef helloworld_mod = {
	PyModuleDef_HEAD_INIT,
	"helloworld",
	helloworldmod_docs,
	-1,
	helloworld_funcs,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC PyInit_helloworld(void) {
	return PyModule_Create(&helloworld_mod);
}
