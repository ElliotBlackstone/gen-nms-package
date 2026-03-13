#include <Python.h>

extern "C" PyObject* PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",
        nullptr,
        -1,
        nullptr,
    };
    return PyModule_Create(&module_def);
}
