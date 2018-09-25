from distutils.core import setup, Extension
#import numpy.distutils.misc_util

# setup(
#     ext_modules=[Extension("greet", ["greetmodule.cpp"])],
#     )

module1 = Extension('greet',
                    include_dirs = ['/usr/local/include'],
                    libraries = ['opencv_core', 'opencv_highgui', 'opencv_imgproc'],
                    library_dirs = ['/usr/lib','/usr/local/lib'],
                    sources = ['greetmodule.cpp'])

setup (
       ext_modules = [module1])


"""
flags used on caffe make:
-I.build_release/src -I./src -I./include

-L/usr/lib -L/usr/local/lib -L/usr/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart -lcublas -lcurand -lpthread -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lcblas -latlas

root caffe dir:
/home/bcalmeida/dev/tg/DeepDriving/Caffe_driving/

flags needed:
include:
/home/bcalmeida/dev/tg/DeepDriving/Caffe_driving/.build_release/src
/home/bcalmeida/dev/tg/DeepDriving/Caffe_driving/src
/home/bcalmeida/dev/tg/DeepDriving/Caffe_driving/include

library:
"""