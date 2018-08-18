from distutils.core import setup, Extension

module1 = Extension('helloworld',
                    include_dirs = ['/usr/local/include'],
                    libraries = ['opencv_core', 'opencv_highgui', 'opencv_imgproc'],
                    library_dirs = ['/usr/lib','/usr/local/lib'],
                    sources = ['bind.cpp', "libmypy.cpp"])


setup(
	name = "helloworld",
	version = "1.0",
	ext_modules = [module1]
	)
