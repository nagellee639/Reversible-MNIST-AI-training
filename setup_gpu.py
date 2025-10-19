import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys # Import sys

# A custom build extension command that runs CMake
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required CMake arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}'
        ]
        
        # --- THE FIX IS HERE ---
        # The build command is "cmake --build .", not just "--build ."
        build_args = ['cmake', '--build', '.']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake and Make
        print("--- Running CMake ---")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        print("--- Running Make (Build Step) ---")
        subprocess.check_call(build_args, cwd=self.build_temp)

setup(
    name='fast_network_gpu',
    version='2.0.1', # Bump version
    author='Your Name',
    description='GPU-accelerated reversible network simulator',
    ext_modules=[CMakeExtension('fast_network_gpu')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)

