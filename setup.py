from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='berkelib',
    version='0.0.1',
    description='Pytorch Extension Library of Optimized Sparse Matrix multiplication',
    author='Berke Kisin',
    author_email="kisinberke@gmail.com",
    python_requires='>=3.8',
    ext_modules=[
          #CppExtension('berkelib_cpu', [
          #     'berkelib.cpp',
          #     'cpu/berkelib_cpu.cpp',
          #]),
          CUDAExtension('berkelib', [
                'berkelib.cpp',
                'cuda/berkelib_cuda.cu'
          ])

    ],
    cmdclass={
          'build_ext': BuildExtension
    }
)