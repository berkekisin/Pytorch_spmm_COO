from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='scatter_edge',
    version='1.0',
    description='Pytorch Extension Library of Optimized Sparse Matrix multiplication',
    author='Berke Kisin',
    author_email="kisinberke@gmail.com",
    python_requires='>=3.8',
    ext_modules=[
          #CppExtension('berkelib_cpu', [
          #     'berkelib.cpp',
          #     'cpu/berkelib_cpu.cpp',
          #]),
          CUDAExtension('scatter_edge', [
                'scatter_edge.cpp',
                'cuda/scatter_edge_cuda.cu'
          ])

    ],
    cmdclass={
          'build_ext': BuildExtension
    }
)