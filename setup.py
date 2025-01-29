from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='custom_sycl_op',
    ext_modules=[
        CppExtension(
            'custom_sycl_op',
            ['custom_sycl_op.cpp'],
            extra_compile_args=['-fsycl'],
            extra_link_args=['-fsycl']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)