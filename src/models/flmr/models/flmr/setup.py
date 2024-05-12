from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="segmented_maxsim_cpp",
    ext_modules=[
        cpp_extension.CppExtension(
            "segmented_maxsim_cpp",  # Extension name
            ["segmented_maxsim.cpp"],  # Source files
            extra_compile_args=["-std=c++17"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
