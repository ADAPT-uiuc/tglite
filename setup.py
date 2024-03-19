from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def main():
    curr_dir = Path(__file__).absolute().parent
    setup(
        ext_modules=[
            CppExtension(
                name="tglite._c",
                sources=[
                    "lib/bind.cpp",
                    "lib/cache.cpp",
                    "lib/dedup.cpp",
                    "lib/sampler.cpp",
                    "lib/tcsr.cpp",
                    "lib/utils.cpp"
                ],
                include_dirs=[curr_dir/"include/"],
                extra_compile_args=["-std=c++14", "-fopenmp"],
                extra_link_args=["-fopenmp"]
            )],
        cmdclass={
            "build_ext": BuildExtension
        })


if __name__ == "__main__":
    main()
