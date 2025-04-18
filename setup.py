import os
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

VERSION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "stable_retro/VERSION.txt",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
README = open(os.path.join(SCRIPT_DIR, "README.md")).read()


class CMakeBuild(build_ext):
    def run(self):
        suffix = super().get_ext_filename("")
        pyext_suffix = f"-DPYEXT_SUFFIX={suffix}"
        pylib_dir = ""
        if not self.inplace:
            pylib_dir = f"-DPYLIB_DIRECTORY={self.build_lib}"
        if self.debug:
            build_type = "-DCMAKE_BUILD_TYPE=Debug"
        else:
            build_type = ""

        # Provide hints to CMake about where to find Python (this should be enough for most cases)
        python_root_dir = f"-DPython_ROOT_DIR={os.path.dirname(sys.executable)}"
        python_find_strategy = "-DPython_FIND_STRATEGY=LOCATION"

        # These directly specify Python artifacts
        python_executable = f"-DPython_EXECUTABLE={sys.executable}"
        python_include_dir = f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}"
        python_library = f"-DPython_LIBRARY={sysconfig.get_path('platlib')}"

        subprocess.check_call(
            [
                "cmake",
                ".",
                "-G",
                "Unix Makefiles",
                build_type,
                pyext_suffix,
                pylib_dir,
                python_root_dir,
                python_find_strategy,
                python_executable,
                python_include_dir,
                python_library,
            ],
        )
        if self.parallel:
            jobs = f"-j{self.parallel:d}"
        else:
            import multiprocessing

            jobs = f"-j{multiprocessing.cpu_count():d}"

        subprocess.check_call(["make", jobs, "retro"])


platform_globs = [
    "*-%s/*" % plat
    for plat in [
        "Nes",
        "Snes",
        "Genesis",
        "Atari2600",
        "GameBoy",
        "Sms",
        "GameGear",
        "PCEngine",
        "GbColor",
        "GbAdvance",
        "32x",
        "Saturn",
    ]
]


setup(
    name="stable-retro",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Farama Foundation",
    author_email="contact@farama.org",
    url="https://github.com/farama-foundation/stable-retro",
    version=open(VERSION_PATH).read().strip(),
    license="MIT",
    install_requires=["gymnasium>=0.27.1", "pyglet>=1.3.2,==1.*", "farama-notifications>=0.0.1", "wandb==0.16.4", "colorist==1.1.2", "matplotlib==3.8.2", "pandas==2.2.0", "sb3-contrib==2.2.1", "torchvision==0.18.0", "tensorboard==2.16.2", "opencv-python==4.9.0.80", "sample_factory==2.1.1", "scipy==1.12.0", "signal-slot-mp==1.0.5", "colorlog==6.9.0", "tensorboardX==2.6.2.2", "imageio==2.34.0"],
    python_requires=">=3.8.0,<3.13",
    ext_modules=[Extension("stable_retro._retro", ["CMakeLists.txt", "src/*.cpp"])],
    cmdclass={"build_ext": CMakeBuild},
    packages=[
        "stable_retro",
        "stable_retro.data",
        "stable_retro.data.stable",
        "stable_retro.data.experimental",
        "stable_retro.data.contrib",
        "stable_retro.scripts",
        "stable_retro.import",
        "stable_retro.examples",
        "stable_retro.testing",
    ],
    package_data={
        "stable_retro": [
            "cores/*.json",
            "cores/*_libretro*",
            "VERSION.txt",
            "README.md",
            "LICENSES.md",
        ],
        "stable_retro.data.stable": platform_globs,
        "stable_retro.data.experimental": platform_globs,
        "stable_retro.data.contrib": platform_globs,
    },
)
