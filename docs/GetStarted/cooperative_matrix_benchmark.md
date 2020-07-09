# Getting Started building experiemental ModelBuilder

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.

    * This document parallels getting_started_windows_cmake.md and
      getting_started_macos_bazel.md
      Please keep them in sync.
-->

This is a modified version of IREE Getting started instructions to be able to
build experimental matrix multiplication benchmark.

## Prerequisites

### Install CMake

IREE uses CMake version `>= 3.13`. First try installing via your distribution's
package manager and verify the version:

```shell
$ sudo apt install cmake
$ cmake --version # >= 3.13
```

Some package managers (like `apt`) distribute old versions of cmake. If your
package manager installs a version `< 3.13`, then follow the installation
instructions [here](https://cmake.org/install/) to install a newer version (e.g.
the latest).

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Your editor of choice likely has plugins for CMake,
> such as the Visual Studio Code
> [CMake Tools](https://github.com/microsoft/vscode-cmake-tools) extension.

### Install Ninja

[Ninja](https://ninja-build.org/) is a fast build system that you can use as a
CMake generator. Follow Ninja's
[installing documentation](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages).

### Install Vulkan SDK

Install the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) and run
setup-env.sh script to set environment variables.
```shell
$ sh setup-env.sh
```

### Install a Compiler

We recommend Clang. GCC is not fully supported.

```shell
$ sudo apt install clang
```

## Clone and Build

### Clone

Clone the repository and initialize its submodules:

```shell
$ git clone https://github.com/ThomasRaoux/iree.git
$ cd iree
$ git submodule update --init
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Editors and other programs can also clone the
> repository, just make sure that they initialize the submodules.

### Build

Configure:

```shell
$ cmake -G Ninja -B build/ -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DIREE_BUILD_EXPERIMENTAL=ON .
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;The root
> [CMakeLists.txt](https://github.com/google/iree/blob/main/CMakeLists.txt)
> file has options for configuring which parts of the project to enable.<br>
> &nbsp;&nbsp;&nbsp;&nbsp;These are further documented in [CMake Options and Variables](cmake_options_and_variables.md).

Build all targets:

```shell
$ cmake --build build/
```

# Run matrix multiply benchmark
```shell
$ ./build/experimental/ModelBuilder/test/bench-matmul-gpu -vulkan-wrapper=build/third_party/llvm-project/llvm/lib/libvulkan-runtime-wrap
pers.so
```

Or to check correctness
```shell
$ ./build/experimental/ModelBuilder/test/bench-matmul-gpu -vulkan-wrapper=build/third_party/llvm-project/llvm/lib/libvulkan-runtime-wrap
pers.so -correctness
```






