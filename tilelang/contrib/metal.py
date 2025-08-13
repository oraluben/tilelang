# pylint: disable=invalid-name
"""Utility to invoke Metal compiler in the system"""
from __future__ import absolute_import as _abs

import os
import subprocess
import warnings
import sys

import tvm.ffi
from tvm.target import Target

from tvm.base import py_str
from tvm.contrib import utils


def is_metal_available():
    """Check if Metal tools are available on the system.
    
    Returns
    -------
    available : bool
        True if Metal tools are available, False otherwise.
    """
    try:
        # Try to find Metal tools using xcrun
        subprocess.check_output(["xcrun", "-sdk", "macosx", "-find", "metal"],
                               stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If xcrun fails, try common paths
        common_paths = [
            "/usr/bin/metal",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal"
        ]
        for path in common_paths:
            if os.path.exists(path):
                return True
        return False


def compile_metal(code,
                  target_format="metallib",
                  arch=None,
                  options=None,
                  path_target=None,
                  verbose=False):
    """Compile Metal code with Metal compiler from env.

    Parameters
    ----------
    code : str
        The Metal code.

    target_format : str
        The target format of Metal compiler.

    arch : str
        The Metal architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    verbose : bool, optional
        Whether to print the compilation command and output.

    Return
    ------
    metallib : bytearray
        The bytearray of the metallib
    """
    # Check if Metal is available
    if not is_metal_available():
        raise RuntimeError(
            "Metal tools are not available. Please ensure Xcode command line tools are installed."
        )
    
    if arch is None:
        # If None, then it will use a default value
        arch = "default"

    temp = utils.tempdir()
    file_name = "tvm_kernels"
    if target_format not in ["metallib", "air"]:
        raise ValueError("target_format must be in metallib, air")
    temp_code = temp.relpath(f"{file_name}.metal")
    temp_target = temp.relpath(f"{file_name}.{target_format}")

    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    kernels_output_dir = (pass_context.config.get("metal.kernels_output_dir", None))
    if kernels_output_dir is not None:
        if not os.path.isdir(kernels_output_dir):
            os.makedirs(kernels_output_dir)
        temp_code = os.path.join(kernels_output_dir, f"{file_name}.metal")
        temp_target = os.path.join(kernels_output_dir, f"{file_name}.{target_format}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    
    # Metal compilation requires two steps:
    # 1. Compile .metal to .air using metal command
    # 2. Compile .air to .metallib using metallib command
    
    temp_ir = temp.relpath(f"{file_name}.air")
    if kernels_output_dir is not None:
        temp_ir = os.path.join(kernels_output_dir, f"{file_name}.air")
    
    # Step 1: Compile Metal to AIR
    cmd1 = ["xcrun", "-sdk", "macosx", "metal", "-O3"]
    if options:
        if isinstance(options, str):
            cmd1 += [options]
        elif isinstance(options, list):
            cmd1 += options
        else:
            raise ValueError("options must be str or list of str")
    
    cmd1 += ["-c", temp_code, "-o", temp_ir]
    
    # Step 2: Compile AIR to Metallib
    cmd2 = ["xcrun", "-sdk", "macosx", "metallib"]
    cmd2 += [temp_ir, "-o", file_target]
    
    if verbose:
        print("Step 1: Compiling Metal to AIR")
        print(" ".join(cmd1))
    
    try:
        proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out1, _) = proc1.communicate()
    except FileNotFoundError:
        raise RuntimeError("Metal compiler not found. Please ensure Xcode command line tools are installed.")
    
    if verbose:
        print(py_str(out1))
    
    if proc1.returncode != 0:
        msg = f"{code}\n" \
            f"Compilation error (step 1 - Metal to AIR):\n" \
            f"{py_str(out1)}\n" \
            f"Command: {' '.join(cmd1)}\n"
        raise RuntimeError(msg)
    
    if verbose:
        print("Step 2: Compiling AIR to Metallib")
        print(" ".join(cmd2))
    
    try:
        proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out2, _) = proc2.communicate()
    except FileNotFoundError:
        raise RuntimeError("Metallib tool not found. Please ensure Xcode command line tools are installed.")
    
    if verbose:
        print(py_str(out2))
    
    if proc2.returncode != 0:
        msg = f"{code}\n" \
            f"Compilation error (step 2 - AIR to Metallib):\n" \
            f"{py_str(out2)}\n" \
            f"Command: {' '.join(cmd2)}\n"
        raise RuntimeError(msg)

    try:
        with open(file_target, "rb") as f:
            data = bytearray(f.read())
            if not data:
                raise RuntimeError("Compilation error: empty result is generated")
            return data
    except FileNotFoundError:
        raise RuntimeError(f"Compilation error: output file {file_target} not found")


def find_metal_path():
    """Utility function to find Metal path
    
    Returns
    -------
    path : str
        Path to Metal tools.
    """
    # Try to find Metal tools using xcrun
    try:
        metal_path = subprocess.check_output(["xcrun", "-sdk", "macosx", "-find", "metal"],
                                           stderr=subprocess.STDOUT).strip()
        return py_str(metal_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If xcrun fails, try common paths
        common_paths = [
            "/usr/bin/metal",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal"
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        raise RuntimeError(
            "Failed to automatically detect Metal installation. Please ensure Xcode command line tools are installed."
        )


def get_metal_version(metal_path=None):
    """Utility function to get Metal version
    
    Parameters
    ----------
    metal_path : Optional[str]
        Path to Metal compiler. If None is passed, will use `find_metal_path()` as default.
        
    Returns
    -------
    version : tuple
        The Metal version as a tuple of integers.
    """
    # Check if Metal is available
    if not is_metal_available():
        # If Metal is not available, return a default version
        return (2, 0)
    
    if metal_path is None:
        try:
            metal_path = find_metal_path()
        except RuntimeError:
            # If we can't find Metal, return a default version
            return (2, 0)
    
    # Try to get version information from the compiler
    try:
        cmd = [metal_path, "--version"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
        out = py_str(out)
        if proc.returncode == 0:
            # Parse version from output
            lines = out.split("\n")
            for line in lines:
                if "Metal" in line:
                    # Extract version numbers
                    import re
                    version_match = re.search(r"(\d+\.\d+)", line)
                    if version_match:
                        version_str = version_match.group(1)
                        return tuple(int(field) for field in version_str.split("."))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # If we can't determine the version, return a default
    return (2, 0)


@tvm.ffi.register_func("tilelang_callback_metal_compile", override=True)
def tilelang_callback_metal_compile(code, target):  # pylint: disable=unused-argument
    """use metal to generate metallib code for better optimization"""
    # Check if Metal is available before attempting compilation
    if not is_metal_available():
        warnings.warn("Metal tools are not available. Compilation will be skipped.", stacklevel=2)
        return bytearray()
    
    metallib = compile_metal(code, target_format="metallib")
    return metallib


def find_metal_device_path():
    """Utility function to find Metal device libraries
    
    Returns
    -------
    path : str
        Path to Metal device libraries.
    """
    # Metal doesn't have device libraries in the same way CUDA does
    # Return an empty string or a default path
    return ""


def callback_metal_compile(code, target):
    """TVM callback for Metal compilation"""
    try:
        return tilelang_callback_metal_compile(code, target)
    except RuntimeError as e:
        warnings.warn(f"Metal compilation failed: {str(e)}", stacklevel=2)
        return bytearray()


def get_metal_compute_version(target=None):
    """Utility function to get compute capability of Metal target.
    
    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target
        
    Returns
    -------
    compute_version : str
        Compute capability of Metal device (e.g. "2.0" or "2.4")
    """
    # For Metal, we'll return a default version or try to detect from the target
    if target and hasattr(target, 'attrs') and 'arch' in target.attrs:
        # Try to extract version from target
        arch = target.attrs["arch"]
        if "metal" in arch:
            # Extract version from metal arch string
            # This is a simplified approach - in practice, you might need more sophisticated parsing
            return "2.0"
    
    # Return default Metal version
    return "2.0"