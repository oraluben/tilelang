import tvm
from tvm.target import Target
from .arch_base import TileDevice
from typing import List, Union


def is_metal_arch(arch: TileDevice) -> bool:
    return isinstance(arch, METAL)


class METAL(TileDevice):

    def __init__(self, target: Union[Target, str]):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        self.target = target
        
        # Try to get Metal device, fallback to default if not available
        try:
            device = tvm.runtime.metal(0)
            if not device.exist:
                raise RuntimeError("Cannot find Metal device 0.")
            self.device: tvm.runtime.Device = device
            # Get Metal-specific properties
            self.smem_cap = device.max_shared_memory_per_block
            self.compute_max_core = device.multi_processor_count
            self.warp_size = device.warp_size
            self.compute_capability = device.compute_version.replace(".", "")
        except Exception:
            # Fallback values for Metal devices
            self.device = None
            self.smem_cap = 32768  # 32KB shared memory per block
            self.compute_max_core = 24  # Typical for modern Apple GPUs
            self.warp_size = 32  # Metal thread execution width
            self.compute_capability = "20"  # Default version
            
        self.platform: str = "METAL"
        self.reg_cap: int = 32768  # 32K registers
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4  # Similar to other architectures
        self.l2_cache_size_bytes: int = getattr(target, 'l2_cache_size_bytes', 2 * 1024 * 1024)  # 2MB default
        # the number of transaction size in bytes
        self.transaction_size: List[int] = [32, 128]  # in bytes
        # bandwidth in MB/s, will be used for recommend basic tile size
        self.bandwidth: List[int] = [700, 8000]  # Approximate values for Apple GPUs

    def get_avaliable_tensorintrin_shapes(self):
        # Metal-specific tensor instruction shapes
        # Return common matrix multiplication shapes supported by Metal
        return [[8, 8], [16, 16], [32, 32]]

    def __repr__(self):
        return f"METAL({self.target})"