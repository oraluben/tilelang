import ctypes
import sys
from typing import Optional

try:
    # Try to import Metal framework
    import Metal
    import MetalKit
    import objc
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


class MetalDeviceProp:
    """
    Metal device properties structure, similar to cudaDeviceProp but with Metal-specific properties.
    """
    def __init__(self):
        self.name = ""
        self.sharedMemPerThreadgroup = 0  # Similar to sharedMemPerBlock in CUDA
        self.threadExecutionWidth = 0     # Similar to warpSize in CUDA
        self.maxThreadsPerThreadgroup = 0
        self.maxThreadsPerMultiprocessor = 0
        self.multiProcessorCount = 0      # Similar to multiProcessorCount in CUDA
        self.memoryClockRate = 0          # Memory clock rate in kHz
        self.memoryBusWidth = 0           # Memory bus width in bits
        self.l2CacheSize = 0              # L2 cache size in bytes
        self.computeCapability = ""       # Metal version as string (e.g., "2.4")
        self.totalGlobalMem = 0           # Total global memory in bytes
        self.maxTransferRate = 0          # Maximum memory transfer rate in MB/s


def get_metal_device_properties(device_id: int = 0) -> Optional[MetalDeviceProp]:
    """
    Get Metal device properties for the specified device ID.
    
    Args:
        device_id (int): The Metal device ID (default: 0 for the default device)
        
    Returns:
        Optional[MetalDeviceProp]: Device properties or None if Metal is not available
        
    Raises:
        RuntimeError: If device detection fails
    """
    if not METAL_AVAILABLE:
        return None
        
    try:
        # Get the default Metal device
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("Failed to create Metal device")
            
        # Create properties object
        prop = MetalDeviceProp()
        
        # Get device name
        prop.name = str(device.name())
        
        # Get Metal-specific properties
        prop.sharedMemPerThreadgroup = device.maxThreadgroupMemoryLength()
        prop.threadExecutionWidth = device.threadExecutionWidth()
        prop.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup().width * \
                                       device.maxThreadsPerThreadgroup().height * \
                                       device.maxThreadsPerThreadgroup().depth
        prop.multiProcessorCount = device.processorCount()
        
        # Approximate values for Metal devices (as these aren't directly available)
        # These would need to be looked up in a database or estimated based on device name
        prop.memoryClockRate = 0  # Not directly available in Metal
        prop.memoryBusWidth = 0   # Not directly available in Metal
        prop.l2CacheSize = 0      # Not directly available in Metal
        prop.maxTransferRate = 0  # Not directly available in Metal
        
        # Estimate compute capability from Metal feature set
        if hasattr(device, 'supportsFamily'):
            # Modern approach for getting Metal version
            if device.supportsFamily_(Metal.MTLGPUFamilyApple7):
                prop.computeCapability = "2.4"
            elif device.supportsFamily_(Metal.MTLGPUFamilyApple6):
                prop.computeCapability = "2.3"
            elif device.supportsFamily_(Metal.MTLGPUFamilyApple5):
                prop.computeCapability = "2.2"
            elif device.supportsFamily_(Metal.MTLGPUFamilyApple4):
                prop.computeCapability = "2.1"
            elif device.supportsFamily_(Metal.MTLGPUFamilyApple3):
                prop.computeCapability = "2.0"
            elif device.supportsFamily_(Metal.MTLGPUFamilyApple2):
                prop.computeCapability = "1.2"
            elif device.supportsFamily_(Metal.MTLGPUFamilyApple1):
                prop.computeCapability = "1.1"
            else:
                prop.computeCapability = "1.0"
        else:
            # Fallback for older systems
            prop.computeCapability = "1.0"
            
        # Get memory information if possible
        try:
            # This is a simplified approach; actual implementation might need to use
            # other APIs or heuristics to determine total memory
            prop.totalGlobalMem = 0  # Would need to be implemented based on system info
        except:
            prop.totalGlobalMem = 0
            
        return prop
    except Exception as e:
        print(f"Error getting Metal device properties: {str(e)}")
        return None


def get_metal_device_name(device_id: int = 0) -> Optional[str]:
    """
    Get the name of the Metal device.
    
    Args:
        device_id (int): The Metal device ID (default: 0 for the default device)
        
    Returns:
        Optional[str]: Device name or None if Metal is not available
        
    Raises:
        RuntimeError: If device detection fails
    """
    if not METAL_AVAILABLE:
        return None
        
    prop = get_metal_device_properties(device_id)
    if prop:
        return prop.name
    else:
        raise RuntimeError("Failed to get Metal device properties.")


def get_shared_memory_per_threadgroup(device_id: int = 0, format: str = "bytes") -> Optional[int]:
    """
    Get the shared memory per threadgroup (similar to sharedMemPerBlock in CUDA).
    
    Args:
        device_id (int): The Metal device ID (default: 0 for the default device)
        format (str): The format to return the size in ("bytes", "kb", "mb")
        
    Returns:
        Optional[int]: Shared memory size or None if Metal is not available
        
    Raises:
        RuntimeError: If device detection fails
    """
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    
    if not METAL_AVAILABLE:
        # Return fallback values if Metal is not available
        if format == "bytes":
            return 32768  # 32KB
        elif format == "kb":
            return 32
        elif format == "mb":
            return 0
        return None
        
    prop = get_metal_device_properties(device_id)
    if prop:
        shared_mem = prop.sharedMemPerThreadgroup
        if format == "bytes":
            return shared_mem
        elif format == "kb":
            return shared_mem // 1024
        elif format == "mb":
            return shared_mem // (1024 * 1024)
        else:
            raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")
    else:
        raise RuntimeError("Failed to get Metal device properties.")


def get_thread_execution_width(device_id: int = 0) -> Optional[int]:
    """
    Get the thread execution width (similar to warpSize in CUDA).
    
    Args:
        device_id (int): The Metal device ID (default: 0 for the default device)
        
    Returns:
        Optional[int]: Thread execution width or None if Metal is not available
        
    Raises:
        RuntimeError: If device detection fails
    """
    if not METAL_AVAILABLE:
        # Return fallback value if Metal is not available
        return 32  # Typical for Metal devices
        
    prop = get_metal_device_properties(device_id)
    if prop:
        return prop.threadExecutionWidth
    else:
        raise RuntimeError("Failed to get Metal device properties.")


def get_max_threads_per_threadgroup(device_id: int = 0) -> Optional[int]:
    """
    Get the maximum threads per threadgroup.
    
    Args:
        device_id (int): The Metal device ID (default: 0 for the default device)
        
    Returns:
        Optional[int]: Maximum threads per threadgroup or None if Metal is not available
        
    Raises:
        RuntimeError: If device detection fails
    """
    if not METAL_AVAILABLE:
        # Return fallback value if Metal is not available
        return 1024  # Common default for Metal devices
        
    prop = get_metal_device_properties(device_id)
    if prop:
        return prop.maxThreadsPerThreadgroup
    else:
        raise RuntimeError("Failed to get Metal device properties.")


def get_compute_capability(device_id: int = 0) -> Optional[str]:
    """
    Get the compute capability equivalent for Metal.
    
    Args:
        device_id (int): The Metal device ID (default: 0 for the default device)
        
    Returns:
        Optional[str]: Compute capability as string or None if Metal is not available
        
    Raises:
        RuntimeError: If device detection fails
    """
    if not METAL_AVAILABLE:
        # Return fallback value if Metal is not available
        return "2.0"  # Default version
        
    prop = get_metal_device_properties(device_id)
    if prop:
        return prop.computeCapability
    else:
        raise RuntimeError("Failed to get Metal device properties.")


def get_num_compute_units(device_id: int = 0) -> Optional[int]:
    """
    Get the number of compute units (similar to multiProcessorCount in CUDA).
    
    Args:
        device_id (int): The Metal device ID (default: 0 for the default device)
        
    Returns:
        Optional[int]: Number of compute units or None if Metal is not available
        
    Raises:
        RuntimeError: If device detection fails
    """
    if not METAL_AVAILABLE:
        # Return fallback value if Metal is not available
        return 24  # Typical for modern Apple GPUs
        
    prop = get_metal_device_properties(device_id)
    if prop:
        return prop.multiProcessorCount
    else:
        raise RuntimeError("Failed to get Metal device properties.")


def get_memory_bandwidth(device_id: int = 0) -> Optional[int]:
    """
    Get memory bandwidth information (approximate).
    
    Args:
        device_id (int): The Metal device ID (default: 0 for the default device)
        
    Returns:
        Optional[int]: Memory bandwidth in MB/s or None if Metal is not available
        
    Raises:
        RuntimeError: If device detection fails
    """
    if not METAL_AVAILABLE:
        # Return fallback value if Metal is not available
        return 8000  # Approximate value for Apple GPUs
        
    # Note: Metal doesn't directly expose memory bandwidth
    # This would need to be looked up in a database or estimated
    # For now, we'll return a placeholder value
    return 8000  # Approximate value for Apple GPUs


def is_metal_available() -> bool:
    """
    Check if Metal framework is available on the system.
    
    Returns:
        bool: True if Metal is available, False otherwise
    """
    return METAL_AVAILABLE


def get_metal_device_count() -> int:
    """
    Get the number of available Metal devices.
    
    Returns:
        int: Number of available Metal devices (0 if Metal is not available)
    """
    if not METAL_AVAILABLE:
        return 0
        
    # In Metal, there's typically one default device
    # More complex implementations might enumerate all devices
    device = Metal.MTLCreateSystemDefaultDevice()
    return 1 if device is not None else 0