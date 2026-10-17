"""작업 프로세스별 운영체제 최대 메모리 기록."""
import sys


def peak_memory_bytes():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(value.PeakWorkingSetSize)
    import resource
    amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(amount * (1 if sys.platform == 'darwin' else 1024))
