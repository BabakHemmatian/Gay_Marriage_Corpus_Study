from subprocess import check_output
import sys

def parse_colon_divided_text(txt):
    return dict(
        (s.strip() for s in items)
        for items in (li.split(':') for li in txt.split('\n'))
        if len(items) == 2)

### function for the number of physical CPUs (for parallel processing of LDA)
# NOTE: Based on code from https://gist.github.com/luc-j-bourhis
def CpuInfo():
    physical_cpu_count = None

    if sys.platform.startswith('linux'):
        info = parse_colon_divided_text(check_output(('lscpu')))
        sockets = int(info['Socket(s)'])
        cores_per_socket = int(info['Core(s) per socket'])
        physical_cpu_count = sockets*cores_per_socket

    elif sys.platform == 'win32':
        from win32com.client import GetObject
        root = GetObject("winmgmts:root\cimv2")
        cpus = root.ExecQuery("Select * from Win32_Processor")
        physical_cpu_count = sum(
            cpu.NumberOfCores for cpu in cpus)

    elif sys.platform == 'darwin':
        info = parse_colon_divided_text(check_output(
            ('sysctl', 'hw.physicalcpu', 'hw.logicalcpu')))
        physical_cpu_count = int(info['hw.physicalcpu'])

    return physical_cpu_count - 1
