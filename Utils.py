from __future__ import print_function
from subprocess import check_output
import sys
import time
from config import *

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


### Function for writing parameters and model performance to file
## TODO: Write a separate set of variables to file for NN
def Write_Performance(output_path=output_path, NN=NN):
    with open(output_path+"/Performance",'a+') as perf:
        if not NN:
            print("***",file=perf)
            print("Time: "+time.strftime("%Y-%m-%d %H:%M:%S"),file=perf)
            print("*** Hyperparameters ***", file=perf)
            print("Training fraction = " + str(training_fraction),file=perf)
            print("Maximum vocabulary size = " + str(MaxVocab),file=perf)
            print("Minimum number of documents a token can appear in and be included = " + str(no_below),file=perf)
            print("Fraction of documents, tokens appearing more often than which will be filtered out = " + str(no_above),file=perf)
            print("Number of topics = " + str(num_topics),file=perf)
            print("Fraction of topics sampled = " + str(sample_topics),file=perf)
            print("Number of top words recorded for each topic = " + str(topn),file=perf)
            print("Number of comments sampled from each top topic = " + str(sample_comments),file=perf)
            print("Minimum comment length for sampled comments = " + str(min_comm_length),file=perf)
            print("Alpha (LDA) = " + str(alpha),file=perf)
            print("Eta (LDA) = " + str(eta),file=perf)
            print("Minimum topic probability = " + str(minimum_probability),file=perf)
            print("Minimum term probability = " + str(minimum_phi_value),file=perf)
