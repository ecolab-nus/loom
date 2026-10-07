from glob import glob
import os

def extract_tflops(text):
    return [float(x.split("TFLOPS:")[1].strip()) for x in text if 'TFLOPS' in x][0]


def collect_tflops(folder):
    perf = {}
    l = glob(os.path.join(folder, '*.log'))

    for log in l:
        if 'compile' in log:
            continue
        perf[log.split('/')[-1].split('.')[0]] = extract_tflops(open(log).readlines())
    
    return perf

print("TileLoom:", collect_tflops('/workspace/loom/tmp_logs/tileloom_chunk_scan_bh'))
print("Loom:", collect_tflops('/workspace/loom/tmp_logs/loom_chunk_scan_bh'))
    