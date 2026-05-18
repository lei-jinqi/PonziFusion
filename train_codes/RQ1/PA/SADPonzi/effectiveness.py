#!/usr/bin/python3 -u
# cmd: ./effectiveness.py /data/SADPonzi_experiments/tse_data/ $PWD/eval_results/
import os
from random import sample
import subprocess
import sys

def process(cmd, shell=False):
    if isinstance(cmd, str):
        cmd = cmd.split(' ')
    return subprocess.run(cmd, shell=shell)

if len(sys.argv) != 3:
    print('Usage: %s [flags] <path_to_dataset> <path_to_results>' % sys.argv[0], file=sys.stderr)
    exit(0)


reports_path_base = sys.argv[2] #'./eval_results/effectiveness/'

for file in os.listdir(sys.argv[1]):
    sample_path = os.path.join(sys.argv[1], file)
    cmd = 'python3 sadponzi.py -i {} -o {}'.format(sample_path, reports_path_base+'/'+file)
    process(cmd)

# collect results:
ponzi_cnt = 0
profiles = [0,0,0]

for report_name in os.listdir(reports_path_base):
    with open(reports_path_base + '/' + report_name, 'r') as f:
        lineOne = f.readline()
        if lineOne.startswith("Found:"):
            ponzi_cnt += 1
            scheme = lineOne.split(" ", 2)[1]
            if scheme == "handover":
                profiles[0] += 1
            elif scheme == "chain":
                profiles[1] += 1
            elif scheme == "tree":
                profiles[2] += 1
            else:
                print("[-] Error-> \"{}\"".format(lineOne))
        else:
            print(report_name)

print('{} results {}'.format('-'*20, '-'*20))
print("#Handover schemes {}\n#Chain schemes {}\nTree schemes {}\n".format(*profiles))
print("-> Total = {}".format(ponzi_cnt))