import sys
import os
from logparser import Drain

if len(sys.argv) != 2:
    print("Usage: python test_parser.py <log_file_path>")
    sys.exit(1)

log_file_path = sys.argv[1]
input_dir = os.path.dirname(log_file_path)  # The input directory of log file
output_dir = input_dir  # The output directory of parsing results
log_file_all = os.path.basename(log_file_path)  # The input log file name

log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"  # HDFS log format
# Regular expression list for optional preprocessing (default: [])
regex = [
    r"blk_(|-)[0-9]+",  # block id
    r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)",  # IP
    r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$",  # Numbers
]
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

# run on training dataset
parser = Drain.LogParser(
    log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex
)
parser.parse(log_file_all)
