import sys
import os
import numpy as np
import pandas as pd
from sliding_window_processor import collect_event_ids, FeatureExtractor

if len(sys.argv) != 3:
    print("Usage: python project_processor.py <log_structured_path> <anomaly_path>")
    sys.exit(1)

log_structured_path = sys.argv[1]
anomaly_path = sys.argv[2]

data_version = "_v5"
data_version = "_tf-idf{}".format(data_version)
save_location = os.path.dirname(log_structured_path)
print("save location: {}".format(save_location))

print("loading structured log file")
x_data = pd.read_csv(log_structured_path)

print("loading anomaly label file")
y_data = pd.read_csv(anomaly_path)

re_pat = r"(blk_-?\d+)"
col_names = ["BlockId", "EventSequence"]

print("collecting events for data")
events_data = collect_event_ids(x_data, re_pat, col_names)

print("merging block frames with labels")
events_data = events_data.merge(y_data, on="BlockId")

events_data_values = events_data["EventSequence"].values

# fit transform & transform
fe = FeatureExtractor()

print("fit_transform data")
subblocks_data = fe.fit_transform(
    events_data_values,
    term_weighting="tf-idf",
    length_percentile=95,
    window_size=16,
)

print("collecting y data")
y_data = events_data[["BlockId", "Label"]]

# saving files
print("writing y to csv")
y_data.to_csv("{}/y_test{}.csv".format(save_location, data_version))

print("saving x to numpy object")
np.save("{}/x_test{}.npy".format(save_location, data_version), subblocks_data)
