import sys

from plot.utils.time_acc_base import plot_cache_level

acc = [88.14, 88.61, 88.97, 89.11]
time_cost = [27.87, 29.78, 32.25, 35.86]

data = {
    "ACC": acc,
    "TIME": time_cost,
}

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_cache_level("", data, save_path, plot_size="2")
