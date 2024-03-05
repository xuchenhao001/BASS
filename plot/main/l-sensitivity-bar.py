import sys

from plot.utils.time_acc_base import plot_sensitivity_bar

lr_0_01 = [48.3, 88.14, 79.22]
lr_0_1 = [31.9, 82.03, 61.8]
lr_0_001 = [47.8, 60.9, 71.9]

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_sensitivity_bar("", lr_0_01, lr_0_1, lr_0_001, save_path, plot_size="2")
