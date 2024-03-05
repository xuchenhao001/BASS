import sys

from plot.utils.time_acc_base import plot_time_cost_bar

bass = [3.00, 0.03, 0.30, 1.29, 4.62]
sign_sgd = [2.99, 0.03, 0.01, 3.19, 6.22]
dtwn = [3.12, 0.03, 0.57, 4.01, 7.73]
fed_avg = [3.00, 0.03, 0.38, 3.48, 6.89]
local = [3.02, 0.03, 0.00, 0.00, 3.05]

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_time_cost_bar("", bass, sign_sgd, dtwn, fed_avg, local, save_path, plot_size="2")
