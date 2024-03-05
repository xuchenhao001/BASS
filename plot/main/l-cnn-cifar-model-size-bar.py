import sys

from plot.utils.time_acc_base import plot_model_size_bar

sgd = [2.061404228, 48.91030598, 1.66344738]
sign_sgd = [0.156475067, 3.301578522, 0.100452423]

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_model_size_bar("", sgd, sign_sgd, save_path, plot_size="2")
