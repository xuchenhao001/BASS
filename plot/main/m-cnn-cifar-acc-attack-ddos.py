import sys

from plot.utils.time_acc_base import plot_time_acc_attack

fed_sync_sgd = [17.9, 15.37, 32.07, 29.97, 38.6, 37.23, 41.97, 42.77, 44.73, 45.17, 46.97, 47.1, 47.9, 48.6, 48.7, 49.03, 49.5, 50.03, 50.87, 50.5, 50.33, 49.9, 50.1, 49.93, 49.87, 50.4, 50.2, 50.0, 49.8, 49.5, 49.77, 50.07, 50.37, 50.4, 50.23, 50.13, 50.23, 50.23, 50.53, 50.37, 50.3, 50.57, 50.33, 50.17, 50.13, 50.0, 50.23, 49.8, 49.93, 50.0, 49.83, 49.73, 49.73, 49.5, 49.23, 49.5, 49.37, 49.23, 49.23, 49.57, 49.6, 49.33, 49.3, 49.3, 49.27, 49.37, 49.6, 49.23, 49.03, 48.77, 48.97, 48.9, 49.03, 48.8, 48.5, 48.87, 48.87, 48.73, 48.13, 48.13, 48.53, 48.43, 48.4, 48.4, 48.2, 48.3, 48.73, 48.93, 48.8, 48.7, 48.57, 48.63, 48.43, 48.5, 48.4, 48.5, 48.37, 48.1, 48.13, 48.1, 48.17, 47.97, 48.0, 48.03, 48.13, 47.87, 47.8, 47.77, 47.77, 47.73, 47.73, 47.6, 47.6, 47.63, 47.77, 47.6, 47.6, 47.77, 47.8, 47.73, 47.6, 47.97, 48.07, 48.0, 48.03, 48.13, 48.1, 48.3, 48.3, 48.1, 48.07, 48.07, 48.2, 48.17, 48.33, 48.37, 48.4, 48.27, 48.0, 48.07, 48.03, 47.97, 47.9, 47.93, 47.9, 47.73, 47.9, 47.97, 48.07, 48.0, 48.0, 48.0, 48.0, 48.0, 48.03, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.03, 48.0, 48.0, 48.0, 48.07, 48.1, 48.03, 48.03, 48.03, 48.03, 48.07, 48.07, 48.03, 48.03, 48.0, 48.03, 48.03, 48.03, 48.07, 48.07, 48.07, 48.07, 48.03, 48.03, 48.03, 48.03, 48.0, 48.03, 48.03, 48.03, 48.03, 48.03, 48.07, 48.07, 48.07]
fed_avg = [26.87, 26.87, 34.23, 34.23, 34.23, 42.93, 42.93, 42.93, 42.93, 42.93, 42.93, 42.93, 42.93, 42.93, 42.93, 41.9, 46.8, 46.8, 46.8, 46.8, 46.8, 46.8, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.27, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 46.13, 44.23, 44.23, 44.23, 44.23, 44.23, 44.23, 44.23, 46.0, 46.0, 46.0, 46.0, 46.0, 46.0, 46.0, 47.37, 47.37, 47.37, 47.37, 47.37, 47.37, 47.37, 47.37, 46.83, 46.83, 46.83, 46.83, 46.83, 46.83, 46.83, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 48.07, 47.5, 47.5, 47.5, 46.6, 46.6, 46.6, 46.6, 46.6, 46.6, 46.6, 46.6, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.57, 48.03, 48.03, 48.03, 48.03, 48.17, 48.17, 48.17, 48.17, 48.17, 48.17, 48.17, 48.17, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03, 48.03]
fed_ecsign = [12.87, 24.0, 31.67, 34.17, 39.0, 39.97, 42.3, 47.53, 49.43, 49.83, 50.83, 50.1, 49.7, 50.0, 49.0, 49.67, 49.7, 49.27, 49.87, 49.37, 49.2, 49.17, 49.6, 49.3, 49.67, 49.53, 48.97, 48.73, 48.87, 49.27, 48.83, 49.0, 48.73, 48.33, 48.17, 48.13, 47.83, 47.93, 47.23, 47.03, 47.13, 47.1, 47.13, 46.57, 47.3, 46.73, 47.6, 47.2, 47.07, 46.67, 46.47, 47.27, 46.8, 46.33, 46.23, 46.4, 45.93, 45.97, 45.97, 46.57, 46.8, 46.67, 45.53, 46.13, 45.73, 46.13, 46.9, 46.37, 46.13, 45.97, 45.87, 45.23, 45.7, 45.37, 45.9, 46.37, 46.8, 45.33, 46.2, 46.13, 45.67, 45.67, 45.87, 45.47, 45.93, 46.33, 46.27, 46.0, 45.73, 45.43, 45.3, 44.8, 45.27, 45.6, 45.97, 45.6, 45.5, 46.07, 45.7, 45.8, 45.77, 45.3, 45.23, 45.33, 45.4, 45.43, 45.63, 45.6, 45.5, 45.3, 45.17, 45.3, 45.2, 45.27, 45.27, 45.5, 45.4, 45.53, 45.67, 45.67, 45.77, 45.73, 45.87, 45.9, 45.8, 46.03, 45.87, 46.13, 45.93, 46.13, 45.8, 46.37, 46.4, 46.03, 46.37, 46.33, 46.23, 46.3, 46.37, 46.03, 46.73, 46.37, 46.4, 46.73, 46.33, 46.2, 46.5, 46.2, 46.63, 46.53, 46.43, 46.53, 46.4, 46.5, 46.4, 46.47, 46.5, 46.5, 46.47, 46.43, 46.43, 46.4, 46.4, 46.37, 46.37, 46.37, 46.33, 46.33, 46.37, 46.37, 46.37, 46.37, 46.37, 46.3, 46.3, 46.3, 46.27, 46.3, 46.27, 46.3, 46.3, 46.3, 46.3, 46.3, 46.27, 46.3, 46.33, 46.33, 46.37, 46.33, 46.33, 46.33, 46.3, 46.23, 46.27, 46.3, 46.37, 46.37, 46.37, 46.37]
fed_efsign = [10.0, 12.9, 26.83, 30.4, 32.93, 36.07, 38.6, 40.53, 43.27, 44.97, 46.2, 46.57, 46.87, 48.13, 47.2, 47.43, 47.27, 47.9, 48.23, 48.53, 48.8, 47.97, 47.57, 47.23, 47.53, 47.17, 48.87, 46.33, 48.47, 46.9, 48.17, 46.97, 48.1, 47.23, 48.27, 48.4, 46.57, 45.97, 47.67, 47.23, 46.17, 46.53, 46.43, 45.37, 44.83, 45.93, 45.9, 45.07, 46.87, 46.77, 45.87, 46.1, 45.6, 45.63, 45.7, 45.27, 45.23, 45.0, 45.5, 45.13, 45.93, 44.2, 45.67, 45.17, 46.47, 44.17, 45.0, 45.87, 45.0, 45.57, 45.5, 45.0, 45.3, 45.23, 45.4, 44.13, 45.17, 45.17, 44.33, 44.67, 45.1, 45.23, 43.33, 44.83, 44.97, 45.0, 44.77, 43.7, 43.63, 43.37, 43.87, 43.8, 44.0, 43.5, 44.23, 44.37, 42.2, 44.27, 43.7, 42.77, 43.7, 42.8, 42.2, 44.5, 42.5, 42.27, 42.17, 41.37, 42.03, 42.7, 42.7, 43.07, 43.2, 42.1, 43.2, 43.2, 42.5, 41.67, 43.53, 42.03, 43.17, 41.57, 43.3, 41.9, 42.3, 41.87, 43.5, 41.9, 43.73, 40.1, 43.0, 41.23, 42.0, 39.9, 42.8, 41.03, 41.97, 40.4, 43.13, 43.5, 42.53, 41.23, 43.8, 41.03, 41.63, 40.43, 40.53, 41.1, 43.7, 42.77, 41.63, 41.67, 41.87, 41.33, 41.97, 42.13, 42.47, 41.63, 44.03, 42.0, 40.5, 41.6, 40.37, 40.73, 41.07, 40.77, 40.93, 41.3, 41.07, 40.8, 41.67, 40.67, 41.07, 39.43, 41.93, 40.73, 40.03, 42.2, 38.5, 42.6, 39.47, 39.67, 41.2, 41.37, 39.77, 40.67, 39.33, 40.33, 40.73, 41.3, 39.0, 42.83, 39.07, 41.97, 40.4, 40.6, 39.17, 40.27, 40.73, 41.73]
fed_mvsign = [14.3, 28.37, 34.73, 37.33, 39.47, 42.37, 43.4, 44.87, 46.97, 47.37, 47.33, 47.97, 48.33, 48.2, 49.83, 49.3, 48.53, 48.67, 49.37, 49.6, 49.8, 49.8, 49.1, 49.13, 49.27, 49.3, 49.27, 49.2, 48.27, 48.83, 48.27, 48.4, 48.8, 48.43, 47.63, 47.57, 47.77, 48.37, 48.77, 47.6, 46.83, 47.43, 47.27, 47.23, 47.27, 46.7, 47.13, 46.5, 45.27, 46.53, 46.37, 46.4, 46.8, 46.27, 45.47, 45.87, 45.93, 45.1, 45.9, 45.07, 45.4, 46.0, 45.8, 45.13, 45.2, 46.23, 45.93, 45.9, 45.63, 44.93, 45.13, 44.2, 44.27, 44.0, 44.17, 45.47, 44.37, 45.47, 45.37, 44.43, 44.03, 45.47, 45.6, 45.0, 44.93, 44.4, 43.93, 43.47, 43.27, 43.27, 43.3, 44.17, 44.2, 44.3, 44.1, 43.73, 45.53, 44.97, 44.37, 44.37, 44.53, 44.77, 44.5, 44.23, 44.27, 44.27, 44.27, 44.27, 44.67, 44.53, 44.43, 44.37, 44.43, 44.63, 45.03, 45.27, 45.1, 45.0, 45.17, 45.37, 45.47, 45.53, 45.57, 45.63, 45.63, 45.93, 45.93, 46.07, 46.27, 46.17, 45.9, 46.33, 46.07, 45.8, 46.6, 45.7, 46.5, 46.2, 45.87, 46.43, 46.03, 46.23, 46.07, 46.37, 46.43, 45.6, 46.2, 45.5, 46.0, 45.9, 45.87, 45.83, 45.8, 45.63, 45.7, 45.73, 45.9, 45.9, 45.93, 46.0, 46.03, 45.97, 45.9, 45.87, 45.93, 45.9, 45.87, 45.9, 45.9, 45.93, 45.93, 45.93, 45.93, 45.93, 45.97, 46.03, 46.0, 45.93, 45.97, 45.93, 45.93, 45.93, 45.93, 45.93, 45.93, 45.93, 45.93, 45.93, 45.93, 45.93, 45.9, 45.9, 45.97, 45.93, 45.97, 45.97, 46.0, 46.07, 46.03, 46.03]
fed_rlrsign = [20.03, 25.9, 29.5, 32.83, 35.0, 37.87, 40.53, 41.57, 42.87, 43.9, 43.9, 43.93, 43.8, 43.23, 43.43, 44.4, 44.4, 43.7, 43.13, 42.33, 41.63, 41.53, 41.0, 39.63, 40.53, 40.57, 40.6, 41.13, 40.77, 40.87, 40.47, 40.23, 40.57, 40.37, 40.87, 41.27, 41.23, 41.67, 41.83, 42.1, 41.0, 39.63, 39.57, 39.77, 39.7, 40.37, 40.27, 40.4, 41.27, 41.1, 42.03, 40.83, 39.97, 39.97, 40.17, 40.47, 41.03, 41.7, 41.6, 41.73, 40.4, 39.1, 39.6, 39.93, 39.87, 40.27, 38.97, 39.73, 40.67, 40.17, 40.4, 40.83, 41.13, 41.33, 41.03, 40.53, 39.67, 39.1, 40.0, 39.93, 40.87, 40.8, 41.43, 41.53, 40.77, 40.2, 40.57, 41.63, 41.5, 42.03, 40.83, 41.07, 41.73, 40.97, 41.3, 42.07, 41.37, 41.93, 41.5, 41.67, 41.83, 41.73, 41.87, 41.93, 42.0, 41.6, 41.6, 41.67, 41.37, 41.33, 41.33, 41.4, 41.7, 41.67, 41.93, 41.87, 42.27, 42.2, 42.17, 42.37, 42.37, 42.57, 42.4, 42.6, 42.47, 42.43, 42.57, 42.33, 42.33, 42.3, 42.6, 42.93, 42.67, 42.6, 42.6, 42.6, 42.53, 42.67, 42.53, 42.67, 42.2, 42.4, 42.37, 42.73, 42.6, 42.7, 42.57, 42.33, 42.53, 42.57, 42.5, 42.47, 42.47, 42.33, 42.3, 42.3, 42.3, 42.33, 42.33, 42.23, 42.23, 42.2, 42.23, 42.2, 42.33, 42.37, 42.37, 42.33, 42.37, 42.4, 42.4, 42.37, 42.37, 42.33, 42.33, 42.43, 42.4, 42.4, 42.37, 42.37, 42.43, 42.43, 42.47, 42.37, 42.37, 42.43, 42.47, 42.53, 42.43, 42.43, 42.4, 42.37, 42.4, 42.3, 42.33, 42.37, 42.43, 42.47, 42.43, 42.43]
fed_err = [9.73, 9.73, 9.73, 9.73, 9.73, 9.73, 9.73, 9.73, 9.73, 9.73, 9.73, 9.73, 9.73, 23.73, 23.73, 23.73, 23.73, 23.73, 23.73, 23.73, 23.73, 23.73, 23.73, 23.73, 23.73, 32.53, 32.53, 32.53, 32.53, 32.53, 32.53, 32.53, 32.53, 32.53, 32.53, 32.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.53, 38.2, 38.2, 38.2, 38.2, 38.2, 38.2, 38.2, 42.37, 42.37, 42.37, 42.37, 42.37, 42.37, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 38.83, 42.6, 42.6, 42.6, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 41.3, 41.3, 41.3, 41.3, 41.3, 41.3, 41.3, 41.3, 41.3, 41.3, 41.3, 41.3, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 42.77, 42.77, 42.77, 47.13, 47.13, 47.13, 47.13, 47.13, 45.87, 45.87, 45.87, 45.87, 45.87, 45.87, 45.87, 45.87, 45.87, 46.0, 45.73, 45.73, 45.73, 45.73, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9, 46.9]
fed_lfr = [8.93, 8.93, 8.93, 24.8, 24.8, 24.8, 24.8, 24.8, 24.8, 24.8, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 32.93, 38.3, 38.3, 38.3, 38.3, 38.3, 38.3, 38.3, 38.3, 38.3, 38.3, 40.77, 40.77, 40.77, 41.03, 41.03, 41.03, 41.03, 41.03, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 43.7, 44.3, 43.43, 48.03, 46.8, 46.8, 46.8, 46.8, 46.8, 46.8, 46.8, 46.8, 44.63, 44.63, 44.63, 44.63, 44.63, 44.63, 44.63, 44.63, 44.63, 45.43, 45.43, 45.43, 45.43, 45.07, 45.07, 45.07, 45.07, 45.07, 45.07, 45.07, 44.2, 45.3, 45.3, 45.3, 45.3, 44.2, 44.5, 44.5, 44.5, 44.5, 44.5, 44.5, 44.5, 46.63, 46.63, 46.63, 46.63, 46.63, 46.63, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 46.73, 45.9, 45.9, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 47.8, 43.97, 47.93, 45.43, 45.43, 45.43, 45.43, 45.43, 45.43, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83, 47.83]
fed_fleam = [9.53, 9.53, 9.53, 9.53, 9.53, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 34.0, 34.0, 34.0, 34.0, 34.0, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 39.3, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.63, 43.63, 44.97, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 48.37, 48.37, 48.37, 48.37, 48.37, 48.37, 48.37, 48.37, 48.37, 48.37, 48.37, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 47.47, 45.1, 45.1, 45.1, 46.93, 46.93, 46.93, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 49.63, 48.4, 48.4, 48.4, 48.4, 48.4, 48.03, 48.03, 48.03, 48.03, 48.03, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 48.7, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.53, 46.0, 46.0, 46.0, 46.0, 46.0, 49.43, 49.43, 49.43, 49.43, 49.43, 47.83, 49.87, 49.87, 49.87, 49.87, 49.87, 49.87, 49.87, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37, 49.37]
fed_trust = [10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 10.07, 29.2, 29.2, 29.2, 29.2, 29.2, 29.2, 29.2, 29.2, 29.2, 29.2, 29.2, 36.37, 36.37, 36.37, 36.37, 36.37, 37.83, 44.6, 44.6, 44.6, 44.6, 44.6, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.03, 42.43, 42.43, 42.43, 42.43, 42.43, 42.43, 42.43, 42.43, 42.43, 42.43, 46.93, 46.93, 46.93, 46.93, 41.9, 41.9, 41.9, 41.9, 41.9, 41.9, 41.9, 41.9, 41.9, 41.9, 41.9, 41.9, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.7, 45.17, 45.17, 45.17, 45.17, 45.17, 45.17, 45.17, 45.17, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 44.07, 48.2, 48.2, 48.2, 48.2, 48.2, 48.2, 48.2, 48.2, 48.2, 48.2, 48.2, 48.2, 48.2, 46.3, 46.33, 46.33, 46.33, 46.33, 46.33, 46.33, 46.33, 46.33, 46.33, 46.33, 47.13]

data = {
    "fed_sync_sgd": fed_sync_sgd,
    "fed_avg": fed_avg,
    "fed_ecsign": fed_ecsign,
    "fed_efsign": fed_efsign,
    "fed_mvsign": fed_mvsign,
    "fed_rlrsign": fed_rlrsign,
    "fed_err": fed_err,
    "fed_lfr": fed_lfr,
    "fed_trust": fed_trust,
    "fed_fleam": fed_fleam,
}

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_time_acc_attack("", data, "out", save_path, plot_size="3")