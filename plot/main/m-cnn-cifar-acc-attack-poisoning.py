import sys

from plot.utils.time_acc_base import plot_time_acc_attack

fed_sync_sgd = [28.13, 29.1, 35.8, 37.37, 37.6, 38.67, 41.03, 43.0, 44.23, 46.53, 47.17, 48.57, 49.1, 49.93, 50.23, 50.93, 50.9, 51.97, 51.8, 52.13, 52.7, 53.53, 53.33, 53.53, 53.13, 53.97, 53.57, 54.03, 53.4, 53.5, 52.93, 53.47, 53.3, 53.9, 53.83, 54.13, 54.0, 54.17, 53.67, 54.17, 53.37, 53.73, 54.0, 54.43, 54.17, 54.17, 53.3, 53.47, 53.5, 54.03, 54.0, 54.03, 54.27, 54.17, 53.9, 53.8, 53.6, 54.3, 53.93, 54.47, 54.9, 54.3, 54.27, 53.4, 52.9, 53.03, 52.87, 53.2, 53.17, 53.23, 53.33, 53.47, 53.6, 54.07, 53.6, 53.17, 53.1, 53.4, 52.9, 53.6, 53.37, 53.87, 53.7, 53.5, 53.53, 54.1, 53.3, 53.4, 52.9, 53.47, 53.9, 53.93, 53.47, 53.5, 53.13, 52.5, 52.27, 52.6, 52.33, 52.33, 52.23, 52.27, 52.23, 52.33, 52.23, 52.27, 52.13, 52.27, 52.27, 52.4, 52.57, 52.43, 52.67, 52.67, 52.7, 52.73, 52.6, 52.67, 52.63, 52.8, 52.87, 52.87, 52.97, 52.93, 52.93, 53.03, 53.13, 53.17, 53.13, 53.1, 53.13, 53.23, 53.13, 53.1, 52.97, 53.0, 53.1, 53.07, 53.03, 52.83, 52.8, 52.77, 52.7, 52.67, 52.6, 52.73, 52.73, 52.97, 52.73, 52.73, 52.77, 52.77, 52.77, 52.73, 52.7, 52.73, 52.73, 52.73, 52.8, 52.8, 52.8, 52.8, 52.8, 52.8, 52.9, 52.87, 52.87, 52.83, 52.9, 52.93, 52.93, 52.93, 52.87, 52.87, 52.87, 52.87, 52.83, 52.8, 52.77, 52.7, 52.7, 52.73, 52.77, 52.77, 52.8, 52.77, 52.8, 52.8, 52.8, 52.8, 52.8, 52.8, 52.77, 52.77, 52.73, 52.77, 52.8, 52.77, 52.73, 52.73]
fed_avg = [18.57, 20.4, 27.27, 32.87, 31.43, 31.57, 25.53, 39.2, 42.87, 35.37, 44.37, 38.2, 43.97, 46.47, 36.6, 30.7, 43.0, 40.4, 47.5, 43.93, 47.97, 43.87, 46.3, 49.97, 43.17, 47.93, 31.67, 41.33, 40.77, 46.8, 42.47, 50.2, 49.93, 39.0, 46.7, 45.53, 50.03, 47.77, 46.13, 48.2, 49.67, 50.77, 50.5, 51.1, 47.13, 50.07, 47.4, 47.0, 51.07, 51.93, 50.43, 46.03, 51.3, 50.6, 44.53, 50.47, 44.03, 44.67, 49.6, 46.23, 44.77, 44.53, 43.83, 51.4, 44.67, 49.87, 52.37, 51.9, 48.93, 45.37, 45.97, 40.7, 40.5, 46.83, 39.4, 48.57, 48.7, 50.13, 41.93, 49.03, 41.77, 44.27, 46.9, 53.03, 51.4, 51.2, 50.13, 43.53, 51.6, 47.47, 51.63, 49.9, 50.2, 37.7, 50.7, 46.67, 50.87, 50.4, 51.93, 49.5, 47.57, 47.73, 48.87, 52.2, 51.0, 52.23, 52.7, 52.2, 53.43, 52.6, 49.27, 52.0, 51.17, 53.0, 48.93, 52.63, 52.4, 49.53, 50.93, 50.4, 50.93, 50.17, 52.5, 52.1, 50.8, 51.77, 48.17, 49.87, 52.7, 40.07, 51.0, 54.5, 50.67, 50.87, 52.43, 52.47, 50.6, 46.97, 49.43, 52.2, 49.53, 49.8, 50.13, 50.3, 53.53, 34.2, 51.47, 51.07, 51.73, 53.2, 52.1, 52.83, 50.87, 50.83, 49.8, 49.47, 51.6, 51.13, 53.03, 52.67, 50.8, 46.73, 52.73, 52.7, 52.43, 53.37, 53.83, 52.23, 50.73, 54.63, 50.47, 50.1, 53.4, 52.8, 53.23, 46.6, 52.03, 52.37, 52.3, 39.4, 46.5, 52.83, 47.83, 53.43, 51.2, 52.67, 51.33, 52.4, 50.53, 52.3, 54.17, 53.7, 55.13, 53.33, 50.07, 51.93, 53.8, 52.87, 51.87, 51.33]
fed_ecsign = [18.47, 30.4, 32.43, 35.17, 38.93, 42.77, 44.37, 45.7, 48.13, 48.43, 48.3, 49.53, 50.27, 50.53, 50.13, 49.03, 49.77, 51.27, 51.07, 50.63, 50.23, 50.43, 50.9, 50.37, 50.13, 50.03, 49.57, 49.87, 49.93, 50.17, 50.27, 50.33, 50.33, 49.0, 49.53, 49.97, 49.53, 48.97, 48.77, 48.43, 48.67, 49.03, 49.73, 49.6, 48.7, 47.87, 47.97, 49.33, 49.83, 49.3, 49.73, 49.23, 48.5, 48.5, 49.47, 49.37, 50.2, 50.23, 50.73, 50.03, 50.5, 50.73, 50.37, 51.63, 51.17, 51.13, 49.97, 51.03, 51.27, 50.17, 50.17, 51.0, 50.63, 50.2, 50.37, 50.5, 50.7, 50.73, 50.93, 51.7, 51.13, 50.5, 50.77, 49.9, 50.83, 50.27, 50.73, 49.47, 50.4, 51.23, 52.07, 51.83, 50.73, 51.37, 50.67, 50.43, 50.87, 50.27, 51.37, 51.53, 51.77, 51.97, 52.23, 52.07, 52.17, 52.27, 52.33, 52.37, 52.63, 52.5, 52.77, 52.67, 52.47, 52.37, 52.37, 52.2, 52.2, 52.27, 52.07, 52.27, 52.27, 52.5, 52.47, 52.53, 52.9, 52.87, 52.83, 52.87, 52.8, 52.97, 52.73, 52.67, 52.77, 52.7, 52.87, 53.0, 52.47, 52.13, 51.97, 52.03, 51.97, 52.1, 52.07, 51.7, 52.07, 52.1, 52.1, 52.2, 52.37, 52.37, 52.37, 52.4, 52.33, 52.4, 52.43, 52.47, 52.43, 52.43, 52.5, 52.43, 52.5, 52.57, 52.5, 52.43, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.47, 52.4, 52.4, 52.43, 52.27, 52.37, 52.47, 52.43, 52.5, 52.37, 52.33, 52.3, 52.37, 52.33, 52.23, 52.4, 52.4, 52.33, 52.3, 52.37, 52.33, 52.33, 52.2, 52.23, 52.23, 52.27, 52.2, 52.2, 52.3, 52.33]
fed_efsign = [10.0, 16.43, 29.97, 34.53, 38.23, 40.77, 42.7, 44.13, 44.5, 45.83, 46.6, 47.1, 48.1, 48.23, 48.5, 48.47, 48.03, 46.27, 46.1, 47.07, 45.97, 45.4, 45.33, 46.07, 45.17, 45.73, 45.2, 45.47, 46.53, 45.43, 46.1, 44.3, 45.17, 46.57, 45.37, 46.0, 44.13, 44.73, 45.07, 45.3, 44.7, 46.0, 45.3, 46.0, 44.87, 43.63, 45.37, 42.4, 45.57, 44.9, 45.83, 43.03, 45.17, 44.0, 43.03, 44.47, 44.6, 44.63, 43.63, 44.23, 42.67, 46.33, 43.57, 42.6, 43.63, 43.6, 44.23, 43.0, 44.97, 44.73, 44.8, 43.83, 44.73, 44.0, 41.73, 42.27, 41.57, 42.2, 39.5, 41.53, 41.73, 41.23, 43.2, 42.4, 41.67, 42.37, 40.1, 42.23, 40.6, 42.67, 42.6, 40.27, 42.7, 40.8, 43.77, 42.8, 42.87, 41.9, 43.9, 42.2, 40.6, 43.73, 41.8, 43.0, 39.37, 39.5, 40.17, 42.43, 41.7, 41.77, 42.33, 42.7, 41.63, 43.7, 41.6, 42.27, 40.1, 43.6, 43.3, 42.1, 42.57, 41.6, 43.27, 38.93, 41.37, 39.57, 41.57, 40.93, 41.1, 41.47, 39.4, 39.9, 40.63, 42.57, 41.87, 40.27, 41.43, 39.4, 43.47, 40.8, 41.77, 39.07, 40.63, 38.67, 38.53, 38.73, 38.57, 40.77, 39.93, 41.2, 40.53, 42.3, 39.03, 41.77, 40.87, 41.77, 41.6, 41.47, 41.57, 41.17, 38.77, 39.73, 43.13, 41.23, 41.03, 39.97, 41.9, 41.07, 41.13, 41.77, 41.1, 38.83, 40.43, 41.0, 39.17, 36.23, 38.43, 40.43, 39.47, 39.53, 41.77, 39.37, 41.17, 38.93, 40.37, 39.27, 41.87, 40.27, 41.5, 41.57, 41.37, 42.2, 43.3, 40.27, 42.0, 40.83, 43.3, 36.0, 40.53, 38.27]
fed_mvsign = [28.5, 33.43, 34.33, 36.03, 37.7, 39.33, 43.07, 46.87, 47.13, 49.43, 50.4, 50.13, 49.5, 49.87, 51.53, 51.3, 50.17, 50.67, 50.87, 52.27, 53.33, 51.93, 53.13, 52.67, 53.1, 52.2, 51.0, 51.93, 53.17, 53.63, 53.33, 51.27, 51.63, 51.7, 52.47, 51.63, 51.67, 52.0, 52.03, 52.3, 52.77, 51.67, 51.5, 52.63, 51.93, 52.2, 51.57, 50.87, 51.67, 51.87, 51.57, 51.23, 50.8, 52.43, 52.17, 52.13, 51.0, 50.8, 51.27, 51.63, 51.63, 51.27, 51.13, 52.13, 52.93, 52.27, 50.93, 51.3, 51.63, 52.03, 51.3, 50.73, 51.57, 50.33, 50.1, 49.57, 49.7, 50.33, 50.47, 49.67, 49.9, 50.57, 50.43, 50.53, 49.57, 50.07, 49.53, 50.23, 50.37, 50.43, 49.9, 50.23, 50.03, 50.37, 50.53, 49.53, 49.57, 50.07, 49.73, 49.77, 49.8, 50.2, 50.37, 50.23, 50.07, 50.23, 50.37, 50.37, 50.3, 50.03, 50.13, 50.47, 50.6, 50.63, 50.9, 51.13, 50.87, 50.83, 50.4, 50.57, 50.17, 50.07, 50.07, 49.9, 49.87, 50.2, 50.3, 50.57, 50.43, 50.4, 50.67, 50.5, 50.1, 50.27, 50.2, 50.27, 50.83, 50.63, 50.83, 51.13, 51.2, 51.13, 51.07, 51.03, 51.17, 50.87, 50.87, 50.73, 50.73, 50.83, 50.83, 50.87, 50.8, 50.8, 50.8, 50.9, 50.83, 50.8, 50.77, 50.83, 50.83, 50.8, 50.77, 50.77, 50.63, 50.67, 50.7, 50.7, 50.7, 50.73, 50.77, 50.77, 50.7, 50.7, 50.73, 50.67, 50.67, 50.67, 50.6, 50.63, 50.63, 50.6, 50.6, 50.63, 50.7, 50.8, 50.87, 50.83, 50.9, 50.83, 50.87, 50.9, 50.93, 50.93, 50.93, 50.87, 50.9, 50.83, 50.77, 50.8]
fed_rlrsign = [17.27, 18.4, 10.53, 10.0, 10.0, 10.0, 10.0, 17.3, 16.9, 12.33, 10.0, 10.0, 10.57, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.1, 10.03, 10.0, 10.0, 10.7, 10.2, 10.0, 10.77, 13.57, 14.83, 10.0, 10.37, 12.13, 10.0, 13.23, 9.27, 18.77, 22.13, 20.6, 20.63, 23.67, 16.67, 22.43, 21.43, 17.03, 19.73, 11.7, 10.0, 10.53, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.67, 10.7, 10.0, 17.0, 17.3, 10.57, 10.0, 10.0, 11.13, 10.13, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.47, 13.27, 14.2, 12.23, 12.47, 10.0, 10.0, 10.47, 16.3, 16.37, 14.57, 11.27, 10.37, 13.77, 14.47, 10.53, 10.4, 10.43, 10.5, 10.77, 10.63, 10.5, 10.33, 10.4, 10.27, 10.27, 10.33, 10.73, 10.93, 11.4, 11.4, 10.97, 10.73, 10.73, 10.63, 10.3, 10.2, 10.07, 10.13, 10.2, 10.17, 10.17, 10.17, 10.23, 10.47, 10.5, 10.67, 10.6, 10.57, 10.37, 10.3, 10.5, 10.5, 10.57, 10.37, 10.4, 10.43, 10.43, 10.6, 10.7, 10.7, 10.5, 10.4, 10.2, 10.1, 10.17, 10.17, 10.17, 10.2, 10.2, 10.2, 10.23, 10.2, 10.27, 10.3, 10.3, 10.33, 10.33, 10.4, 10.4, 10.47, 10.6, 10.6, 10.63, 10.7, 10.73, 10.73, 10.73, 10.9, 10.93, 10.83, 10.93, 10.93, 11.03, 11.1, 11.1, 11.33, 11.33, 11.43, 11.37, 11.3, 11.17, 11.27, 11.37, 11.33, 11.33, 11.47, 11.4, 11.33, 11.13, 11.13, 11.27, 11.3, 11.4, 11.43, 11.43, 11.5]
fed_err = [23.87, 34.3, 40.77, 44.53, 46.67, 48.67, 49.1, 49.7, 50.4, 50.8, 51.3, 51.63, 52.23, 51.9, 51.67, 51.47, 51.4, 51.47, 51.1, 51.97, 52.17, 51.5, 51.7, 51.03, 50.73, 50.5, 50.8, 50.07, 50.27, 50.07, 50.37, 49.23, 49.27, 49.3, 49.67, 49.77, 48.93, 48.63, 48.63, 48.77, 49.17, 49.33, 48.4, 48.57, 49.2, 49.1, 48.93, 49.33, 49.47, 49.7, 49.73, 49.57, 49.47, 49.4, 49.23, 49.6, 49.6, 49.33, 49.2, 49.3, 49.33, 49.83, 49.77, 49.97, 49.77, 49.8, 49.83, 49.53, 49.63, 49.4, 49.53, 49.6, 49.6, 49.53, 49.6, 49.47, 49.47, 49.53, 49.53, 49.6, 49.6, 49.6, 49.53, 49.6, 49.63, 49.63, 49.67, 49.67, 49.73, 49.67, 49.73, 49.73, 49.77, 49.8, 49.73, 49.8, 49.8, 49.73, 49.7, 49.7, 49.7, 49.67, 49.67, 49.73, 49.73, 49.7, 49.7, 49.7, 49.67, 49.67, 49.67, 49.67, 49.63, 49.63, 49.67, 49.63, 49.6, 49.6, 49.63, 49.67, 49.67, 49.67, 49.63, 49.6, 49.63, 49.63, 49.67, 49.67, 49.67, 49.67, 49.67, 49.73, 49.73, 49.73, 49.73, 49.73, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.8, 49.83, 49.87, 49.83, 49.87, 49.9, 49.9, 49.97, 50.0, 50.0, 50.0, 50.0, 49.97, 49.97, 49.93, 49.9, 49.9, 49.9, 49.87, 49.87, 49.87, 49.87, 49.93, 49.93, 49.97, 49.93, 49.97, 49.97, 49.97, 50.0, 50.0, 50.0, 49.97, 49.97, 49.97, 49.97, 49.97, 49.97, 50.03, 50.03, 50.03, 50.03, 50.03, 50.07, 50.07, 50.07, 50.07, 50.03, 50.03, 50.03, 50.03, 50.03, 50.03]
fed_lfr = [23.87, 34.3, 40.77, 44.53, 46.67, 48.67, 49.1, 49.7, 50.4, 50.8, 51.3, 51.63, 52.23, 51.9, 51.67, 51.47, 51.4, 51.47, 51.1, 51.97, 52.17, 51.5, 51.7, 51.03, 50.73, 50.5, 50.8, 50.07, 50.27, 50.07, 50.37, 49.23, 49.27, 49.3, 49.67, 49.77, 48.93, 48.63, 48.63, 48.77, 49.17, 49.33, 48.4, 48.57, 49.2, 49.1, 48.93, 49.33, 49.47, 49.7, 49.73, 49.57, 49.47, 49.4, 49.23, 49.6, 49.6, 49.33, 49.2, 49.3, 49.33, 49.83, 49.77, 49.97, 49.77, 49.8, 49.83, 49.53, 49.63, 49.4, 49.53, 49.6, 49.6, 49.53, 49.6, 49.47, 49.47, 49.53, 49.53, 49.6, 49.6, 49.6, 49.53, 49.6, 49.63, 49.63, 49.67, 49.67, 49.73, 49.67, 49.73, 49.73, 49.77, 49.8, 49.73, 49.8, 49.8, 49.73, 49.7, 49.7, 49.7, 49.67, 49.67, 49.73, 49.73, 49.7, 49.7, 49.7, 49.67, 49.67, 49.67, 49.67, 49.63, 49.63, 49.67, 49.63, 49.6, 49.6, 49.63, 49.67, 49.67, 49.67, 49.63, 49.6, 49.63, 49.63, 49.67, 49.67, 49.67, 49.67, 49.67, 49.73, 49.73, 49.73, 49.73, 49.73, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.8, 49.83, 49.87, 49.83, 49.87, 49.9, 49.9, 49.97, 50.0, 50.0, 50.0, 50.0, 49.97, 49.97, 49.93, 49.9, 49.9, 49.9, 49.87, 49.87, 49.87, 49.87, 49.93, 49.93, 49.97, 49.93, 49.97, 49.97, 49.97, 50.0, 50.0, 50.0, 49.97, 49.97, 49.97, 49.97, 49.97, 49.97, 50.03, 50.03, 50.03, 50.03, 50.03, 50.07, 50.07, 50.07, 50.07, 50.03, 50.03, 50.03, 50.03, 50.03, 50.03]
fed_fleam = [10.0, 15.63, 11.37, 10.93, 15.33, 11.2, 19.73, 17.67, 18.37, 20.67, 18.87, 17.07, 21.37, 25.37, 25.0, 20.93, 25.0, 19.97, 23.2, 21.77, 27.97, 28.13, 19.6, 32.4, 23.43, 38.57, 41.0, 25.5, 17.77, 33.97, 28.63, 39.9, 39.93, 45.5, 43.1, 41.07, 37.73, 32.37, 33.3, 44.63, 44.53, 25.4, 37.87, 45.97, 45.83, 33.0, 43.0, 45.4, 41.27, 47.33, 49.03, 48.17, 39.5, 47.63, 46.07, 48.37, 51.07, 49.93, 50.4, 43.5, 44.03, 47.23, 43.67, 47.7, 48.5, 48.3, 50.0, 51.13, 49.17, 50.27, 48.47, 48.07, 45.9, 49.5, 45.53, 46.9, 50.13, 46.33, 47.83, 38.93, 42.2, 49.03, 48.17, 49.13, 47.77, 48.93, 49.9, 51.47, 48.73, 50.4, 50.3, 48.5, 51.97, 49.33, 46.07, 48.13, 51.73, 51.1, 50.87, 52.53, 49.63, 46.83, 48.5, 50.7, 48.9, 52.03, 48.6, 45.0, 52.1, 52.03, 52.6, 53.0, 46.2, 41.7, 49.07, 47.63, 46.7, 50.97, 49.63, 53.2, 50.33, 51.63, 52.97, 50.73, 51.4, 50.1, 46.97, 51.4, 51.43, 51.1, 48.7, 51.27, 54.67, 53.73, 54.17, 51.33, 53.07, 48.4, 51.33, 52.67, 54.27, 54.1, 46.97, 51.7, 43.5, 50.17, 52.0, 53.57, 52.2, 48.2, 53.43, 49.7, 52.87, 52.57, 51.07, 53.87, 53.63, 53.77, 53.67, 51.07, 53.77, 55.23, 49.7, 51.57, 54.57, 52.53, 52.5, 45.13, 54.63, 50.3, 52.4, 46.37, 52.4, 52.9, 54.37, 53.9, 55.47, 54.9, 52.87, 50.37, 54.23, 53.3, 50.93, 54.87, 51.97, 53.4, 52.5, 51.33, 52.2, 50.9, 51.3, 53.6, 50.67, 53.07, 52.23, 52.37, 53.13, 53.93, 53.9, 51.4]
fed_trust = [10.9, 26.53, 32.93, 37.23, 41.87, 44.43, 44.8, 46.6, 46.07, 48.2, 47.73, 49.13, 49.67, 50.6, 50.1, 48.8, 49.33, 50.07, 50.9, 51.37, 50.27, 49.97, 51.53, 50.87, 50.47, 50.63, 50.47, 50.9, 50.33, 50.07, 50.43, 50.8, 51.23, 50.63, 51.17, 51.07, 51.0, 51.3, 51.17, 51.03, 51.3, 51.73, 51.7, 50.57, 50.93, 50.63, 51.0, 50.6, 50.9, 51.2, 50.83, 51.5, 51.2, 51.53, 51.53, 51.53, 51.57, 51.63, 51.53, 51.9, 51.83, 51.63, 52.7, 51.9, 52.03, 51.6, 52.13, 52.3, 52.17, 52.07, 52.17, 51.8, 52.63, 51.9, 52.17, 53.1, 52.93, 52.57, 52.93, 52.6, 52.7, 52.4, 52.43, 52.2, 52.43, 52.23, 52.37, 52.33, 52.77, 51.87, 52.07, 52.87, 52.37, 52.33, 52.4, 52.8, 52.43, 51.97, 52.53, 52.13, 51.87, 52.43, 52.0, 51.97, 52.73, 52.8, 52.33, 51.7, 51.97, 52.63, 52.5, 52.63, 53.23, 52.87, 52.73, 53.43, 53.07, 53.33, 52.7, 51.9, 52.73, 52.43, 52.1, 53.0, 52.73, 51.87, 52.77, 51.83, 52.77, 52.73, 52.67, 53.3, 52.27, 52.37, 52.3, 52.87, 52.77, 52.6, 52.33, 51.97, 52.53, 52.63, 52.83, 52.67, 52.67, 52.83, 52.7, 52.6, 52.3, 52.2, 51.87, 51.93, 52.33, 52.3, 52.03, 52.07, 51.67, 51.17, 51.2, 51.7, 52.67, 52.57, 52.57, 52.03, 52.6, 51.43, 51.03, 51.77, 52.23, 52.37, 53.17, 52.33, 52.27, 52.37, 51.97, 52.3, 50.93, 52.7, 52.23, 51.47, 52.3, 51.73, 52.7, 51.8, 52.0, 51.6, 51.6, 52.0, 52.27, 52.13, 52.1, 51.4, 51.7, 51.5, 51.7, 51.73, 52.17, 51.7, 52.4, 52.07]

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

plot_time_acc_attack("", data, "", save_path, plot_size="3")