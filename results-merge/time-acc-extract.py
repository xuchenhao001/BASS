import os

from utils.parse_output import latest_acc_by_timeline


def main():
    sampling_frequency = 3  # sampling frequency (seconds)
    final_time = 300
    exp_node_number = "first-compare"
    model_name = "cnn"
    dataset_name = "cifar"
    experiment_names = ["fed_avg", "fed_sync", "fed_sync_sgd"]

    for path, dirs, files in os.walk("output"):
        if path.endswith(model_name + "-" + dataset_name) and exp_node_number in path:
            for experiment_name in experiment_names:
                experiment_path = os.path.join(path, experiment_name)
                acc_avg_list = latest_acc_by_timeline(experiment_path, sampling_frequency, final_time)
                print(experiment_name, "=", acc_avg_list)


if __name__ == "__main__":
    main()
