import os

from utils.parse_output import calculate_average_across_files


def extract_round_acc():
    exp_node_number = "v3"
    model_name = "cnn"
    dataset_name = "fmnist"

    experiment_names = ["fed_avg", "fed_efsign", "fed_sync", "fed_sync_sgd", "fed_sign", "local_train"]

    # time_results = {}
    print("{:15s}\t{}\t{}\t{}\t{}".format("", "round_time", "train_time", "test_time", "communication_time"))
    for path, dirs, files in os.walk("output"):
        if path.endswith(model_name + "-" + dataset_name) and exp_node_number in path:
            for experiment_name in experiment_names:
                experiment_path = os.path.join(path, experiment_name)
                files_numbers_mean_2d_np = calculate_average_across_files(experiment_path)
                round_time = [round(i, 2) for i in files_numbers_mean_2d_np[:50, 1]]
                train_time = [round(i, 2) for i in files_numbers_mean_2d_np[:50, 2]]
                test_time = [round(i, 2) for i in files_numbers_mean_2d_np[:50, 3]]
                communication_time = [round(i, 2) for i in files_numbers_mean_2d_np[:50, 4]]
                result_time_array = [
                    round(sum(round_time)/len(round_time), 2),
                    round(sum(train_time)/len(train_time), 2),
                    round(sum(test_time)/len(test_time), 2),
                    round(sum(communication_time)/len(communication_time), 2),
                ]
                # result_time_array = [
                #     round_time,
                #     train_time,
                #     test_time,
                #     communication_time,
                # ]
                # time_results[experiment_name] = result_time_array
                print("{:15s}:\t{}\t{}\t{}\t{}".format(experiment_name, result_time_array[0], result_time_array[1], result_time_array[2], result_time_array[3]))

    # time_types = ["round_time", "train_time", "test_time", "communication_time"]
    # for time_idx in range(len(time_types)):
    #     print("\n" + time_types[time_idx] + ": ")
    #     for experiment_name in experiment_names:
    #         print(experiment_name, "=", time_results[experiment_name][time_idx])


def main():
    extract_round_acc()


if __name__ == "__main__":
    main()
