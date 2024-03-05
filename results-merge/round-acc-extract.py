from utils.parse_output import calculate_average_across_files


def extract_round_acc():
    experiment_names = ["fed_fleam", "fed_trust"]

    for exp_name in experiment_names:
        exp_result_path = "./output/DDoS/mlp-fmnist/{}/output".format(exp_name)
        files_numbers_mean_2d_np = calculate_average_across_files(exp_result_path)
        acc = [round(i, 2) for i in files_numbers_mean_2d_np[:, 5]]
        print(exp_name, "=", acc)

    # for path, dirs, files in os.walk("./output"):
    #     if path.endswith(model_name + "-" + dataset_name) and exp_node_number in path:
    #         for experiment_name in experiment_names:
    #             experiment_path = os.path.join(path, experiment_name)
    #             files_numbers_mean_2d_np = calculate_average_across_files(experiment_path)
    #             acc = [round(i, 2) for i in files_numbers_mean_2d_np[:, 5]]
    #             print(experiment_name, "=", acc)


def main():
    extract_round_acc()


if __name__ == "__main__":
    main()
