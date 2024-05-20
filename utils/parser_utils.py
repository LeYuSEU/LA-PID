def get_args():
    import argparse
    import os
    import torch
    parser = argparse.ArgumentParser(description='Welcome to the L2F training and inference system')

    parser.add_argument('--batch_size', nargs="?", type=int, help='Batch_size for experiment')
    parser.add_argument('--image_height', nargs="?", type=int)
    parser.add_argument('--image_width', nargs="?", type=int)
    parser.add_argument('--image_channels', nargs="?", type=int)
    parser.add_argument('--gpu_to_use', type=int)
    parser.add_argument('--indexes_of_folders_indicating_class', nargs='+')
    parser.add_argument('--train_val_test_split', nargs='+')
    parser.add_argument('--num_dataprovider_workers', nargs="?", type=int)
    parser.add_argument('--max_models_to_save', nargs="?", type=int)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--reset_stored_paths', type=str)
    parser.add_argument('--experiment_name', nargs="?", type=str)
    # [latest, -2, -1]
    parser.add_argument('--continue_from_epoch', nargs="?", type=str, help='Continue from ck-point of epoch')
    parser.add_argument('--dropout_rate_value', type=float, help='Dropout_rate_value')
    parser.add_argument('--num_target_samples', type=int, help='query samples per class')
    parser.add_argument('--second_order', type=str, help='Dropout_rate_value')
    parser.add_argument('--total_epochs', type=int, help='Number of epochs per experiment')
    parser.add_argument('--total_iter_per_epoch', type=int, help='Number of iters per epoch')
    parser.add_argument('--min_learning_rate', type=float, help='Min learning rate')
    parser.add_argument('--meta_learning_rate', type=float, help='Learning rate of overall MAML system')
    parser.add_argument('--norm_layer', type=str, default="batch_norm")
    parser.add_argument('--max_pooling', type=str)
    parser.add_argument('--per_step_bn_statistics', type=str)
    parser.add_argument('--num_classes_per_set', type=int, help='Number of classes to sample per set')
    parser.add_argument('--num_samples_per_class', type=int, help='Number of samples per class to sample')
    parser.add_argument('--number_of_training_steps_per_iter', type=int, help='num step per iter')
    parser.add_argument('--number_of_evaluation_steps_per_iter', type=int, help='num step per val-iter')
    parser.add_argument('--cnn_num_filters', type=int, help='cnn channels of output')
    parser.add_argument('--cnn_blocks_per_stage', type=int, help='cnn block per stage')
    parser.add_argument('--backbone', type=str, help='Base learner architecture backbone')
    parser.add_argument('--LAPID', type=str, help='Whether to perform adaptive inner-loop optimization')
    parser.add_argument('--random_init', type=str, help='Whether to use random initialization')

    # ======================== json 配置文件中没有的参数 ================================
    parser.add_argument('--reset_stored_filepaths', type=str, default="False")
    parser.add_argument('--reverse_channels', type=str, default="False")
    parser.add_argument('--num_of_gpus', type=int, default=1)
    parser.add_argument('--samples_per_iter', nargs="?", type=int, default=1)
    parser.add_argument('--labels_as_int', type=str, default="False")
    parser.add_argument('--seed', type=int, default=104)
    parser.add_argument('--architecture_name', nargs="?", type=str)
    parser.add_argument('--meta_opt_bn', type=str, default="False")
    parser.add_argument('--task_learning_rate', type=float, default=0.1, help='Learning rate per task gradient step')
    parser.add_argument('--cnn_num_blocks', type=int, default=4, help='cnn layers per block')
    parser.add_argument('--name_of_args_json_file', type=str,
                        default="FC100+4conv+5w5s15q+PID+maml.json")

    args = parser.parse_args()
    args_dict = vars(args)

    # 读取 json 配置文件
    if args.name_of_args_json_file is not "None":
        args_dict = extract_args_from_json('experiment_config/' + args.name_of_args_json_file, args_dict)

    for key in list(args_dict.keys()):
        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False

        if key == "dataset_path":
            # ============================== mini-imagenet
            if 'mini_imagenet' in args.dataset_name:
                os.environ['DATASET_DIR'] = '/data2/YuLe/03-予你/ALFA/dataset/'
                # os.environ['DATASET_DIR'] = 'D:/dataset/01-fewshot/mini_imagenet_full_size/'
            # ================================  CIFAR-FS
            elif 'cifar100' in args.dataset_name:
                # os.environ['DATASET_DIR'] = 'D:/dataset/01-fewshot/cifar100/CIFAR-FS/'
                os.environ['DATASET_DIR'] = '/home/lhq/code/CIFAR-FS/'
            # ============================ FC100
            elif 'FC100' in args.dataset_name:
                # os.environ['DATASET_DIR'] = '/home/lhq/code/FC100/'
                os.environ['DATASET_DIR'] = 'D:/Dataset/01_FewShot/FC100'
            # ============================= omniglot
            elif 'omniglot' in args.dataset_name:
                os.environ['DATASET_DIR'] = 'D:/dataset/omniglot_dataset'
            elif 'tiered-imagenet' in args.dataset_name:
                os.environ['DATASET_DIR'] = '/data2/YuLe/03-予你/dataset/tiered-imagenet'
                # os.environ['DATASET_DIR'] = 'D:/Dataset/01_FewShot/tiered-imagenet'

            args_dict[key] = os.path.join(os.environ['DATASET_DIR'], args_dict[key])
            print(key, os.path.join(os.environ['DATASET_DIR'], args_dict[key]))

        print(key, args_dict[key], type(args_dict[key]))

    args = Bunch(args_dict)

    args.use_cuda = torch.cuda.is_available()

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print("use GPU", device)
        print("GPU ID {}".format(args.gpu_to_use))
    else:
        print("use CPU")
        device = torch.device('cpu')

    return args, device


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
        # if "continue_from" in key:
        #     pass
        # if "gpu_to_use" in key:
        #     pass
        # else:
        # args_dict[key] = summary_dict[key]
    args_dict['experiment_name'] = f"{args_dict['dataset_name']}+{args_dict['backbone']}+5w{args_dict['num_samples_per_class']}s15q+PID+maml+bs{args_dict['batch_size']}+minLR{args_dict['min_learning_rate']}+metaLR{args_dict['meta_learning_rate']}+epoches{args_dict['total_epochs']}+meanStd"
    # args_dict['experiment_name'] = "FC100+5w5s+ResNet12+55.66+PID+maml+bs4+minLR1e-05+metaLR0.001+epoches50+meanStd"
    
    return args_dict
