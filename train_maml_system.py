from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# CUDA_VISIBLE_DEVICES=2

if __name__ == '__main__':
    args, device = get_args()
    model = MAMLFewShotClassifier(args=args, device=device, im_shape=(2, 3, args.image_height, args.image_width))
    data = MetaLearningSystemDataLoader
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()
