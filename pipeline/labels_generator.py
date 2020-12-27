"""
Topic model to get symptoms categories
Select relevant topics and their top n keywords
Filter for these keywords to extract defects
"""

import sys
sys.path.append('../')

from comet_ml import Experiment

from dataloader.data_loader import DataLoader
from preprocess.lda_preprocessor import LDAPreprocessor
from models.lda_model import LDAModel
from trainers.lda_trainer import LDATrainer

from utils.utils import get_args
from utils.config import process_config


def generate():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except ValueError:
        print("Missing or invalid arguments")
        exit(0)

    print("Logging experiment name: {name}".format(name=config.experiment.experiment_name))
    experiment = Experiment(
        api_key=config.experiment.api_key,
        project_name=config.experiment.project_name,
        workspace=config.experiment.workspace
    )
    experiment.set_name(config.experiment.experiment_name)
    params = config.labels_generator.model
    experiment.log_parameters(params)

    print('Creating the data loader...')
    data_loader = DataLoader(config.labels_generator.paths)
    data = data_loader.get_data()

    print('Creating the Preprocessor...')
    preprocessor = LDAPreprocessor(config, data)
    preprocessor.preprocess_data()

    print('Creating and training the Model...')
    model = LDAModel(config, preprocessor.get_data(), preprocessor.get_dictionary())
    trainer = LDATrainer(config, model)

    print('Evaluating the model...')
    coherence_score = trainer.evaluate()
    trainer.generate_topics()

    print('Saving the trained model...')
    model.save()

    # Log the rest of the experiment
    metrics = {"coherence": coherence_score}
    experiment.log_metrics(metrics)

    experiment.log_model(name=config.experiment.model_name,
                         file_or_folder=config.labels_generator.paths.save_model_path)


if __name__ == '__main__':
    generate()
