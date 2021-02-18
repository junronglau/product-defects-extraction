"""
Topic model to get symptoms categories
Select relevant topics and their top n keywords
Filter for these keywords to extract defects
"""
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from comet_ml import Experiment

from dataloader.ds_data_loader import DataLoader
from preprocess.corex_preprocessor import CorexPreprocessor
from models.corex_model import CorexModel
from trainers.corex_trainer import CorexTrainer

from utils.utils import get_args
from utils.config import process_config


def generate_topics():
    # capture the config path from the run arguments then process the json configuration file
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
    preprocessor = CorexPreprocessor(data, config)
    preprocessor.prepare_data()

    print('Creating and training the Model...')
    model = CorexModel(config, preprocessor)
    trainer = CorexTrainer(model, preprocessor.get_data())
    trainer.train()

    print('Evaluating the model...')
    coherence_lst, avg_coherence = trainer.evaluate(preprocessor.get_data(), preprocessor.get_corpus())
    trainer.generate_topics()
    print("Coherence score: {score_lst} \nAvg coherence score: {avg_score}"
          .format(score_lst=coherence_lst,
                  avg_score=avg_coherence))

    print('Saving the trained model...')
    model.save()

    # Log the rest of the experiment
    metrics = {"coherence": avg_coherence}
    experiment.log_metrics(metrics)

    experiment.log_model(name=config.experiment.model_name,
                         file_or_folder=config.labels_generator.paths.save_model_path)


if __name__ == '__main__':
    generate_topics()
