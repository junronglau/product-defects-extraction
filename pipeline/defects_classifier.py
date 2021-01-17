from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from comet_ml import Experiment
from dataloader.dc_data_loader import DataLoader
from preprocess.svm_preprocessor import SvmPreprocessor
from models.svm_model import SvmModel
from trainers.svm_trainer import SvmTrainer

from utils.utils import get_args
from utils.config import process_config


def defects_classifier():
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
    params = config.defects_classifier.model
    experiment.log_parameters(params)

    print('Creating the data loader...')
    data_loader = DataLoader(config)
    data = data_loader.get_data()

    print('Creating the Preprocessor...')
    preprocessor = SvmPreprocessor(*data)
    preprocessor.prepare_data()
    train_data = preprocessor.get_train_data()
    test_data = preprocessor.get_test_data()

    print('Loading and evaluating the Model...')
    model = SvmModel(config, load=False)
    trainer = SvmTrainer(model, **train_data)
    trainer.train()
    overall_scores = trainer.evaluate_all(**test_data)
    protocol_scores = trainer.evaluate_protocol(config.defects_classifier.evaluate.protocol.ratings, **test_data)

    print('Saving the trained model...')
    model.save()

    print("accuracy: {acc} \nprecision: {precision} \nrecall: {recall}".format(
        acc=overall_scores['accuracy'],
        precision=overall_scores['precision'],
        recall=overall_scores['recall']
    ))
    [print("{proto}: acc: {acc} prec: {prec} recall: {recall}"
           .format(proto=proto, acc=score['accuracy'], prec=score['precision'], recall=score['recall']))
           for proto, score in protocol_scores.items()]

    # Log the rest of the experiment
    experiment.log_metrics(overall_scores)

    experiment.log_model(name=config.experiment.model_name,
                         file_or_folder=config.defects_classifier.paths.save_model_path)


if __name__ == '__main__':
    defects_classifier()
