from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from comet_ml import Experiment

from dataloader.ds_data_loader import DataLoader
from preprocess.corex_preprocessor import CorexPreprocessor
from preprocess.textrank_preprocessor import TextRankPreprocessor
from models.corex_model import CorexModel
from models.textrank_model import TextRankModel
from trainers.corex_trainer import CorexTrainer
from trainers.textrank_trainer import TextRankTrainer

from utils.utils import get_args
from utils.config import process_config


def generate_categories():
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

    print('Creating the data loader...')
    data_loader = DataLoader(config.defects_summarizer.paths)
    train_data, test_data = data_loader.get_data()

    print('Creating the Preprocessor...')
    preprocessor = CorexPreprocessor(train_data, config)
    preprocessor.prepare_data()

    print('Loading and evaluating the Model...')
    model = CorexModel(config, preprocessor, seed=False)
    trainer = CorexTrainer(model, preprocessor.get_data())
    trainer.train()
    trainer.generate_topics()
    top_docs_df = trainer.get_top_documents(config.defects_summarizer.evaluate.extract_topics,
                                            preprocessor.get_raw_corpus(),
                                            config.defects_summarizer.evaluate.extraction_quantile,
                                            labels=True)

    print('Preprocessing the summarizer...')
    summary_preprocessor = TextRankPreprocessor(top_docs_df, n_docs=config.defects_summarizer.evaluate.n_docs)
    summary_preprocessor.prepare_data()

    print('Loading and evaluating the summarizer...')
    summary_model = TextRankModel(config)
    summary_trainer = TextRankTrainer(summary_model, summary_preprocessor)
    avg_prec, avg_recall, avg_f1 = summary_trainer.train_and_evaluate(test_data)

    # Log the rest of the experiment
    metrics = {"precision": avg_prec,
               "recall": avg_recall,
               "f1": avg_f1}
    experiment.log_metrics(metrics)

    experiment.log_model(name=config.experiment.model_name,
                         file_or_folder=config.labels_generator.paths.save_model_path)


if __name__ == '__main__':
    generate_categories()
