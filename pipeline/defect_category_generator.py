from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from dataloader.lg_data_loader import DataLoader
from preprocess.corex_preprocessor import CorexPreprocessor
from models.corex_model import CorexModel
from trainers.corex_trainer import CorexTrainer

from utils.utils import get_args
from utils.config import process_config


def generate_labels():
    # capture the config path from the run arguments then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except ValueError:
        print("Missing or invalid arguments")
        exit(0)

    print('Creating the data loader...')
    data_loader = DataLoader(config.labels_generator.paths)
    data = data_loader.get_data()

    print('Creating the Preprocessor...')
    preprocessor = CorexPreprocessor(data, config)
    preprocessor.prepare_data()

    print('Loading and evaluating the Model...')
    model = CorexModel(config, preprocessor, load=True)
    trainer = CorexTrainer(model, preprocessor.get_data())
    top_docs_df = trainer.get_top_documents(config.labels_generator.evaluate.extract_topics,
                                            preprocessor.get_raw_corpus(),
                                            config.labels_generator.evaluate.extraction_quantile)
    top_docs_df.to_csv(config.labels_generator.paths.save_data_path, index=False)


if __name__ == '__main__':
    generate_labels()
