from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from dataloader.dc_data_loader import DataLoader
from preprocess.twostep_preprocessor import TwoStepPreprocessor
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

    print('Creating the data loader...')
    data_loader = DataLoader(config)
    train_data, test_data = data_loader.get_data()

    print('Creating the Preprocessor...')
    preprocessor = TwoStepPreprocessor(train_data, test_data)
    preprocessor.prepare_data(evaluate=True)
    test_data = preprocessor.get_all_data()

    print('Loading and evaluating the Model...')
    model = SvmModel(config, load=True)
    trainer = SvmTrainer(model, preprocessor)
    predictions = trainer.generate_predictions(**test_data)
    train_data.iloc[predictions > 0.5].to_csv(config.defects_classifier.paths.save_data_path, index=False)


if __name__ == '__main__':
    defects_classifier()
