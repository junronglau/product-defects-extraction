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

    # Topic model to get symptoms categories
    # Select relevant topics and their top n keywords
    # Filter for these keywords to extract defects

    print('Creating the data loader...')
    data_loader = DataLoader(config.labels_generator)
    data = data_loader.get_data()

    print('Creating the Preprocessor...')
    preprocessor = LDAPreprocessor(config, data)
    preprocessor.preprocess_data()

    print('Creating and training the Model...')
    model = LDAModel(config, preprocessor.get_data, preprocessor.get_dictionary)
    trainer = LDATrainer(config, model)

    print('Evaluating the model...')
    trainer.evaluate()
    trainer.generate_topics()

    print('Saving the generated dataframes...')
    model.save()


if __name__ == '__main__':
    generate()