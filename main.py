from src.models import TrainingModel


configs = {
    'dataset_adress': 'ted_hrlr_translate/pt_to_en',
    'buffer_size': 20000,
    'batch_size': 64
}

training_model = TrainingModel(configs)