from models import simple_model, vgg


def get_model(modelType, print_summary=False):
    if modelType == 'simple':
        return simple_model.get_model(print_summary)
    if modelType == 'vgg':
        return vgg.get_model(print_summary)
    return None
