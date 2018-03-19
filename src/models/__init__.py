from models import simple_model, vgg


def get_model(modelType, img_width, print_summary=False):
    if modelType == 'simple':
        return simple_model.get_model(print_summary, img_width)
    if modelType == 'vgg':
        return vgg.get_model(print_summary, img_width)
    return None
