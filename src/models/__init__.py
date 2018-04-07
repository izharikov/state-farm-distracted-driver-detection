from models import simple_model, vgg16, vgg19, inception_v3, xception


def get_model(modelType, img_width, print_summary=False, fc_layers=None, dropout=None):
    if modelType == 'simple':
        return simple_model.get_model(print_summary, img_width)
    if modelType == 'vgg16':
        return vgg16.get_model(print_summary, img_width, fc_layers, dropout)
    if modelType == 'vgg19':
        return vgg19.get_model(print_summary, img_width, fc_layers, dropout)
    if modelType == 'inception_v3':
        return inception_v3.get_model(print_summary, img_width, fc_layers, dropout)
    if modelType == 'xception':
        return xception.get_model(print_summary, img_width, fc_layers, dropout)
    return None
