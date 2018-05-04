from models import simple_model, vgg16, vgg19, inception_v3, xception, resnet50, resnet152, densenet, inception_v4, \
    inception_resnet_v2


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
    if modelType == 'resnet50':
        return resnet50.get_model(print_summary, img_width, fc_layers, dropout)
    if modelType == 'resnet152':
        return resnet152.get_model(print_summary, img_width, fc_layers, dropout)
    if modelType == 'densenet121' or modelType == 'densenet':
        return densenet.get_model(print_summary, img_width, fc_layers, dropout)
    if modelType == 'inception_v4':
        return inception_v4.get_model(print_summary, img_width, fc_layers, dropout)
    if modelType == 'inception_resnet_v2':
        return inception_resnet_v2.get_model(print_summary, img_width, fc_layers, dropout)
    return None
