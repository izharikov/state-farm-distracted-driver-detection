from prediction.make_prediction import make_prediction


def getopts():
    from sys import argv
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


if __name__ == "__main__":
    myargs = getopts()
    mode = myargs['--mode']
    if mode == "predict":
        path_to_model = myargs['--path_to_model']
        output_file_csv = myargs['--output_file']
        result = make_prediction(path_to_model, output_file_csv)
    if mode == "predict_test":
        path_to_model = myargs['--path_to_model']
        output_file_csv = myargs['--output_file']
        result = make_prediction(path_to_model, output_file_csv, 10)
