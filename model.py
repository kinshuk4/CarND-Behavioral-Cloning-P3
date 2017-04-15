import model_list as ml
import model_util as mu

import numpy as np
import data_util as du
from sklearn.model_selection import train_test_split


def main():
    data_subfolder_ids = [2]
    samples = du.read_all_logs(data_subfolder_ids)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = du.generator(train_samples, batch_size=32)
    validation_generator = du.generator(validation_samples, batch_size=32)

    print("Image Samples: " + str(len(train_samples)))
    # model_type="simple"
    # model_type="lenet"
    model_type="nvidia"
    print("Getting the model.........................."+model_type + "\n")
    model, epochs = ml.choose_model(model_type)

    mu.compile_model_generator(model, train_samples, validation_samples, train_generator, validation_generator,epochs)


if __name__ == "__main__":
    main()
