def accuracy(predictions, labels, divide_by_nsamples=True):
    n_samples = labels.shape[0]
    n_correct = number_of_correct_samples(predictions, labels)

    if divide_by_nsamples:
        acc = n_correct.true_divide(n_samples)

    return acc


def number_of_correct_samples(predictions, labels):
    n_correct = (labels == predictions).sum()

    return n_correct
