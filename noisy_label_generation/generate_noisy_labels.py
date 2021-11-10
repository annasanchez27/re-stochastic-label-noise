import numpy as np
import random


def generate_symmetric_noisy_labels(labels, p=0.4):
    """Flip each label to any other class with a probability of p."""
    unique_labels = np.unique(labels)
    noisy_labels = [
        i if random.uniform(0, 1) > p else unique_labels[random.choice(unique_labels)]
        for i in labels
    ]
    return noisy_labels


def generate_asymmetric_noisy_labels_cifar10(labels, p=0.4):
    """Flip each label to a specific class with a probability of p.

    For CIFAR-10 the class that are linked are:
    TRUCK → AUTOMOBILE
    BIRD → AIRPLANE
    DEER → HORSE
    CAT ↔ DOG

    With:
    Airplane -> 0
    Automobile -> 1
    bird -> 2
    cat -> 3
    deer -> 4
    dog -> 5
    frog -> 6
    horse -> 7
    ship -> 8
    truck -> 9
    """
    labels_mapper = {
        0: 0,
        1: 1,
        2: 0,
        3: 5,
        4: 7,
        5: 3,
        6: 6,
        7: 7,
        8: 8,
        9: 1,
    }
    noisy_labels = [i if random.uniform(0, 1) > p else labels_mapper[i] for i in labels]
    return noisy_labels


def generate_asymmetric_noisy_labels_cifar100(labels, p=0.4):
    """Flip each label to a specific class with a probability of p.

    For CIFAR-100 the specific class is the next class.
    So, 0 → 1, 1 → 2, ... , n → 0
    """
    unique_labels = np.unique(labels)
    labels_mapper = {i: i + 1 if i + 1 in unique_labels else 0 for i in unique_labels}
    noisy_labels = [i if random.uniform(0, 1) > p else labels_mapper[i] for i in labels]
    return noisy_labels


def generate_noisy_labels(dataset="cifar10"):
    """Main generation script."""
    labels = np.load(f"labels/{dataset}/original/labels_{dataset}.npy")
    label_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Generate symmetric noise labels
    for fraction in label_fractions:
        noisy_labels = generate_symmetric_noisy_labels(labels, p=fraction)
        np.save(f"labels/{dataset}/symmetric/labels_{fraction}", noisy_labels)

    # Generate asymmetric noisy labels
    if dataset == "cifar10":
        for fraction in label_fractions:
            noisy_labels = generate_asymmetric_noisy_labels_cifar10(labels, p=fraction)
            np.save(f"labels/{dataset}/asymmetric/labels_{fraction}", noisy_labels)
    elif dataset == "cifar100":
        for fraction in label_fractions:
            noisy_labels = generate_asymmetric_noisy_labels_cifar100(labels, p=fraction)
            np.save(f"labels/{dataset}/asymmetric/labels_{fraction}", noisy_labels)


if __name__ == """__main__""":
    generate_noisy_labels("cifar10")
    generate_noisy_labels("cifar100")
