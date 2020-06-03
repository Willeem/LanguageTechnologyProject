from sklearn.model_selection import train_test_split


def main():
    # Load train-dev
    with open('data/traindev_deepL_en.txt') as traindev_deepL:
        Xtrain_dev_deepL = traindev_deepL.readlines()
    with open('data/traindev_src_en.txt') as traindev_src:
        Xtrain_dev_src = traindev_src.readlines()
    with open('data/traindev_src_de.txt') as traindev_de:
        Xtrain_dev_de = traindev_de.readlines()

    # Perform split
    Xtrain_deepL, Xdev_deepL, Xtrain_src, Xdev_src, Xtrain_de, Xdev_de = train_test_split(
        Xtrain_dev_deepL, Xtrain_dev_src, Xtrain_dev_de, test_size=0.20, random_state=42)

    # Save train
    with open('data/train_deepL_en.txt', 'w') as train_deepL:
        train_deepL.write(''.join(Xtrain_deepL))
    with open('data/train_src_en.txt', 'w') as train_src:
        train_src.write(''.join(Xtrain_src))
    with open('data/train_src_de.txt', 'w') as train_de:
        train_de.write(''.join(Xtrain_de))

    # Save dev
    with open('data/dev_deepL_en.txt', 'w') as dev_deepL:
        dev_deepL.write(''.join(Xdev_deepL))
    with open('data/dev_src_en.txt', 'w') as dev_src:
        dev_src.write(''.join(Xdev_src))
    with open('data/dev_src_de.txt', 'w') as dev_de:
        dev_de.write(''.join(Xdev_de))


if __name__ == "__main__":
    main()
