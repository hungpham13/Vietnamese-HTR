from preprocessing.vocab import Vocab
import matplotlib.pyplot as plt
import math


def plot_class_instance(dataloader, title=""):
    batch = next(iter(dataloader))
    imgs = batch['img']
    targets = batch['tgt_output'].detach().numpy().tolist()
    # print(batch)
    x = list(zip(imgs, targets))
    ncols = 2
    nrows = math.floor(len(x) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 1.5))
    for (img, target), ax in zip(x, axes.flatten()):
        img = img.cpu().data
        # Display the image
        ax.imshow(img.squeeze(), cmap="gray")
        # print(target)
        ax.set_xlabel(Vocab().decode(target))
    plt.show()
