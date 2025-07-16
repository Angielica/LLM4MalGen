import matplotlib.pyplot as plt


def plot_training_loss(losses, params):
    plt.figure()
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(params['training_loss_path'])
