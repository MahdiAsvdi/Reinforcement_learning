import matplotlib.pyplot as plt

try:
    from IPython import display

    HAS_IPYTHON = True
except Exception:
    display = None
    HAS_IPYTHON = False

plt.ion()


def plot(scores, mean_scores):
    if HAS_IPYTHON:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)

    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(round(mean_scores[-1], 2)))

    plt.show(block=False)
    plt.pause(0.01)
