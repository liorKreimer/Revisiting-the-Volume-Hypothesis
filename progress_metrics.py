import numpy as np, matplotlib.pyplot as plt

data = np.load('progress_metrics.npy', allow_pickle=True)
runs, iters, e, q = zip(*data)

# keep only fully trained models
e16 = [ee for ee,qq in zip(e,q) if ee == 16]
q16 = [qq for ee,qq in zip(e,q) if ee == 16]

plt.figure(figsize=(7,5))
plt.hist(q16, bins=30, color='tomato', edgecolor='black')
plt.xlabel('Test accuracy (Q)')
plt.ylabel('Count')
plt.title('Distribution of test accuracy for fully trained models (E=16)')
plt.show()
