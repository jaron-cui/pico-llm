import matplotlib.pyplot as plt


losses = []
with open('abs_losses.txt', 'r') as file:
    for line in file:
        # print(line.strip())
        losses.append(float(line.split()[-1]))

epoch_ticks = [625, 1250, 1875]
epoch_labels = ['Epoch 1', 'Epoch 2', 'Epoch 3']

textstr = f'Final Loss: {losses[-1]:.4f}'
plt.gca().text(
    0.95, 0.95, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='black')
)

plt.plot(losses, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Absolute Positional Embedding - Training Loss Over Steps')
plt.legend()
plt.grid(True)
plt.xticks(epoch_ticks, epoch_labels)
# plt.show()
plt.savefig('absolute_losses.png')