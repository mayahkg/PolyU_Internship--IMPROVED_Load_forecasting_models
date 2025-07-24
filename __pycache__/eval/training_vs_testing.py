import matplotlib.pyplot as plt

# Training and testing loss data
epochs = list(range(1, 11))
training_loss = [
    0.1634052266765918,
    0.09750400705351717,
    0.0775635581064437,
    0.0679394865098099,
    0.06255944638646074,
    0.058912274782501516,
    0.05647065851926094,
    0.05440592889984449,
    0.052817361163241525,
    0.05145589051590789
]
testing_loss = [
    0.13848124502138012,
    0.08729798408846061,
    0.07313560186663554,
    0.0657184460377764,
    0.06081610002244512,
    0.057843350361855256,
    0.05559414057504563,
    0.053813912107476165,
    0.05253765257518916,
    0.05147488798857445
]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_loss, label='Training Loss', marker='o')
plt.plot(epochs, testing_loss, label='Testing Loss', marker='o')
plt.title('Training and Testing Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid()
plt.show()