#this file just runs a forward pass, no training
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def forwardPass(X, mask, transformationMatrix):
    maskedInput = X * mask.T
    outputMatrix = transformMatrix @ maskedInput.T
    return outputMatrix

digits = load_digits()
X = digits.data.astype(np.float32) #matrix: 1797 rows=sample/vecIn, 64 columns=features 
X = X/16 #normailising
Y = digits.target.astype(np.int64) #vector of target digits

mask = np.random.rand(len(X[0]),1) #creates a random mask to apply to the input data
transformMatrix = np.random.rand(len(X[0]), len(X[0])) #reservoir

outputMatrix = forwardPass(X, mask, transformMatrix)

print("shape of input: ", X.shape)
print("shape of target digits: ", Y.shape)
print("shape of mask: ", mask.shape)
print("shape of transformation matrix: ", transformMatrix.shape)
print("shape of output: ", outputMatrix.shape)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

# ax1.imshow(image_tensor.squeeze(), cmap='gray')
# ax1.set_title(f"Input Image (True Label: {label})")
# ax1.axis('off')

# indices = np.arange(len(outputMatrix))
# plt.bar(indices, outputMatrix)
# plt.xticks(indices)
# plt.xlabel("Output Index (0-9)")
# plt.ylabel("Intensity / Value")
# plt.title(f"Output Vector (True Label: {label})")
# plt.show()


