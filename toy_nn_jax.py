import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

numPixels = 2 #number of pixels in mask per pixel in input image X 

def forwardPass(X, mask, transformationMatrix):
    X = jnp.exp(1j * jnp.pi * X) #phase encoding, can be switched on or off
    X = jnp.repeat(X, numPixels, axis=1)
    maskedInput = X * mask                              # E_in[k, n] = X[k, n] * M[n]   (N, 64)
    fieldOut = transformationMatrix @ maskedInput.T     # E_out = T E_in                (64, N), complex
    intensity = fieldOut.real**2 + fieldOut.imag**2     # I = |E_out|^2                 (64, N)
    return intensity.T                                  # one row of 64 intensities per image (N, 64)

def classScores(intensity):
    return intensity[:, :60].reshape(-1, 10, 6).sum(axis=2)   # reshapes the intensity to be a vector of length 10, a score for each digit, 0-5 is 0; 6-11 is 1...

def lossFn(mask, X, Y, transformationMatrix):
    scores = classScores(forwardPass(X, mask, transformationMatrix))
    return optax.softmax_cross_entropy_with_integer_labels(scores, Y).mean()

def accuracy(mask, X, Y, transformationMatrix):
    scores = classScores(forwardPass(X, mask, transformationMatrix))
    return jnp.mean(jnp.argmax(scores, axis=1) == Y)


digits = load_digits()
X = digits.data.astype(np.float32) #matrix: row=sample/vecIn, column=features
X = X/16 #normalising
Y = digits.target.astype(np.int64) #vector of target digits

# split: 1257 train / 270 validation / 270 test, classes kept approximately balanced
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=270, stratify=Y, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=270, stratify=Y_temp, random_state=0)

mask = np.random.rand(numPixels*len(X[0])).astype(np.float32) #one shared trainable mask, 64 weights

# fixed complex transmission matrix, generated once and never updated
transformMatrix = (np.random.randn(numPixels*len(X[0]), numPixels*len(X[0])) + 1j*np.random.randn(numPixels*len(X[0]), numPixels*len(X[0])))/np.sqrt(2*numPixels*len(X[0]))
transformMatrix = transformMatrix.astype(np.complex64)

X_train, X_val, X_test = jnp.array(X_train), jnp.array(X_val), jnp.array(X_test)
Y_train, Y_val, Y_test = jnp.array(Y_train, dtype=jnp.int32), jnp.array(Y_val, dtype=jnp.int32), jnp.array(Y_test, dtype=jnp.int32)
mask = jnp.array(mask)
transformMatrix = jnp.array(transformMatrix)

# JAX time: training the mask
epochs = 10000
learningRate = 0.001

optimizer = optax.adam(learningRate)
optState = optimizer.init(mask)

print(f"initial (random mask)  train acc {accuracy(mask, X_train, Y_train, transformMatrix):.3f}  "
      f"val acc {accuracy(mask, X_val, Y_val, transformMatrix):.3f}")

@jax.jit
def step(mask, optState):
    # full training set used for every update
    loss, grads = jax.value_and_grad(lossFn)(mask, X_train, Y_train, transformMatrix)
    updates, optState = optimizer.update(grads, optState, mask)
    mask = optax.apply_updates(mask, updates)
    trainAcc = accuracy(mask, X_train, Y_train, transformMatrix)
    valAcc = accuracy(mask, X_val, Y_val, transformMatrix)   # validation is only monitored, never used for updates
    return mask, optState, loss, trainAcc, valAcc

lossHistory = []
trainAccHistory = [] 
valAccHistory = []
for epoch in range(epochs):
    mask, optState, loss, trainAcc, valAcc = step(mask, optState)
    lossHistory.append(loss)
    trainAccHistory.append(trainAcc)
    valAccHistory.append(valAcc)
    if epoch % 1000 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:5d}  loss {float(loss):.4f}  train acc {float(trainAcc):.3f}  val acc {float(valAcc):.3f}")

lossHistory = np.array(lossHistory)
trainAccHistory = np.array(trainAccHistory)
valAccHistory = np.array(valAccHistory)

print(f"final test accuracy: {accuracy(mask, X_test, Y_test, transformMatrix):.3f}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(lossHistory)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy loss (train)")
ax1.set_title("Training loss")

ax2.plot(trainAccHistory, label="train")
ax2.plot(valAccHistory, label="validation")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy")
ax2.legend()

plt.tight_layout()

k = np.random.randint(len(X_test))
image = X_test[k:k+1] # keep the batch axis: shape (1, 64)
trueLabel = int(Y_test[k])

scores = classScores(forwardPass(image, mask, transformMatrix))[0] # the 10 long score vector
predicted = int(jnp.argmax(scores))
print(f"scores: {np.round(np.array(scores), 3)}")
print(f"true label: {trueLabel}, predicted: {predicted}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax1.imshow(np.array(image).reshape(8, 8), cmap='gray')
ax1.set_title(f"Input Image (True Label: {trueLabel})")
ax1.axis('off')

indices = np.arange(10)
colours = ['tab:orange' if i == predicted else 'tab:blue' for i in indices]
ax2.bar(indices, np.array(scores), color=colours)
ax2.set_xticks(indices)
ax2.set_xlabel("Digit class (0-9)")
ax2.set_ylabel("Score (summed intensity)")
ax2.set_title(f"Class scores (Predicted: {predicted})")

plt.tight_layout()
plt.show()
