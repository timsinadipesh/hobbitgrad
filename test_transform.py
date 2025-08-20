from hobbitgrad.modules import *

np.random.seed(42)

m = 5
features = 3

X = np.random.rand(m, features)

true_weights = np.array([11.0, 13.0, -16.5])
true_bias = 3.0

y = X @ true_weights + true_bias + np.random.randn(m) * 0.1
y = y.reshape(-1, 1)
# print("y:\n", y)

model = Model([
    Dense(3, 6), RELU(), 
    Dense(6, 3), RELU(), 
    Dense(3, 1)
    ])
# print(model)

loss_fn = MSELoss()
optimizer = SGD(model)
epochs = 100

for epoch in range(epochs):
    # forward
    y_pred = model.forward(X)
    loss = loss_fn.forward(y, y_pred)

    # backward
    d_out = loss_fn.backward()
    model.backward(d_out)

    # update params
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss}")

# final prediction
y_pred = model.forward(X)
print("\nFinal predictions:\n", y_pred)
print("True y:", y)
print("Loss:", loss_fn.forward(y, y_pred))
