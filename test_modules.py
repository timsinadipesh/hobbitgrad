from hobbitgrad.modules import *

np.random.seed(42)

m = 10
input_dim = 3

X = np.random.rand(m, input_dim)

true_weights = np.array([11.0, 13.0, -16.5])
true_bias = 3.0

y = X @ true_weights + true_bias + np.random.randn(m) * 0.1
y = y.reshape(-1, 1)
# print("y:\n", y)

model = Model([
    Linear(3, 6), RELU(), 
    Linear(6, 3), RELU(), 
    Linear(3, 1)
    ])
# print(model)

loss_fn = MSELoss()
optimizer = SGD(model)
epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()

    # forward
    y_pred = model.forward(X)
    loss = loss_fn.forward(y, y_pred)

    print(f"Epoch {epoch}, Loss: {loss}")

    # backward
    d_out = loss_fn.backward()
    model.backward(d_out)

    # update params
    optimizer.step()


# final prediction
y_pred = model.forward(X)
# print("\nFinal predictions:\n", y_pred)
# print("True y:\n", y)
print("Final Loss:", loss_fn.forward(y, y_pred))

# gotta test & update __repr__ functions. create a new thing.
# always have tests for everything