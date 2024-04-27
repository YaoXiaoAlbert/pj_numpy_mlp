from model import *

def train_model(model, X_train, y_train, X_val, y_val, learning_rate, decay_rate, beta, num_epochs, batch_size,activation='relu'):
    num_train = X_train.shape[0]
    num_batches = num_train // batch_size

    history_train_losses = []
    history_train_accuracies = []
    history_val_losses = []
    history_val_accuracies = []
    best_val_accuracy = 0.0
    best_model = model

    for epoch in range(num_epochs):
        random_indices = np.random.choice(num_train, num_train, replace=False)
        X_train_epoch = X_train[random_indices]
        y_train_epoch = y_train[random_indices]

        for i in range(num_batches):
            X_batch = X_train_epoch[i * batch_size : (i + 1) * batch_size]
            y_batch = y_train_epoch[i * batch_size : (i + 1) * batch_size]
            y_hat, cache = forward_propagation(model, X_batch,activation)
            # loss = compute_loss(y_batch, y_hat, model, beta)
            # accuracy = compute_accuracy(y_hat, y_batch)
            grads = backward_propagation(model, cache, X_batch, y_batch, y_hat, beta, activation)
            model = update_model(model, grads, learning_rate)

        y_train_hat, _ = forward_propagation(model, X_train,activation)
        train_loss = compute_loss(y_train, y_train_hat, model, beta)
        train_accuracy = compute_accuracy(y_train_hat, y_train)
        y_val_hat, _ = forward_propagation(model, X_val,activation)
        val_loss = compute_loss(y_val, y_val_hat, model, beta)
        val_accuracy = compute_accuracy(y_val_hat, y_val)
        if best_val_accuracy < val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model

        history_train_losses.append(train_loss)
        history_train_accuracies.append(train_accuracy)
        history_val_losses.append(val_loss)
        history_val_accuracies.append(val_accuracy)
        learning_rate *= decay_rate

        print(f'Epoch {epoch+1}/{num_epochs}: Training loss = {train_loss}, Training accuracy = {train_accuracy}, Testing loss = {val_loss}, Testing accuracy = {val_accuracy}')

    return model, best_model, history_train_losses, history_train_accuracies, history_val_losses, history_val_accuracies
