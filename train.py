import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

accuracy = BinaryAccuracy()
f1score = BinaryF1Score()

def training(net, loader, optimizer, loss_function):
    n_train = len(loader)
    train_loss = 0
    train_acc = 0
    train_f1 = 0
    net.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        output = net.forward(x_batch)
        y_pred = output[:,1:2,:,:]
        y_batch = torch.round(y_batch)
        loss = loss_function(y_pred, y_batch)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += accuracy(y_pred, y_batch).item()
        train_f1 += f1score(y_pred, y_batch).item()

    return net, {'train_loss': train_loss/n_train,
                 'train_acc': train_acc/n_train,
                 'train_f1': train_f1/n_train}


def validation(net, loader, loss_function):
    n_valid = len(loader)
    valid_loss = 0
    valid_acc = 0
    valid_f1 = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            output = net.forward(x_batch)
            y_pred = output[:,1:2,:,:]
            y_batch = torch.round(y_batch)
            val_loss = loss_function(y_pred, y_batch)
            valid_loss += val_loss.item()
            valid_acc += accuracy(y_pred, y_batch).item()
            valid_f1 += f1score(y_pred, y_batch).item()

    return net, {'valid_loss': valid_loss/n_valid,
                 'valid_acc': valid_acc/n_valid,
                 'valid_f1': valid_f1/n_valid}
        

def train_validation_loop(net, train_loader, valid_loader, epochs, optimizer, loss_function):
    train_loss_all, valid_loss_all = [], []
    train_acc_all, valid_acc_all = [], []
    train_f1_all, valid_f1_all = [], []

    for epoch in range(epochs):
        net, train_out = training(net, train_loader, optimizer, loss_function)
        train_loss_all.append(train_out['train_loss'])
        train_acc_all.append(train_out['train_acc'])
        train_f1_all.append(train_out['train_f1'])
        net.eval()

        net, val_out = validation(net, valid_loader, loss_function)
        valid_loss_all.append(val_out['valid_loss'])
        valid_acc_all.append(val_out['valid_acc'])
        valid_f1_all.append(val_out['valid_f1'])

        print(f'Epoch {epoch+1}/{epochs} done, train_loss={train_loss_all[epoch]}, valid_loss={valid_loss_all[epoch]}, \
              train_acc={train_acc_all[epoch]}, valid_acc={valid_acc_all[epoch]}')

    return net, {'train_loss': train_loss_all,
                 'train_acc': train_acc_all,
                 'train_f1': train_f1_all,
                 'valid_loss': valid_loss_all,
                 'valid_acc': valid_acc_all,
                 'valid_f1': valid_f1_all}