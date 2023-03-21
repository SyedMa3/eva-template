'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm


# Training
def train(model, device, train_loader, criterion, scheduler, optimizer):
    
    model.train()
    pbar = tqdm(train_loader)
    lr_trend = []
    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        y_pred = model(inputs)
        
        loss = criterion(y_pred, targets)
        loss.backward()
        
        optimizer.step()

        if scheduler:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                lr_trend.append(scheduler.get_last_lr()[0])

        train_loss += loss.item()
        
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()
        processed += len(inputs)

        pbar.set_description(desc= f'Batch_id={batch_idx} Loss={train_loss/(batch_idx+1):.5f} Accuracy={100*correct/processed:0.2f}')

    return 100*correct/processed, train_loss/(batch_idx+1), lr_trend




def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for (inputs, targets) in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
    return 100. * correct / len(test_loader.dataset), test_loss

    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc

def fit_model(net, optimizer, criterion, device, NUM_EPOCHS,train_loader, test_loader, use_l1=False, scheduler=None, save_best=False):

    training_acc, training_loss, testing_acc, testing_loss = list(), list(), list(), list()
    lr_trend= []

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, NUM_EPOCHS+1):
        print("EPOCH: {} (LR: {})".format(epoch, optimizer.param_groups[0]['lr']))

        train_acc, train_loss, lr_hist = train(net, device, train_loader, criterion, scheduler, optimizer)
        test_acc, test_loss = test(net, device, test_loader, criterion)
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)

        training_acc.append(train_acc)
        training_loss.append(train_loss)
        testing_acc.append(test_acc)
        testing_loss.append(test_loss)
        lr_trend.extend(lr_hist)

    if scheduler:
        return net, (training_acc, training_loss, testing_acc, testing_loss, lr_trend)
    else:
        return net, (training_acc, training_loss, testing_acc, testing_loss)
