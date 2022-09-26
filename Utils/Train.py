from scipy.io import savemat, loadmat
import torch
import numpy as np

class TrainSI():

    def __init__(self):
        pass

    def training(self, train_dataloader, model, loss_fn, bs, optimizer, scheduler):
        model.train()
        train_loss = 0
        num_batches = len(train_dataloader)
        for batch_index, data in enumerate(train_dataloader):
            Y = data['echo']
            X = data['target']
            out = model(Y)
            # X = X.softmax(dim=1)
            # loss = loss_fn[1](out, X)
            loss = loss_fn[0](out, X)
            loss1 = loss_fn[0](out, X)/loss_fn[0](X, torch.zeros_like(X))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss1.item()/bs

        train_loss = 10 * torch.log10(torch.tensor(train_loss) / torch.tensor(num_batches))

        return train_loss

    def validing(self, valid_dataloader, model, loss_fn, bs):
        model.eval()
        valid_loss = 0
        num_batches = len(valid_dataloader)
        with torch.no_grad():
            for batch_index, data in enumerate(valid_dataloader):
                Y = data['echo']
                X = data['target']
                out = model(Y)
                loss = loss_fn[0](out, X)/loss_fn[0](X, torch.zeros_like(X))
                valid_loss += loss.item()/bs

        valid_loss = 10 * torch.log10(torch.tensor(valid_loss) / torch.tensor(num_batches))

        return valid_loss

    def testing(self, test_dataloader, test_model, test_path, test_name1, loss_fn, bs):
        test_model.eval()
        test_loss = 0
        num_batches = len(test_dataloader)
        print(num_batches)
        for batch_index, data in enumerate(test_dataloader):
            Y = data['echo']
            X = data['target']
            out = test_model(Y)

            loss = loss_fn[0](out, X)/loss_fn[0](torch.zeros_like(X), X)

            test_loss += loss.item()/bs

            out = np.array(out.detach().cpu())
            savemat(f'{test_path}/' + f'{test_name1}.mat', {f'{test_name1}': out})
        valid_loss = 10 * torch.log10(torch.tensor(test_loss) / torch.tensor(num_batches))
        print(valid_loss)
