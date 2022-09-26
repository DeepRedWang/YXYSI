import sys
import datetime

from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from Model.transformer_module import SI
from Utils.GetData import GetData, GetTestData

from Utils.Time import Timer
from Utils.Train import TrainSI
from Utils.makefile import makefile

def train(Project_name="YXY_SI"):
    """config information"""
    cuda0 = "cuda:0" if torch.cuda.is_available() else "cpu"

    ################
    """hyper-parameters"""
    batch_size = 4
    learning_rate = 1e-3
    epoch_num = 500
    warmup_step = 100
    train_valid_rate = 0.9

    """These configs must be modified every time."""
    paramfilename = f'{Project_name}_lr{learning_rate}_bs{batch_size}_wu{warmup_step}_{epoch_num}epoch'
    test_name1 = 'Test'
    is_training = True

    ################
    time = Timer()
    device = cuda0
    test_device = "cpu"
    print(f"Using {device} device")

    """File configs including TrainLog, Param and Results."""
    if is_training:
        writer = SummaryWriter(f"TrainLog/{paramfilename}/{datetime.datetime.today().strftime('%Y-%m-%d-%H_%M_%S')}")
    path = 'Param/' + paramfilename
    test_path1 = 'Results/' + paramfilename
    test_path = 'Results/' + paramfilename + f'/{test_name1}'
    makefile(path)
    makefile(test_path1)
    makefile(test_path)


    """Load Training_Validing Data"""
    train_valid_path = 'Data/Training_Validing'
    train_in_data_file = 'Dataset_echo_810.mat'
    train_label_file = 'Dataset_label_810.mat'
    training_file = [train_in_data_file, train_label_file]

    if is_training:
        dataset = GetData(trainingpath=train_valid_path, trainingfile=training_file, device=device)
        dataset_num = len(dataset)
        train_dataset, valid_dataset = random_split(
            dataset=dataset,
            lengths=[int(dataset_num * train_valid_rate), dataset_num - int(dataset_num * train_valid_rate)]
        )

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    """model"""
    model = SI(2, 400, 1000, 4, 0.1)
    model.to(device=device)

    """loss function and optimizer"""
    loss_fn = []
    loss_fn1 = nn.MSELoss()
    loss_fn.append(loss_fn1)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
    lambda1 = lambda epoch: min((epoch + 1) ** (-0.5), (epoch + 1) * warmup_step ** (-1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    """training"""
    Train = TrainSI()
    if is_training:
        time.start()

        for epoch_index in range(epoch_num):
            train_loss = Train.training(train_dataloader, model, loss_fn, batch_size, optimizer, scheduler)
            valid_loss = Train.validing(valid_dataloader, model, loss_fn, batch_size)
            writer.add_scalar('train/loss', train_loss, epoch_index + 1)
            writer.add_scalar('valid/loss', valid_loss, epoch_index + 1)

            if epoch_index % 10 == 0:
                try:
                    for name, parameters in model.named_parameters():
                        writer.add_histogram(f'param/{name}', parameters.detach(), epoch_index + 1)
                        writer.add_histogram(f'grad/{name}', parameters.grad.data, epoch_index + 1)
                except:
                    print(f"{epoch_index}'s histogram has something wrong.")
                    sys.exit(1)

            if epoch_index == 0:
                valid_loss_flag = valid_loss
                param_dict = model.state_dict()
            if valid_loss_flag > valid_loss:
                valid_loss_flag = valid_loss
                flag = epoch_index
                param_dict = model.state_dict()

            if (epoch_index % (epoch_num / 10) == 0) & (epoch_index != 0):
                torch.save(param_dict,
                           f"{path}/bs{batch_size}_"
                           f"lr{str(learning_rate).split('.')[-1]}_"
                           f"eph{epoch_num}"
                           f"_{flag}_minvalidloss.pth")
        print('Training and validating have been finished!')
        print(f'Total training and validating time is: {time.end():.2f} secs')

        """Save model parameters"""
        torch.save(param_dict, f"{path}/bs{batch_size}_"
                               f"lr{str(learning_rate).split('.')[-1]}"
                               f"_eph{epoch_num}"
                               f"_{flag}_totalminvalidloss.pth")
        torch.save(flag, f"{path}/flag.pth")
        print(f'Param of {flag} epoch have been saved!')

    """Test model on measured data."""
    test_valid_path = 'Data/Testing'
    test_in_data_file = 'Dataset_echo_810.mat'
    test_label_file = 'Dataset_label_810.mat'
    testing_file = [test_in_data_file, test_label_file]
    test_dataset = GetTestData(testingpath=test_valid_path, testingfile=testing_file, device=test_device)

    test_bs = len(test_dataset)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=False)
    flag = torch.load(f"{path}/flag.pth")
    test_model = SI(2, 400, 1000, 4, 0.1)
    test_model.to(test_device)
    test_model.eval()
    #
    test_model.load_state_dict(torch.load(f"{path}/bs{batch_size}_"
                                          f"lr{str(learning_rate).split('.')[-1]}"
                                          f"_eph{epoch_num}"
                                          f"_{flag}_totalminvalidloss.pth"))

    Train.testing(test_dataloader, test_model, test_path, test_name1, loss_fn, test_bs)


if __name__ == '__main__':
    train()

