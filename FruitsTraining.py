print("Importing the libraries")
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import os

from DbConnection import DbConnection
from FruitsDataset import FruitsDataset
from torch.utils.tensorboard import SummaryWriter

class FruitsTraining():
    writer={}
    current_model={}

    def launch(self):
        params = self.get_params()
        self.writer = SummaryWriter(f'runs/{params.experiment_name}')
        self.train(params)

    def get_params(self):
        class Opt:
            epochs=10
            print_every = 3
            batch_size=16
            model_name="fruits_model.pth"
            experiment_name= "fruits"
        opt = Opt();
        return opt


    def train(self, params):
        # print("Definir conexión a la BD")
        # connection= DbConnection()
        # print("Realizar Consulta de imágenes")
        # train_dataframe= connection.get_fruits(is_test=False)
        # test_dataframe= connection.get_fruits(is_test=True)
        # labels_dataframe= connection.get_labels()
        # labels_list= labels_dataframe.iloc[:, 0]
        # print("Definir datasets")
        # train_dataset = FruitsDataset(train_dataframe)
        # test_dataset = FruitsDataset(test_dataframe)
        # print("Definir Data loaders")
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)
        train_loader, test_loader = self.load_split_train_test(.2)
        print(test_loader.dataset.classes)

        print("Obtener e imprimir la forma de los datos")
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        print("images.shape")
        print(images.shape)
        print("labels.shape")
        print(labels.shape)
        device = torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
        model = models.resnet50(pretrained=True)
        print(model)
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 132),
                                        nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
        model.to(device)
        steps = 0
        running_loss = 0
        for epoch in range(params.epochs):
            for images, labels in train_loader:
                steps += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model.forward(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % params.print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for images, labels in test_loader:
                            images, labels = images.to(device), labels.to(device)
                            logps = model.forward(images)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_loss = running_loss/len(train_loader)
                    self.writer.add_scalar('training loss',train_loss)

                    test_loss=test_loss/len(test_loader)
                    self.writer.add_scalar('test loss',train_loss)

                    self.writer.add_scalar('epoch', epoch+1)

                    self.writer.add_scalar('Test acuracy', accuracy/len(test_loader))
                    print(f"Epoch {epoch+1}/{params.epochs}.. "
                        f"Train loss: {running_loss/params.print_every:.3f}.. "
                        f"Test loss: {test_loss/len(test_loader):.3f}.. "
                        f"Test accuracy: {accuracy/len(test_loader):.3f}")
                    running_loss = 0
                    model.train()
                    self.current_model = model

        self.save(model, params.model_name)
        self.test(test_loader=test_loader, model=model)

    def save(self, model, path):
        torch.save(model, path)

    def test(self, test_loader, model):
        print("Getting predictions on test set and measuring the performance")
        correct_count, all_count = 0, 0
        for images,labels in test_loader:
            for i in range(len(labels)):
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                img = images[i].view(1, 1, 28, 28)
                with torch.no_grad():
                    logps = model(img)
                ps = torch.exp(logps)
                probab = list(ps.cpu()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.cpu()[i]
                if(true_label == pred_label):
                    correct_count += 1
                all_count += 1

        print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy =", (correct_count/all_count))

    def images_to_probs(self, net, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = net(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_classes_preds(self, net, images, labels, classes):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self.images_to_probs(net, images)
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
            self.matplotlib_imshow(images[idx], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"))
        return fig


    def matplotlib_imshow(self, img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def load_split_train_test(self, valid_size = .2):
        train_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        ])
        test_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        ])
        print("ruta actual:", os.getcwd())
        train_data = datasets.ImageFolder("./fruits/Training", transform=train_transforms)
        test_data = datasets.ImageFolder("./fruits/Test",transform=test_transforms)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        from torch.utils.data.sampler import SubsetRandomSampler
        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        trainloader = torch.utils.data.DataLoader(train_data,
                    sampler=train_sampler, batch_size=64)
        testloader = torch.utils.data.DataLoader(test_data,
                    sampler=test_sampler, batch_size=64)
        return trainloader, testloader