print("Importing the libraries")
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from FruitsNeuralNetwork import FruitsNeuralNetwork

from DbConnection import DbConnection
from FruitsDataset import FruitsDataset
import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model-name', default="model.pth")

    opt = parser.parse_args()
    print(opt)
    return opt


def train(params):
    print("Definir conexión a la BD")
    connection= DbConnection()
    print("Realizar Consulta de imágenes")
    train_dataframe= connection.get_fruits(is_test=False)
    test_dataframe= connection.get_fruits(is_test=True)
    labels_dataframe= connection.get_labels()
    labels_list= labels_dataframe.iloc[:, 0]
    print("Definir datasets")
    train_dataset = FruitsDataset(train_dataframe)
    test_dataset = FruitsDataset(test_dataframe)
    print("Definir Data loaders")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)
    print("Obtener e imprimir la forma de los datos")
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print("images.shape")
    print(images.shape)
    print("labels.shape")
    print(labels.shape)
    print(labels)

    print("Definir modelo")
    model = FruitsNeuralNetwork()
    print("Definir optimizador")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print("Definir función de pérdida")
    criterion = nn.CrossEntropyLoss()

    print("Verificar si se puede entrenar con CUDA")
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    print("Imprimir modelo")
    print(model)
    print("Entrenar según el número de épocas")
    for i in range(params.epochs):
        running_loss = 0
        for images, labels in train_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            print("Training pass")
            optimizer.zero_grad()

            output = model(images)
            print("output",output)
            print("output.shape",output.shape)
            print("forma de labels",labels.shape)
            print(labels)
            normalized_labels = []
            for Lbl in labels:
                normalized_labels.append([int(1) if x == Lbl else int(0) for x in labels_list])
            normalized_labels = torch.tensor(np.asarray(normalized_labels), dtype=torch.long)

            #normalized_labels = torch.tensor(np.asarray([[x] for x in labels]))
            loss = criterion(output, normalized_labels)

            print("Is where the model learns by backpropagating")
            loss.backward()

            print("Optimizes its weights here")
            optimizer.step()

            running_loss += loss.item()

            print("Guardar modelo")
            save(model, params.model_name)
        else:
            print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(train_loader)))
    print("Realizar prueba de modelo")
    test(test_loader=test_loader, model=model)


def save(model, path):
    torch.save(model, path)

def test(test_loader, model):
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

if __name__ == "__main__":
    params = get_params()
    train(params)