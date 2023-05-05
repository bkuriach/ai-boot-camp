import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import config
import numpy as np

def train_fn(data_loader, valid_loader, model, optimizer, criterion, device, model_name="lstm"):
    counter = 0
    for e in range(config.EPOCHS):
        for inputs, labels in data_loader:
            counter += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            inputs = inputs.long()
            if config.MODEL_ARCH =='CNN':
                output = model(inputs)
            else:
                output, h = model(inputs)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
            optimizer.step()

            # loss stats
            if counter % config.PRINT_EVERY == 0:
                # Get validation loss
                val_losses = []
                model.eval()
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                    # output, val_h = model(inputs)
                    if config.MODEL_ARCH == 'CNN':
                        output = model(inputs)
                    else:
                        output, val_h = model(inputs)
                    # output = model(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(e+1, config.EPOCHS),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
    return model

def test_fn(test_loader, model, criterion, device):
    # Get test data loss and accuracy
    test_losses = []  # track loss
    num_correct = 0
    model.eval()
    # iterate over test data
    for inputs, labels in test_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        # get predicted outputs

        # output, h = model(inputs)
        # output = model(inputs)
        if config.MODEL_ARCH == 'CNN':
            output = model(inputs)
        else:
            output, h = model(inputs)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy()) #if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

def predict(model, data_object, input):

    padded_input = data_object.process_new_instance(str(input))
    padded_input = torch.from_numpy(padded_input)
    padded_input = padded_input.to(config.DEVICE)

    if config.MODEL_ARCH == 'CNN':
        output = model(padded_input)
    else:
        output, h = model(padded_input)
    # output, h = model(padded_input)
    # output= model(padded_input)
    confidence = output.squeeze()
    pred = torch.round(confidence)
    result = None
    if pred == 1:
        result = 'Positive'
    else:
        result = 'Negative'
        confidence = 1-confidence

    print(f' Review - {input} ::{result} --> confidence :{confidence}')