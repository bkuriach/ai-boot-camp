import config
import dataset
import torch
import TextClassification.DeepLearning.src.engine as engine
import TextClassification.DeepLearning.src.model as model
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


def data_split(features, encoded_labels):
    split_idx = int(len(features) * config.SPLIT)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    ## print out the shapes of your resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    return train_x, train_y, val_x, val_y, test_x, test_y

def run():

    sent_data = dataset.SentimentDataset()
    sent_data.load_data()
    sent_data.clean_data()
    sent_data.vocab_dict()
    sent_data.encode_text()
    sent_data.encode_label()
    sent_data.remove_outliers()
    sent_data.pad_features(config.SEQ_LENGTH)

    features= sent_data.features
    encoded_labels = sent_data.encoded_labels

    train_x, train_y, val_x, val_y, test_x, test_y = data_split(features, encoded_labels)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config.BATCH_SIZE)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=config.BATCH_SIZE)

    vocab_size = len(sent_data.vocab_to_int)+1
    output_size = config.OUTPUT_SIZE

    if config.MODEL_ARCH =='LSTM':
        net = model.SentimentLSTM(vocab_size, output_size, config.EMBEDDING_DIM, config.HIDDEN_DIM, config.N_LAYERS)
    elif config.MODEL_ARCH =='CNN':
        net = model.SentimentCNN(vocab_size,config.EMBEDDING_DIM,output_size)
    elif config.MODEL_ARCH =='LSTM+CNN':
        net = model.SentimentCNNLSTM(vocab_size, config.EMBEDDING_DIM, output_size)
    elif config.MODEL_ARCH =='BiLSTM':
        net = model.SentimentBiLSTM(vocab_size, output_size, config.EMBEDDING_DIM, config.HIDDEN_DIM, config.N_LAYERS)



    print(net)
    lr=0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.to(device=config.DEVICE)
    net.train()

    net = engine.train_fn(train_loader, valid_loader, net, optimizer, criterion, config.DEVICE)
    engine.test_fn(test_loader, net, criterion, config.DEVICE)

    print(" Testing few insances ")
    engine.predict(net, sent_data ," I Love this movie")
    engine.predict(net, sent_data, " This movie is not good")
    engine.predict(net, sent_data, "The worst movie I have seen; acting was terrible and I want my money back")
    engine.predict(net, sent_data, " I enjoyed this movie")
    engine.predict(net, sent_data, " this movie is pathetic")

if __name__ =="__main__":

    print(f" {config.MODEL_ARCH} Model Architecture")
    run()













