import torch
import torch.nn.functional as F
import torch.nn as nn
from models.densenet import *

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class Densenet_CRNN(nn.Module):
    def __init__(self, num_class, input_channel=3, output_channel = 1024):
        super(Densenet_CRNN,self).__init__()
        self.densnet = DenseNet()
        # features = list(self.densnet.children())
        # self.base_features = nn.Sequential(*features)
        self.base_features = self.densnet.features
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        self.output_channel = output_channel
        self.hidden_size = 256
        self.sequence_modeling = nn.Sequential(
                BidirectionalLSTM(self.output_channel, self.hidden_size, self.hidden_size),
                BidirectionalLSTM(self.hidden_size, self.hidden_size, self.hidden_size))
        self.num_class = num_class
        self.Prediction = nn.Linear(self.hidden_size, self.num_class)

    def forward(self,x):
        visual_feature = self.base_features(x)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        contextual_feature = self.sequence_modeling(visual_feature)
        prediction = self.Prediction(contextual_feature.contiguous())
        prediction = prediction.permute(1, 0, 2)
        return prediction

if __name__ == "__main__":
    net = Densenet_CRNN(10)
    x = torch.zeros(8, 3, 32, 320)
    output = net(x)
    print(output.size())