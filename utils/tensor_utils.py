import torch
from utils.alphabets import Alphabets
from torch.autograd import Variable

class TensorProcess(object):
    def __init__(self, alphabets):
        self.alphabets = alphabets

    def post_process(self, output_tensor):
        output = output_tensor.squeeze(1)
        prob, idx = output.topk(1)
        idx = idx.view(-1).detach().cpu().numpy()
        words = list(self.alphabets.encode(idx))
        i = 1
        while i<len(words):
            if words[i] == words[i-1]:
                del words[i]
            else:
                i+=1
        result=''.join(words).replace(' ','')
        return result

    def post_process_batch(self,output_tensor):
        labels = list()
        _,batch,_ = output_tensor.shape
        for index in range(batch):
            tensor = output_tensor[:,index,:]
            result = self.post_process(tensor)
            labels.append(result)
        return labels

    def post_process_ch(self, output_tensor):
        output = output_tensor.squeeze(1)
        prob, idx = output.topk(1)
        idx = idx.view(-1).detach().cpu().numpy()
        words = list(self.alphabets.encode(idx))
        i = 1
        while i<len(words):
            if words[i] == words[i-1]:
                del words[i]
            else:
                i+=1
        result=''.join(words).replace('~','')
        return result

    def post_process_batch_ch(self,output_tensor):
        labels = list()
        _,batch,_ = output_tensor.shape
        for index in range(batch):
            tensor = output_tensor[:,index,:]
            result = self.post_process_ch(tensor)
            labels.append(result)
        return labels

def compare_str_list(label_a, label_b):
    correct_count = 0
    assert(len(label_a) == len(label_b))
    for index in range(len(label_a)):
        # print("gt: {}".format(label_b[index]))
        # print("predict: {}".format(label_a[index]))
        if label_a[index] == label_b[index]:
            correct_count +=1
    return correct_count

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res