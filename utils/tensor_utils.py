from utils.alphabets import Alphabets

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

def compare_str_list(label_a, label_b):
    correct_count = 0
    assert(len(label_a) == len(label_b))
    for index in range(len(label_a)):
        if label_a[index] == label_b[index]:
            correct_count +=1
    return correct_count