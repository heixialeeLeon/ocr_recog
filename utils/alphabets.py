import numpy as np

class Alphabets(object):
    def __init__(self, alphabets_str):
        self.alphabets_str = alphabets_str
        self.num2str = list(self.alphabets_str)
        self.str2num = dict(zip(self.num2str, range(len(self.alphabets_str))))

    def decode(self,strs):
        strs = list(strs)
        length = len(strs)
        data = np.zeros([length, ])
        for i, s in enumerate(strs):
            data[i] = self.str2num[s]
        return data

    def encode(self,nums):
        length = list(nums)
        data = []
        for i, num in enumerate(nums):
            data.append(self.num2str[int(num)])
        return ''.join(data)

if __name__ == "__main__":
    alphabets = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    aa = Alphabets(alphabets)
    decode_data = aa.decode('0123')
    print(decode_data)
    encode_data = aa.encode(decode_data)
    print(encode_data)
