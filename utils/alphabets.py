#coding=utf-8
import numpy as np
import torch

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

    def encode2(self,nums):
        length = list(nums)
        data = []
        for i, num in enumerate(nums):
            data.append(self.num2str[int(num)])
        return ''.join(data).replace(' ','')

class Alphabets_Chinese(object):
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

    def len(self):
        return len(self.alphabets_str)

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        for item in text:
            item = item.decode('utf-8', 'strict')

            length.append(len(item))
            for char in item:

                index = self.dict[char]
                result.append(index)

        text = result
        # print(text,length)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def encode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

# if __name__ == "__main__":
#     alphabets = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     aa = Alphabets(alphabets)
#     decode_data = aa.decode('0123')
#     print(decode_data)
#     encode_data = aa.encode(decode_data)
#     print(encode_data)

if __name__ == "__main__":
    alphabets = """ 某乃菽赅鲍堌窟千嗡持补嚅厍珪郈贱谅邻嬗絷塩戊釜玊刨敬匀塾茞尾宜梗皤气穹Ａ鹧遁景凯臾觊廛"""
    aa = Alphabets_Chinese(alphabets)
    decode_data = aa.decode('千气乃')
    print(decode_data)
    encode_data = aa.encode(decode_data)
    print(encode_data)

    # bb = strLabelConverter(alphabets)
    # decode_data = bb.decode('千气乃')
    # print(decode_data)
    # encode_data = bb.encode(decode_data)
    # print(encode_data)