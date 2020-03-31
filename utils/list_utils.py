import random

def split_with_shuffle(target_list, rate):
    '''
    Split the target_list according to the rate
    @param target_list:
    @param rate:
    @return: list1, list2
    '''
    length=len(target_list)
    split_point = int(length*rate)
    random.shuffle(target_list)
    return target_list[0:split_point],target_list[split_point:]

if __name__ == "__main__":
    target_list = [1,2,3,4,5,6,7,8,9,10]
    l1,l2 = split_with_shuffle(target_list,0.8)
    print(l1)
    print(l2)