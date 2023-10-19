import cv2
import utils

class ActionDict(dict):
    def __init__(self, arg=[]):
        super(ActionDict, self).__init__(arg)

def read_action(in_path:str):
    action_dict=ActionDict()
    for path_i in utils.top_files(in_path):
        name_i=path_i.split('/')[-1]
        action_i=[cv2.imread(path_j,0)
                for path_j in utils.top_files(path_i)]
        action_dict[name_i]=action_i
    return action_dict


in_path='../MSR/frames'
action_dict= read_action(in_path)
print(action_dict.keys())