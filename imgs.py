import seq.io
import numpy as np

def img_action(in_path,out_path):
    def action_helper(img_seq):
        img_action=np.array(img_seq).T
        print(img_action.shape)
        return [img_action]
    seq.io.transform_actions(in_path,out_path,action_helper,
    	img_in=False,img_out=True,whole_seq=True)

img_action('all','imgs')