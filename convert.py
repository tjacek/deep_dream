import os.path
import dtw,seq

def convert_seq(in_path,out_path):
    data=['3DHOI','MSR','MHAD']
    feats=['corl','max_z','skew','std']
    for data_i in data:
        for feat_j in feats:
            in_ij=f'{in_path}/{data_i}/{feat_j}/seqs'
            if(os.path.exists(in_ij)):
                out_ij=f'{out_path}/{data_i}/{feat_j}/seqs'
                s_ij=seq.read_seq(in_ij)
                s_ij.save(out_ij)
                print(in_ij)
                print(out_ij)

in_path="../2023_X/DTW"
out_path="data"
convert_seq(in_path,out_path)

#s=seq.read_seq("s_test.npz")
#s.save("s_test")

#pairs=dtw.read_pairs("test.npz")
#print(type(pairs))
#pairs.save_eff("test")