import dtw

in_path="pairs"

pairs=dtw.read_pairs(in_path)
print(type(pairs))
pairs.save_eff("test")