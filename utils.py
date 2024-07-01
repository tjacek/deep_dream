import os

def top_files(path):
    if(type(path)==str):
        paths=[ f'{path}/{file_i}'#path+'/'+file_i 
            for file_i in os.listdir(path)]
    else:
        paths=path  
    paths=sorted(paths)
    return paths

def iter_paths(dir_path):
    for path_i in top_files(dir_path):
        name_i=path_i.split('/')[-1]
        yield name_i,path_i

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def split(names):
    train,test=[],[]
    for name_i in names:
        person_i=get_person(name_i)
        if((person_i % 2)==1):
            train.append(name_i)
        else:
            test.append(name_i)
    return train,test

def get_person(name_i):
    return int(name_i.split('_')[1])

def get_cat(name_i):
    return int(name_i.split('_')[0])

def read_labels(in_path:str):
    with open(in_path, "r+") as f:
        text=f.read()
        return text.split('\n')