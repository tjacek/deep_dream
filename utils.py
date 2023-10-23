import os

def top_files(path):
    if(type(path)==str):
        paths=[ f'{path}/{file_i}'#path+'/'+file_i 
            for file_i in os.listdir(path)]
    else:
        paths=path  
    paths=sorted(paths)
    return paths

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