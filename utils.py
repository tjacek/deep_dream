import os

def top_files(path):
    if(type(path)==str):
        paths=[ f'{path}/{file_i}'#path+'/'+file_i 
            for file_i in os.listdir(path)]
    else:
        paths=path
    paths=sorted(paths)
    return paths