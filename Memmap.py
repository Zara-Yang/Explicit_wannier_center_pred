import numpy as np

class Memmap():
    @staticmethod
    def Memmap_save(save_path,save_name,matrix):
        shape = matrix.shape
        path = "{}/{}_{}.mapdat".format(save_path,save_name,"-".join([str(i) for i in shape]))
        fp = np.memmap(path, dtype='float32', mode='w+', shape=shape)
        fp[:] = matrix[:]
        fp.flush()
    @staticmethod
    def Memmap_read(save_path):
        shape = save_path.split(".")[0].split("_")[-1].split("-")
        shape = tuple([int(i) for i in shape])
        matrix = np.memmap(save_path, dtype='float32', mode='r', shape=shape)
        return(matrix)
