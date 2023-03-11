import os
import numpy as np

def grab_data(file_name, num_of_arrys):
    if os.path.isfile(file_name):
            saved_data = np.load(file_name)
            data = dict(zip([str(i) for i in range(num_of_arrys)], (saved_data[k] for k in saved_data)))
            out_put = []
            
            for key in data:
                out_put.append(data[key])

            return out_put
    
    else:
        return[[]]*num_of_arrys