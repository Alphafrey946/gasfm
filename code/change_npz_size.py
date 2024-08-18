import os
import argparse
import numpy as np
import pandas as pd
import sys

parser = argparse.ArgumentParser(description="resize the given npz file for gasfm")

# Add arguments
parser.add_argument("file_location", type=str, help="npz_file_location")
parser.add_argument("start_frame", type=int, default=0,  help="the starting frame of the subset")
parser.add_argument("output_dir", type=str, help="the location of output directory")
parser.add_argument("-n", "--frame_num", type=int, default=1, help="overall frame numbers")





if __name__ == '__main__':
    args = parser.parse_args()


    test_npz = np.load(args.file_location, allow_pickle=True)
    d = dict(zip(("data1{}".format(k) for k in test_npz), (test_npz[k] for k in test_npz)))
    total_frame_num = d['data1data1M'].shape[0]/2
    start_frame = args.start_frame
    frame_num = args.frame_num


    if start_frame > total_frame_num-1:
        print("Error, start frame number is bigger than total frame")
        exit(1)


    subset1M = d['data1data1M'][start_frame*2:(start_frame+frame_num)*2,:]

    rows_to_delete = []
    for i in range(subset1M.shape[1]):
        if (subset1M[:,i] == np.zeros((1,frame_num*2))).all():
            rows_to_delete.append(i)

    new_subset1M = np.delete(subset1M, rows_to_delete, axis=1)
    new_subset1P = d['data1data1Ps_gt'][start_frame:start_frame+frame_num,:,:]

    np.savez_compressed(os.path.join(args.output_dir, "resized_"+args.file_location.split("/")[-1]), data1M=new_subset1M, data1Ns=None, data1Ps_gt=new_subset1P)