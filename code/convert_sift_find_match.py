import cv2 as cv
import h5py
import numpy as np
import sys
import os
import pandas as pd
import multiprocessing


def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    
    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt+kp2[pair[0].trainIdx].pt))
    
    matches=np.array(matches)
    return matches

def h5_2_matching(h5file, file_dir, mask):
    file = h5py.File(os.path.join(file_dir, h5file))
    frames = file['frames'][()]
    for frame_num in range(frames.shape[0]-1):
        sift = cv.SIFT_create()
        frame1 = frames[frame_num]
        frame2 = frames[frame_num+1]
        kp, des = sift.detectAndCompute(frames[frame_num], mask=mask)
        sift = cv.SIFT_create()
        kp2, des2 = sift.detectAndCompute(frames[frame_num+1], mask=mask)
        matches = matcher(kp, des, frame1, kp2, des2, frame2, 0.5)
        if frame_num == 0:
            start = matches
        else:
            extra = np.zeros((start.shape[0],2))
            start = np.concatenate((start, extra), axis = 1)
            for i in range(matches.shape[0]):
                is_previous_point = False
                for j in range(start.shape[0]):
                    if (matches[i, :2] == start[j, -4:-2]).all():
                        start[j, -2:] = matches[i, -2:]
                        is_previous_point=True
                if not is_previous_point:
                    new_point = np.zeros((1, start.shape[1]))
                    new_point[0,-4:] = matches[i, :]
                    start = np.concatenate((start, new_point))
                
    # start = get_matching_matrix(frames, mask)

    df = pd.DataFrame(start)
    if not os.path.isdir("pointxy_results"):
        os.mkdir("pointxy_results")
    df.to_csv('pointxy_results/{0}_{1}_pointxy.csv'.format(file_dir.split('/')[-1], h5file.split('.h5')[0]), header=False, index=False)
    return 0


# def get_matching_matrix(frames, mask):
#     for frame_num in range(frames.shape[0]-1):
#         sift = cv.SIFT_create()
#         frame1 = frames[frame_num]
#         frame2 = frames[frame_num+1]
#         kp, des = sift.detectAndCompute(frames[frame_num], mask=mask)
#         sift = cv.SIFT_create()
#         kp2, des2 = sift.detectAndCompute(frames[frame_num+1], mask=mask)
#         matches = matcher(kp, des, frame1, kp2, des2, frame2, 0.5)
#         if frame_num == 0:
#             start = matches
#         else:
#             extra = np.zeros((start.shape[0],2))
#             start = np.concatenate((start, extra), axis = 1)
#             for i in range(matches.shape[0]):
#                 is_previous_point = False
#                 for j in range(matches.shape[0]):
#                     if (matches[i, :2] == start[j, -4:-2]).all():
#                         start[j, -2:] = matches[i, -2:]
#                         is_previous_point=True
#                 if not is_previous_point:
#                     new_point = np.zeros((1, start.shape[1]))
#                     new_point[0,-4:] = matches[i, :]
#                     start = np.concatenate((start, new_point))
#             print(start.shape)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_dir = sys.argv[1]
        mask = sys.argv[2]
    else:
        print("No parameter provided")


    all_h5_file_dirs = os.listdir(file_dir)
    all_file_dirs = []
    for parent_file_dir in all_h5_file_dirs:
        all_file_dirs.append(os.path.join(file_dir, parent_file_dir))

    mask = np.load(mask)
    mask = mask.astype(np.uint8)

    pool = multiprocessing.Pool(50)
    jobs = []

    for file_dirs in all_file_dirs:
        h5files = os.listdir(file_dirs)
        # for count, h5file in enumerate(h5files):
        #     h5files[count] = os.path.join(file_dirs, h5files)
            
        for h5file in h5files:
            jobs.append(pool.apply_async(h5_2_matching, args=(h5file, file_dirs, mask)))

        for job in jobs:
            job.get()
        # while len(jobs) > 0:
        #     jobs = [job for job in jobs if job.is_alive()]
        #     time.sleep(1)
        
        




