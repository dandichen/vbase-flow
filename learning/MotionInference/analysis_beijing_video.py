import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal





def main():
    data_path = '/mnt/scratch/haoyiliang/tozhangke/dense/tozhangke_20160607/20160526_wny_both'
    for _, _, files_name in os.walk(data_path):
        files_name.sort()
        for file_name in files_name[3:]:
            print file_name
            file_path = os.path.join(data_path, file_name)
            data = np.load(file_path)
            direction = signal.medfilt(data['direction'], 21)
            is_stop = data['is_stop']
            vel = signal.medfilt(data['vel'], 21)

            plt.figure()
            plt.plot(vel)
            plt.title('velocity')
            plt.waitforbuttonpress()

            plt.figure()
            direction[np.where(is_stop)] = 0
            direction[np.where(vel < 0.001)] = 0
            plt.plot(direction)
            plt.title('direction')
            plt.waitforbuttonpress()


            plt.close('all')


if __name__ == '__main__':
    main()
