import os

from utility.utils import *

if __name__=='__main__':
    if not os.path.exists(os.path.join('./data/', 'CSV')):
        os.mkdir(os.path.join('./data', 'CSV'))
        os.mkdir(os.path.join('./data', 'CSV', 'Benign assembly'))
        os.mkdir(os.path.join('./data', 'CSV', 'Virus assembly'))
        os.mkdir(os.path.join('./data', 'CSV', 'Virus assembly', 'Winwebsec'))

    benign_assembly = os.listdir('./data/Benign assembly')
    for file in benign_assembly:
        if 'exe.asm' in file:
            df_path = './data/CSV/Benign assembly/'+ file.split('.exe.asm')[0] + '.csv'
            exe_path = os.path.join('./data/Benign assembly', file)
            generate_csv_for_each_sample(exe_path, df_path)

    winwebsec = os.listdir('./data/Virus assembly/Winwebsec')
    for file in benign_assembly:
        if 'exe.asm' in file:
            df_path = './data/CSV/Virus assembly/Winwebsec/' + file.split('.exe.asm')[0] + '.csv'
            exe_path = os.path.join('./data/Virus assembly/Winwebsec', file)
            generate_csv_for_each_sample(exe_path, df_path)


    