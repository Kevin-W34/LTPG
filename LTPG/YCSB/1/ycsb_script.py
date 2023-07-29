import os
import time


def execute(batch):
    DEVICE = 0
    EPOCH = 5000
    BATCH = batch
    YCSB_SIZE = 0
    YCSB_SIZE_START = 4
    YCSB_SIZE_END = 8
    YCSB_OP_SIZE = 10
    # YCSB_READ_SIZE = 10
    DATA_DISTRIBUTION = 'ZIPFIAN'

    conditions = []
    with open('Predefine.h', 'r', encoding='utf-8') as f:
        conditions = f.readlines()
        f.close()
    for i in range(0, len(conditions)):
        conditions[i] = conditions[i].replace('\n', '')
    args_list = []
    for i in range(5, 12):
        tmp_0 = conditions[i].split('=')[0]
        args_list.append(tmp_0)
    # print(args_list)
    # exit()
    for YCSB_READ_SIZE in range(0, 11):
        for i in range(YCSB_SIZE_START, YCSB_SIZE_END):
            YCSB_SIZE = pow(10, i)
            conditions[5] = args_list[0] + \
                '=(unsigned int)' + str(DEVICE) + ';'
            conditions[6] = args_list[1] + '=(unsigned int)' + str(EPOCH) + ';'
            conditions[7] = args_list[2] + '=(unsigned int)' + str(BATCH) + ';'
            conditions[8] = args_list[3] + \
                '=(unsigned long long)' + str(YCSB_SIZE) + ';'
            conditions[9] = args_list[4] + \
                '=(unsigned int)' + str(YCSB_OP_SIZE)+';'
            conditions[10] = args_list[5] + \
                '=(unsigned int)'+str(YCSB_READ_SIZE)+';'
            conditions[11] = args_list[6] + \
                '=(unsigned int)'+str(DATA_DISTRIBUTION)+';'
            with open('Predefine.h', 'w', encoding='utf-8', newline='') as f:
                for element in conditions:
                    f.write(element + '\n')
                f.flush()
                f.close()
            print('Execute info')
            print(' DEVICE: ' + str(DEVICE))
            print(' EPOCH: ' + str(EPOCH))
            print(' BATCH: ' + str(BATCH))
            print(' YCSB_SIZE: ' + str(YCSB_SIZE))
            print(' YCSB_OP_SIZE: ' + str(YCSB_OP_SIZE))
            print(' YCSB_READ_SIZE: ' + str(YCSB_READ_SIZE))
            print(' DATA_DISTRIBUTION: ' + str(DATA_DISTRIBUTION))
            current_time = time.strftime('%Y-%m-%d-%H-%M-%S',
                                         time.localtime(time.time()))
            filename = str(YCSB_SIZE) + '_HO_' + str(YCSB_READ_SIZE) + '.txt'
            path = './log_' + str(BATCH) + '_'+str(YCSB_OP_SIZE) + \
                '_'+str(DATA_DISTRIBUTION) + '/'
            # print(filename)
            # print(path)
            # exit()
            if os.path.exists(path) is False:
                os.mkdir(path)
            if os.path.exists(path + filename) is False:
                print(current_time)
                info = os.popen('bash compile.sh').readlines()
                current_time = time.strftime('%Y-%m-%d-%H-%M-%S',
                                             time.localtime(time.time()))
                for element in info:
                    if element != '\n':
                        print(element)
                print(current_time)
                with open(path + filename,
                          'w',
                          encoding='utf-8') as f:
                    f.writelines(info)
                    f.flush()
                    f.close()


if __name__ == '__main__':
    # 4, 8
    for i in range(3, 7):
        execute(pow(10, i))
