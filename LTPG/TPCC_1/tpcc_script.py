import os
import time


def execute(batch):
    DEVICE = 2
    EPOCH = 5000
    BATCH = batch
    WAREHOUSE = 0
    START = 0
    END = 11
    conditions = []
    with open('Predefine.h', 'r', encoding='utf-8') as f:
        conditions = f.readlines()
        f.close()
    for i in range(0, len(conditions)):
        conditions[i] = conditions[i].replace('\n', '')
    args_list = []
    for i in range(5, 9):
        tmp_0 = conditions[i].split('=')[0]
        args_list.append(tmp_0)
    # print(args_list)
    # exit()
    for i in range(START, END):
        WAREHOUSE = pow(2, i)
        conditions[5] = args_list[0] + '=(unsigned int)' + str(DEVICE) + ';'
        conditions[6] = args_list[1] + '=(unsigned int)' + str(EPOCH) + ';'
        conditions[7] = args_list[2] + '=(unsigned int)' + str(BATCH) + ';'
        conditions[8] = args_list[3] + \
            '=(unsigned long long)' + str(WAREHOUSE) + ';'
        with open('Predefine.h', 'w', encoding='utf-8', newline='') as f:
            for element in conditions:
                f.write(element + '\n')
            f.flush()
            f.close()
        print('Execute info')
        print(' DEVICE: ' + str(DEVICE))
        print(' EPOCH: ' + str(EPOCH))
        print(' BATCH: ' + str(BATCH))
        print(' WAREHOUSE: ' + str(WAREHOUSE))
        current_time = time.strftime('%Y-%m-%d-%H-%M-%S',
                                     time.localtime(time.time()))
        filename = str(WAREHOUSE) + '.txt'
        if os.path.exists("./log_" + str(BATCH) + "/") is False:
            os.mkdir("./log_" + str(BATCH) + "/")
        if os.path.exists("./log_" + str(BATCH) + "/" + filename) is False:
            print(current_time)
            info = os.popen('bash compile.sh').readlines()
            current_time = time.strftime('%Y-%m-%d-%H-%M-%S',
                                         time.localtime(time.time()))
            for element in info:
                if element != '\n':
                    print(element)
            print(current_time)
            with open("./log_" + str(BATCH) + "/" + filename,
                      'w',
                      encoding='utf-8') as f:
                f.writelines(info)
                f.flush()
                f.close()


if __name__ == '__main__':
    # 7, 21
    for i in range(7, 21):
        execute(pow(2, i))
