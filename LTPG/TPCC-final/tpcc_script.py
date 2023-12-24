import os
import time


def execute(batch, NEWORDER_PERSENT):
    DEVICE = 7
    EPOCH = 1000
    BATCH = batch
    WAREHOUSE = 0
    START = 3
    END = 7
    # NEWORDER_PERSENT = 100
    if NEWORDER_PERSENT == 100:
        dicname = "./Neworder_log_" + str(BATCH) + "/"
    elif NEWORDER_PERSENT == 0:
        dicname = "./Payment_log_" + str(BATCH) + "/"
    else:
        dicname = "./Mixed_log_" + str(BATCH) + "/"
    conditions = []
    with open('Predefine.h', 'r', encoding='utf-8') as f:
        conditions = f.readlines()
        f.close()
    for i in range(0, len(conditions)):
        conditions[i] = conditions[i].replace('\n', '')
    args_list = []
    for i in range(5, 10):
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
        conditions[9] = args_list[4] + \
            '=(unsigned int)'+str(NEWORDER_PERSENT)+';'
        with open('Predefine.h', 'w', encoding='utf-8', newline='') as f:
            for element in conditions:
                f.write(element + '\n')
            f.flush()
            f.close()
        # print('Execute info')
        # print(' DEVICE: ' + str(DEVICE))
        # print(' EPOCH: ' + str(EPOCH))
        # print(' BATCH: ' + str(BATCH))
        # print(' WAREHOUSE: ' + str(WAREHOUSE))
        # current_time = time.strftime('%Y-%m-%d-%H-%M-%S',
        #                              time.localtime(time.time()))
        filename = str(WAREHOUSE) + '_2.txt'
        if os.path.exists(dicname) is False:
            os.mkdir(dicname)
        # if os.path.exists(dicname + filename) is False:            
        for i in range(0, 3):
            print('Execute info '+str(i))
            print(' DEVICE: ' + str(DEVICE))
            print(' EPOCH: ' + str(EPOCH))
            print(' BATCH: ' + str(BATCH))
            print(' WAREHOUSE: ' + str(WAREHOUSE))
            current_time = time.strftime('%Y-%m-%d-%H-%M-%S',
                                 time.localtime(time.time()))
            print(current_time)
            # info = ''
            info = os.popen('bash compile.sh').readlines()
            current_time = time.strftime('%Y-%m-%d-%H-%M-%S',
                                         time.localtime(time.time()))
            for element in info:
                if element != '\n':
                    print(element)
            print(current_time)
            with open(dicname + filename,
                      'a',
                      encoding='utf-8') as f:
                f.writelines(info)
                f.flush()
                f.close()


if __name__ == '__main__':
    # 7, 21
    for i in range(16, 17):
        # execute(pow(2, i), 100)
        # execute(pow(2, i), 50)
        execute(pow(2, i), 0)
    # for i in range(1,16):
    #     execute(6144+128*i, 100)
