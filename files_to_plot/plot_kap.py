import matplotlib.pyplot as plt
import csv
import pandas as pd


def main():
    file_name = 'lstm_lstm_one_direction_multi_14.csv'
    df = pd.read_csv(file_name, sep=',', header=None, encoding="ISO-8859-1")
    kap_test = []
    loss = []
    for i in range(600):
        if i < 50: kap_test.append(df.loc[3, i])
        loss.append(df.loc[0, i])

    file_name = 'lstm_noMP_lstm_one_direction_multi_19.csv'
    df = pd.read_csv(file_name, sep=',', header=None, encoding="ISO-8859-1")
    kap_test2 = []
    loss2 = []
    for i in range(600):
        if i < 50: kap_test2.append(df.loc[3, i])
        loss2.append(df.loc[0, i])

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(loss2, label='noMoT')
    ax.plot(loss, label='withMoT')
    plt.xlabel('# batch')
    plt.ylabel('loss')
    ax.legend(loc='upper right', shadow=False)
    fig.savefig(file_name[:9]+"loss.pdf", bbox_inches='tight')

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(kap_test2, label='noMoT')
    ax.plot(kap_test, label='withMoT')
    ax.legend(loc='lower right', shadow=False)
    plt.xlabel('# epoch')
    plt.ylabel('kappa')
    fig.savefig(file_name[:9]+"kappa.pdf", bbox_inches='tight')















    plt.show()

    dbstop = 0





if __name__ == '__main__':
    main()