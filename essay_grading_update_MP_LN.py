import tensorflow as tf
import numpy as np
import time, os
import matplotlib.pyplot as plt
import csv, math

import pandas as pd
from collections import Counter

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# note:
# *1. loss function: look up the data mining book
# 2. fully_connected regression point of view
# 3. loss function: regularization
# _4. unpadding
# 5. bidirectional
# *6. deep LSTM
# 7. validation to find the best parameter
# *8. LSTMCell LSTMBlockCell LSTMBLockFusedCell LSTMBlockWrapper?
# 9. find the frequency of different label, use 1/freq to reweigh the loss in order to augment less frequent label's learning


import sklearn.metrics

from helper import Helper
import utils


def main(embedding_size):
    tf.reset_default_graph()
    # reset the previously defined graph

    # Read in data
    # readfile = './data/small.tsv'
    readfile = './data/training_set_rel3.tsv'
    X_, _, y_ = Helper(set_num=0, file_name=readfile).get_embed(embedding_size)

    print('reading file: ', readfile)
    benchmark_score = 8 * np.ones_like(y_)
    print('accuracy: ', np.sum(np.where(benchmark_score != y_, 1, 0))/len(y_))


    data_size = y_.shape[0]
    train_size = math.floor(0.9 * data_size)
    test_size = data_size - train_size

    X_train = X_[:train_size]
    y_train = y_[:train_size]
    X_test = X_[train_size:]
    y_test = y_[train_size:]


    # X_train is embedded essays (num_essays, num_word, embedding_dim = 100)
    # y_train is the corresponding labels ([num_essays])

    classification = True
    withWeight = False
    print('classification: ', classification)
    print('withWeight:', withWeight)

    if withWeight:
        _, inv_freq = utils.hist_freq(y_train, 12)
    else:
        inv_freq = np.ones(12)

    # experiment: decreasing deep_net hidden neurons should be very possibly
    # equivanlent to taking the averaged vector rep.; the value of these limited
    # hidden units reflects an averaged effect of the sequence.
    # The tf-idf analogy on the other hand is embedding size = 100 being close to 1.
    # There is a w1 between input layer and the hidden lstm unit layer.

    N_CLASSES = 12 if classification else 1

    deep_nets = [[32, 32, 32], [128], [128, 64], [128, 64, 32], [128, 64, 128, 64, 32], [32]*20]
    lstm_sizes = deep_nets[0]

    multilayer = ""
    if len(lstm_sizes) > 1:
        multilayer = "multi"

    cells = ['lstm', 'lstm_block', 'lstm_block_fused', 'gru']
    controllers = ['one_direction', 'bidirection']
    cell_type = cells[0]
    controller_type = controllers[0]

    # Define paramaters for the model
    LEARNING_RATE = 0.01
    BATCH_SIZE = 128
    DROPOUT = 0.8 # the keep_prob
    N_EPOCHS = 50
    l2_reg_lambda = 0.0
    # var_reg_lambda = 0.0
    LAYERNORM = False

    print('LEARNING_RATE:', LEARNING_RATE, 'DROPOUT:', DROPOUT,
          'l2_lambda:', l2_reg_lambda, 'embedding_size:', embedding_size,
          'layernorm', LAYERNORM)

    DISP_STEP = 1



    bidi = tf.constant(1, dtype=tf.int32)
    if controller_type == "bidirection":
        bidi = tf.constant(2, dtype=tf.int32)

    np.random.seed(0)



    print(cell_type)
    print(controller_type)
    print(lstm_sizes)



    def batch_zero_pad(x_batch):
        """
        Zero Pad input messages
        :param x_batch: Input list of encoded messages (num_batch, num_word, embedding_dim)
        # :param seq_ken: Input int, maximum sequence input length
        :return: numpy array.  The padded essays. (num_batch, maxlength, embedding_dim)
        """

        maxlength = 0
        for essay in x_batch:
            if essay.shape[0] > maxlength:
                maxlength = essay.shape[0]

        embedding_size = x_batch[0].shape[1]
        x_batch_padded = np.zeros((len(x_batch), maxlength, embedding_size))
        for idx, essay in enumerate(x_batch):
            non_empty_length = essay.shape[0]
            x_batch_padded[idx, :non_empty_length] = essay

        return x_batch_padded, maxlength
    # test: X_batch = batch_zero_pad(X_train[:100])


    Y_train = utils.one_hot(y_train, N_CLASSES)


    #In[5]

    with tf.name_scope('data'):
        # num_word = tf.Variable(tf.constant(800, tf.int32), name='num_word')
        seqlen = tf.placeholder(tf.int32, [BATCH_SIZE], name='sequence_len')
        X = tf.placeholder(tf.float32, [BATCH_SIZE, None, embedding_size], name="X_placeholder")
        Y = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES], name="Y_placeholder")
        Y_onehot_rev = tf.placeholder(tf.float32, [BATCH_SIZE, None], name="Y_onehot_rev_placeholder")

        # state = tf.placeholder(tf.float32, shape=[None_preds, 2*N_HIDDEN])
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        l2_loss = tf.constant(0.0)
        loss_weight = tf.placeholder(tf.float32, [N_CLASSES])

        phase = tf.placeholder(tf.bool, name='phase')

    #In[6]
    with tf.name_scope('process_data'):
        w1 = tf.Variable(tf.truncated_normal(shape=[embedding_size, lstm_sizes[0]], stddev=1.0), name='w1')
        b1 = tf.Variable(tf.truncated_normal([lstm_sizes[0]], mean=0.0, stddev=1.0), name='b1')

        # _X = tf.add(tf.tensordot(X, w1, [[2], [0]]), b1)  # >> X.shape = BATCH_SIZE, num_steps, 100
        # _X = tf.contrib.layers.batch_norm(_X, center=True, scale=True, is_training=phase)

        _X = X

        l2_loss += tf.nn.l2_loss(w1)
        l2_loss += tf.nn.l2_loss(b1)

    def get_cell(hidden_size, cell_type):
        if cell_type == "lstm":
            return tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0, initializer=tf.orthogonal_initializer())
        # This is added to biases the forget gate
        # in order to reduce the scale of forgetting in the beginning of the training.

        elif cell_type == "lstm_block":
            return tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=1.0)

        # elif cell_type == "lstm_block_fused":
        #     return tf.contrib.rnn.LSTMBlockFusedCell(hidden_size)
        # not supported

        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)

        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

##
    def get_controller(cell, _X, seqlen, controller_type="one_direction"):
        batch_size = _X.shape[0]
        if controller_type == "one_direction":
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
            last_output = tf.gather_nd(outputs, tf.stack([tf.range(batch_size), tf.maximum(seqlen - 1, tf.zeros([batch_size], dtype=tf.int32))], axis=1))

            elem = (outputs, tf.range(batch_size))
            mp_output = tf.map_fn(lambda seq: tf.reduce_mean(
                tf.slice(seq[0], [0, 0], [seqlen[seq[1]], outputs.shape[2]]), axis=0),
                                  elem, dtype=tf.float32)
            assert last_output.shape == mp_output.shape
            return mp_output

        elif controller_type == "bidirection":
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
            last_outputs = [tf.gather_nd(outputs[i], tf.stack([tf.range(batch_size), tf.maximum(seqlen - 1, tf.zeros([batch_size], dtype=tf.int32))], axis=1)) for i in range(2)]

            elem = (outputs[0], tf.range(batch_size))
            mp_output0 = tf.map_fn(lambda seq: tf.reduce_mean(
                tf.slice(seq[0], [0, 0], [seqlen[seq[1]], outputs[0].shape[2]]), axis=0),
                                   elem, dtype=tf.float32)
            elem = (outputs[1], tf.range(batch_size))
            mp_output1 = tf.map_fn(lambda seq: tf.reduce_mean(
                tf.slice(seq[0], [0, 0], [seqlen[seq[1]], outputs[1].shape[2]]), axis=0),
                                   elem, dtype=tf.float32)
            assert last_outputs[0].shape == mp_output0.shape
            assert last_outputs[1].shape == mp_output1.shape
            return tf.add(mp_output0, mp_output1)

        else:
            print("ERROR: '" + controller_type + "' is a wrong controller type. Use default.")
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
            last_output = tf.gather_nd(outputs, tf.stack([tf.range(BATCH_SIZE), tf.maximum(seqlen - 1, tf.zeros([batch_size], dtype=tf.int32))], axis=1))
            # which is the Tensorflow equivalent of
            #  numpy's rnn_outputs[range(30), seqlen-1, :]

            outputs = tf.slice(outputs, [0, 0, 0], [batch_size, seqlen, outputs.shape[2]])
            mp_output = tf.reduce_mean(outputs, axis=1)

            return last_output, mp_output

##
    with tf.name_scope('lstm') as scope:
        # def model(_X, seqlen , lstm_sizes, dropout_keep_prob, cell_type="lstm", controller_type="one_direction"):

        # lstm_cells = [get_cell(lstm_size, cell_type=cell_type) for lstm_size in lstm_sizes ]
        # dropped_cells = [tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob) for lstm_cell in lstm_cells]
        # cell = tf.contrib.rnn.MultiRNNCell(dropped_cells)
        lstm_cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(
            lstm_size,
            forget_bias=1.0,
            layer_norm=LAYERNORM,
            norm_gain=1.0,
            norm_shift=0.0,
            dropout_keep_prob=dropout_keep_prob,
            # initializer=tf.contrib.layers.xavier_initializer()
        ) for lstm_size in lstm_sizes]
        cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
        last_output = get_controller(cell, _X, seqlen, controller_type=controller_type)

    with tf.variable_scope('Output_layer') as scope:
        w2 = tf.Variable(tf.truncated_normal(shape=[lstm_sizes[-1], N_CLASSES]), name='w2')
        # b2 = tf.Variable(tf.constant(0.1, shape=[N_CLASSES]), name='b2')
        b2 = tf.Variable(tf.truncated_normal([N_CLASSES], mean=0.0, stddev=1.0), name='b2')

        logits = tf.nn.xw_plus_b(last_output, w2, b2, name='logits')  # batch_size * n_class

        l2_loss += tf.nn.l2_loss(w2)
        l2_loss += tf.nn.l2_loss(b2)

    with tf.name_scope('loss') as scope:

        if classification:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name='loss') + l2_reg_lambda * l2_loss
        else:
            # prob_loss = tf.nn.softmax(logits) # (1, 1:12)
            # score_candi_loss = tf.ones((BATCH_SIZE, 1)) * tf.cast(tf.range(N_CLASSES) + 1, tf.float32) # (batch_size, 1:12)
            # a_term = score_candi_loss - tf.cast(Y_onehot_rev, dtype=tf.float32) *tf.ones((1, N_CLASSES))   # (batch_size, 1:12)
            # # aterm = tf.range(N_CLASSES) + 1 - Y_onehot_rev
            # a_term = tf.square(a_term) * loss_weight # loss_weight (1, 1:12)
            # loss = tf.reduce_sum(tf.multiply(prob_loss, a_term), name='lossFunction') + l2_reg_lambda * l2_loss
            loss = tf.reduce_sum(tf.square(logits - Y_onehot_rev), name='lossFunction') + l2_reg_lambda * l2_loss

    with tf.name_scope('optimizer') as scope:
        #: define training op
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)
        # optimizer = tf.keras.optimizers.RMSprop().minimize(loss, global_step=global_step)

    #In[23]
    with tf.name_scope('accuracy') as scope:
        query_size = tf.placeholder(dtype=tf.int32, shape=None)

        if classification:
            score_class = tf.ones((query_size, 1)) * tf.cast(tf.range(N_CLASSES) + 1, tf.float32)
            prob = tf.nn.softmax(logits[:query_size])
            pred_class = tf.round(tf.reduce_sum(tf.multiply(prob, score_class), 1))

            correct_preds = tf.equal(tf.to_int64(pred_class), tf.argmax(Y[:query_size], 1)+1)
            accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

            # score_candi = tf.ones((query_size, 1)) * tf.cast(tf.range(N_CLASSES) + 1, tf.float32)
            # prob = tf.nn.softmax(logits[:query_size])
            # pred_class = tf.reduce_sum(tf.multiply(prob, score_candi), 1)  ## attention: score_candi lazy loading?
            #
            # correct_preds = tf.math.exp(
            #     - 0.5 * tf.square((tf.round(pred_class) - tf.cast(tf.argmax(Y[:query_size], 1), tf.float32))) / 3)
            # accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        else:
            # score_candi = tf.ones((query_size, 1)) * np.arange(1, 13)
            # prob = tf.nn.softmax(logits[:query_size])
            # pred_class = tf.reduce_sum(tf.multiply(prob, score_candi), 1)
            pred_class = tf.round(logits[:query_size])

            correct_preds = tf.equal(tf.to_int64(pred_class), tf.argmax(Y[:query_size], 1) + 1)
            accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
            # correct_preds = tf.math.exp(- 0.5 * tf.square((tf.round(pred_class) - tf.cast(tf. argmax(Y[:query_size], 1), tf.float32) )) / 3)
            # accuracy = tf. reduce_mean(tf.cast(correct_preds, tf.float32))


    #In[28]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # to visualize using TensorBoard
        writer = tf.summary.FileWriter('./graphs/aes', sess.graph)
        writer.close()
        ##### You have to create folders to store checkpoints
        # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
        # if that checkpoint exists, restore from checkpoint
        # if ckpt and ckpt.model_checkpoint_path:
        #    saver.restore(sess, ckpt.model_checkpoint_path)

        # initial_step = global_step.eval()

        start_time = time.time()
        tic_time = time.time()

        total_loss = 0.0
        index_batch = 0
        total_correct_preds = 0

        kappa_scors = []
        kappas_test = []
        accuracies = []
        loss_fn = []
        times = []

        for i in range(N_EPOCHS):  # train the model n_epochs times
            machine_score = np.array([])
            human_score = np.array([])

            for X_batch, Y_batch, _seqlen, _ in utils.get_batches(X_train, Y_train, BATCH_SIZE):
                X_batch, num_word = batch_zero_pad(X_batch)
                # assert Y_batch.shape[0] == BATCH_SIZE:
                Y_batch = Y_batch.reshape((BATCH_SIZE, N_CLASSES))

                # print("Prior weight1 = ", w1.eval())
                # print("Prior weight2 = :", w2.eval())

                _, loss_batch, _accuracy, _pred_class = sess.run([optimizer, loss, accuracy, pred_class],
                            feed_dict={X: X_batch, Y: Y_batch, Y_onehot_rev: utils.one_hot_reverse(Y_batch)[:, None],
                                       dropout_keep_prob: DROPOUT, seqlen: _seqlen, query_size: BATCH_SIZE, loss_weight: inv_freq,
                                       phase: True})

                # print("Post weight1 = ", w1.eval())
                # print("Post weight2 = :", w2.eval())


                total_loss += loss_batch

                y_score = utils.one_hot_reverse(Y_batch)
                # print("Labels: ", y_score)

                machine_score = np.append(machine_score, _pred_class, axis=0)
                human_score = np.append(human_score, y_score, axis=0)

                # print('Predictions:', machine_score)
                print('Accuracy:', _accuracy)
                index_batch += 1

                # print("global_step == index_batch? ", global_step.eval() == index_batch)

                if (index_batch + 1) % DISP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index_batch + 1, total_loss / DISP_STEP))

                    loss_fn.append(total_loss / DISP_STEP)
                    accuracies.append(_accuracy)
                    times.append(time.time() - tic_time)
                    tic_time = time.time()

                    total_loss = 0.0
                    # saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index_batch)

            # print("machine score:", np.round(machine_score[:100]).astype(int))
            # print("human score:", human_score[:100].astype(int))

            score = sklearn.metrics.cohen_kappa_score(np.round(machine_score), np.round(human_score), weights='quadratic')
            print("Kappa score: ", score)
            print("epoch: {}".format(i))

            kappa_scors.append(score)

            ## Watch on test set
            Y_test = utils.one_hot(y_test, N_CLASSES)

            test_machine_score = np.array([])
            test_human_score = np.array([])
            count = 0

            for X_batch, Y_batch, _seqlen, effective_size in utils.get_batches(X_test, Y_test, BATCH_SIZE, needall='True'):
                X_batch, num_word = utils.batch_zero_pad(X_batch)
                Y_batch = Y_batch.reshape((BATCH_SIZE, N_CLASSES))

                _accuracy, _pred_class = sess.run(
                    [accuracy, pred_class],
                    feed_dict={X: X_batch, Y: Y_batch, Y_onehot_rev: utils.one_hot_reverse(Y_batch)[:, None],
                               dropout_keep_prob: 1.0, seqlen: _seqlen, query_size: effective_size, loss_weight: inv_freq,
                               phase: False})

                y_score = utils.one_hot_reverse(Y_batch[:effective_size])

                test_machine_score = np.append(test_machine_score , _pred_class)
                test_human_score = np.append(test_human_score, y_score)
                assert _pred_class.shape[0] == effective_size
                assert len(y_score) == effective_size

            kappa_test = sklearn.metrics.cohen_kappa_score(np.round(test_machine_score), np.round(test_human_score), weights='quadratic')

            print('count=', count)
            count += 1

            print('Test accuracy: ', _accuracy)
            print('Test kappa: ', kappa_test)
            kappas_test.append(kappa_test)

            print('')
##

        print("Optimization Finished!")  # should be around 0.35 after 25 epochs
        print("Total time: {0} seconds".format(time.time() - start_time))


        Y_test = utils.one_hot(y_test, N_CLASSES)

        test_machine_score = np.array([])
        test_human_score = np.array([])
        count = 0

        for X_batch, Y_batch, _seqlen, effective_size in utils.get_batches(X_test, Y_test, BATCH_SIZE, needall='True'):
            X_batch, num_word = utils.batch_zero_pad(X_batch)
            Y_batch = Y_batch.reshape((BATCH_SIZE, N_CLASSES))

            _accuracy, _pred_class = sess.run(
                [accuracy, pred_class],
                feed_dict={X: X_batch, Y: Y_batch, Y_onehot_rev: utils.one_hot_reverse(Y_batch)[:, None],
                           dropout_keep_prob: 1.0, seqlen: _seqlen, query_size: effective_size, loss_weight: inv_freq,
                           phase: False})

            y_score = utils.one_hot_reverse(Y_batch[:effective_size])

            test_machine_score = np.append(test_machine_score , _pred_class)
            test_human_score = np.append(test_human_score, y_score)
            assert _pred_class.shape[0] == effective_size
            assert len(y_score) == effective_size

        kappa_test = sklearn.metrics.cohen_kappa_score(np.round(test_machine_score), np.round(test_human_score), weights='quadratic')


        print('count=', count)
        count += 1

        print('Test accuracy: ', _accuracy)
        print('Test kappa: ', kappa_test)


        try:
            os.mkdir('params_results')
        except:
            pass

        output_file_name = './params_results/para_result' + '_' + cell_type + '_' + controller_type + '_' + multilayer + '.csv'
        for ord in range(1000):
            if os.path.exists(
                    './params_results/para_result' + '_' + cell_type + '_' + controller_type + '_' + multilayer + str(ord) + '.csv'):
                continue
            else:
                output_file_name = './params_results/para_result' + '_' + cell_type + '_' + controller_type + '_' + multilayer + str(ord) + '.csv'
                break

        with open(output_file_name, 'w', newline='\n') as fw:
            writer = csv.writer(fw, delimiter=',')
            writer.writerow(['test accuracy', _accuracy, 'kappa_test', kappa_test, 'tot_time', time.time()-start_time])
            writer.writerow(['test_machine_score'])
            writer.writerow(test_machine_score)
            writer.writerow(['test_human_score'])
            writer.writerow(test_human_score)
            writer.writerow(lstm_sizes)
            writer.writerow([['LEARNING_RATE', LEARNING_RATE],
                            ['BATCH_SIZE', BATCH_SIZE],
                            ['DROPOUT', DROPOUT],
                            ['N_EPOCHS', N_EPOCHS],
                            ['l2_reg_lambda', l2_reg_lambda]])
            writer.writerow([''])


    ## Output result:

    def data_write_csv(file_name, datas):
        with open(file_name, 'w+', newline='\n') as file_csv:
        # file_csv = open(file_name,'w+', newline='\n')
            writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            for data in datas:
                writer.writerow(data)
            # writer.write('\n')
        print("save complete")
        print('wrote to' + file_name)

    try:
        os.mkdir('files_to_plot')
    except:
        pass

    output_file_name = './files_to_plot/lstm0' +'_' + cell_type + '_' + controller_type +'_' +multilayer +'.csv'
    for ord in range(1000):
        if os.path.exists('./files_to_plot/lstm' +'_' + cell_type + '_' + controller_type +'_'+multilayer +'_'+ str(ord) + '.csv'):
            continue
        else:
            output_file_name = './files_to_plot/lstm' +'_' + cell_type + '_' + controller_type +'_'+multilayer +'_' + str(ord) + '.csv'
            break

    data_write_csv(output_file_name, [loss_fn, accuracies, kappa_scors, kappas_test])
    # data_write_csv(output_file_name, [accuracies])
    # data_write_csv(output_file_name, [kappa_scors])

    plt.figure()
    plt.plot(loss_fn)
    plt.title('loss_t' + "_"+ cell_type + "_"+ controller_type)
    plt.xlabel('#batch')

    plt.figure()
    plt.plot(accuracies)
    plt.title('training_accuracy'+ "_"+ cell_type + "_"+ controller_type)
    plt.xlabel('batch#')

    plt.figure()
    plt.plot(kappa_scors)
    plt.title('kappa, tot_time = {}'.format(time.time() - start_time) + "_"+ cell_type + "_"+ controller_type)
    plt.xlabel('#epochs')

    plt.figure()
    plt.plot(kappas_test)
    plt.title('test kappa, tot_time = {}'.format(time.time() - start_time) + "_"+ cell_type + "_"+ controller_type)
    plt.xlabel('#epochs')

    plt.figure()
    plt.plot(times)
    plt.title('times')
    plt.xlabel('#batch')

    plt.show()
    dbstop = 0

if __name__ == '__main__':
    # embedding_sizes = [1, 3, 5, 7, 10, 20]
    embedding_sizes = [100]
    for embedding_size in embedding_sizes:
        main(embedding_size)