import tensorflow as tf
import numpy as np
import time, os
import matplotlib.pyplot as plt
import csv, math

import sklearn.metrics

from helper import Helper
import utils

##

def main(embedding_size):
    tf.reset_default_graph()

    #-------------------------
    # Read in data
    #-------------------------

    # readfile = './data/small.tsv'
    readfile = './data/training_set_rel3.tsv'
    X_, _, y_ = Helper(set_num=0, file_name=readfile).get_embed(embedding_size)

    print('reading file: ', readfile)
    benchmark_score = 8 * np.ones_like(y_)
    print('accuracy: ', np.sum(np.where(benchmark_score != y_, 1, 0))/len(y_))

    data_size = y_.shape[0]
    train_size = math.floor(0.9 * data_size)

    X_train = X_[:train_size]
    y_train = y_[:train_size]
    X_test = X_[train_size:]
    y_test = y_[train_size:]



    #-------------------------
    # Parameters
    #-------------------------

    classification = True
    withWeight = False
    print('classification: ', classification)
    print('withWeight:', withWeight)

    if withWeight:
        _, inv_freq = utils.hist_freq(y_train, 12)
    else:
        inv_freq = np.ones(12)

    N_CLASSES = 12 if classification else 1

    deep_nets = [[32, 32], [64, 64, 32], [128], [128, 64], [128, 64, 32], [128, 64, 128, 64, 32], [32]*20]
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
    DROPOUT = 0.5 # the keep_prob
    N_EPOCHS = 50
    l2_reg_lambda = 0.0

    print('LEARNING_RATE:', LEARNING_RATE, 'DROPOUT:', DROPOUT,
          'l2_lambda:', l2_reg_lambda, 'embedding_size:', embedding_size)

    DISP_STEP = 1

    bidi = tf.constant(1, dtype=tf.int32)
    if controller_type == "bidirection":
        bidi = tf.constant(2, dtype=tf.int32)
    np.random.seed(0)

    print(cell_type)
    print(controller_type)
    print(lstm_sizes)

    Y_train = utils.one_hot(y_train, N_CLASSES)
    
##
    #-------------------------
    # Utility functions
    #-------------------------

    def get_cell(hidden_size, cell_type):
        if cell_type == "lstm":
            return tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0, initializer=tf.orthogonal_initializer())
        # This is added to biases the forget gate
        # in order to reduce the scale of forgetting in the beginning of the training.

        elif cell_type == "lstm_block":
            return tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=1.0)

        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)

        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

##
    #-------------------------
    # Customized architecture
    #   MoT layer is added in line 175
    #-------------------------
    def get_controller(cell, _X, seqlen, controller_type="one_direction"):
        batch_size = _X.shape[0]
        if controller_type == "one_direction":
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
            last_output = tf.gather_nd(outputs, tf.stack([tf.range(batch_size), tf.maximum(seqlen - 1, tf.zeros([batch_size], dtype=tf.int32))], axis=1))

            elem = (outputs, tf.range(batch_size))
            mp_output = tf.map_fn(lambda seq: tf.reduce_mean(tf.slice(seq[0], [0, 0], [seqlen[seq[1]], outputs.shape[2]]), axis=0),
                                  elem, dtype=tf.float32)

            assert last_output.shape == mp_output.shape
            return mp_output

        elif controller_type == "bidirection":
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
            last_outputs = [tf.gather_nd(outputs[i], tf.stack([tf.range(batch_size), tf.maximum(seqlen - 1, tf.zeros([batch_size], dtype=tf.int32))], axis=1)) for i in range(2)]

            return tf.add(last_outputs[0], last_outputs[1])

        else:
            print("ERROR: '" + controller_type + "' is a wrong controller type. Use default.")
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
            last_output = tf.gather_nd(outputs, tf.stack([tf.range(BATCH_SIZE), tf.maximum(seqlen - 1, tf.zeros([batch_size], dtype=tf.int32))], axis=1))

            outputs = tf.slice(outputs, [0, 0, 0], [batch_size, seqlen, outputs.shape[2]])
            mp_output = tf.reduce_mean(outputs, axis=1)

            return last_output, mp_output


##
    #-------------------------
    # Build the graph
    #-------------------------

    with tf.name_scope('data'):
        seqlen = tf.placeholder(tf.int32, [BATCH_SIZE], name='sequence_len')
        X = tf.placeholder(tf.float32, [BATCH_SIZE, None, embedding_size], name="X_placeholder")
        Y = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES], name="Y_placeholder")
        Y_onehot_rev = tf.placeholder(tf.float32, [BATCH_SIZE, None], name="Y_onehot_rev_placeholder")

        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        l2_loss = tf.constant(0.0)
        loss_weight = tf.placeholder(tf.float32, [N_CLASSES])

        phase = tf.placeholder(tf.bool, name='phase')

    with tf.name_scope('process_data'):
        w1 = tf.Variable(tf.truncated_normal(shape=[embedding_size, lstm_sizes[0]], stddev=1.0), name='w1')
        b1 = tf.Variable(tf.truncated_normal([lstm_sizes[0]], mean=0.0, stddev=1.0), name='b1')
        _X = X
        l2_loss += tf.nn.l2_loss(w1)
        l2_loss += tf.nn.l2_loss(b1)

    with tf.name_scope('lstm') as scope:

        lstm_cells = [get_cell(lstm_size, cell_type=cell_type) for lstm_size in lstm_sizes ]
        dropped_cells = [tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob) for lstm_cell in lstm_cells]
        cell = tf.contrib.rnn.MultiRNNCell(dropped_cells)
        last_output = get_controller(cell, _X, seqlen, controller_type=controller_type)

    with tf.variable_scope('Output_layer') as scope:
        w2 = tf.Variable(tf.truncated_normal(shape=[lstm_sizes[-1], N_CLASSES]), name='w2')
        b2 = tf.Variable(tf.truncated_normal([N_CLASSES], mean=0.0, stddev=1.0), name='b2')

        logits = tf.nn.xw_plus_b(last_output, w2, b2, name='logits')  # batch_size * n_class

        l2_loss += tf.nn.l2_loss(w2)
        l2_loss += tf.nn.l2_loss(b2)

    with tf.name_scope('loss') as scope:

        if classification:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name='loss') + l2_reg_lambda * l2_loss
        else:
            loss = tf.reduce_sum(tf.square(logits - Y_onehot_rev), name='lossFunction') + l2_reg_lambda * l2_loss

    with tf.name_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

    with tf.name_scope('accuracy') as scope:
        query_size = tf.placeholder(dtype=tf.int32, shape=None)

        if classification:
            score_class = tf.ones((query_size, 1)) * tf.cast(tf.range(N_CLASSES) + 1, tf.float32)
            prob = tf.nn.softmax(logits[:query_size])
            pred_class = tf.round(tf.reduce_sum(tf.multiply(prob, score_class), 1))

            correct_preds = tf.equal(tf.to_int64(pred_class), tf.argmax(Y[:query_size], 1)+1)
            accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        else:
            pred_class = tf.round(logits[:query_size])

            correct_preds = tf.equal(tf.to_int64(pred_class), tf.argmax(Y[:query_size], 1) + 1)
            accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))


##
    #-------------------------
    # Model training
    #-------------------------
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./graphs/aes', sess.graph)
        writer.close()

        start_time = time.time()
        tic_time = time.time()

        total_loss = 0.0
        index_batch = 0

        kappa_scors = []
        kappas_test = []
        accuracies = []
        loss_fn = []
        times = []

        for i in range(N_EPOCHS):  # train the model n_epochs times
            machine_score = np.array([])
            human_score = np.array([])

            for X_batch, Y_batch, _seqlen, _ in utils.get_batches(X_train, Y_train, BATCH_SIZE):
                X_batch, num_word = utils.batch_zero_pad(X_batch)
                Y_batch = Y_batch.reshape((BATCH_SIZE, N_CLASSES))

                _, loss_batch, _accuracy, _pred_class = sess.run([optimizer, loss, accuracy, pred_class],
                            feed_dict={X: X_batch, Y: Y_batch, Y_onehot_rev: utils.one_hot_reverse(Y_batch)[:, None],
                                       dropout_keep_prob: DROPOUT, seqlen: _seqlen, query_size: BATCH_SIZE, loss_weight: inv_freq,
                                       phase: True})

                total_loss += loss_batch

                y_score = utils.one_hot_reverse(Y_batch)

                machine_score = np.append(machine_score, _pred_class, axis=0)
                human_score = np.append(human_score, y_score, axis=0)

                print('Accuracy:', _accuracy)
                index_batch += 1


                if (index_batch + 1) % DISP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index_batch + 1, total_loss / DISP_STEP))

                    loss_fn.append(total_loss / DISP_STEP)
                    accuracies.append(_accuracy)
                    times.append(time.time() - tic_time)
                    tic_time = time.time()

                    total_loss = 0.0


            score = sklearn.metrics.cohen_kappa_score(np.round(machine_score), np.round(human_score), weights='quadratic')
            print("Kappa score: ", score)
            print("epoch: {}".format(i))

            kappa_scors.append(score)

            # Watch on test set
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

        print("Optimization Finished!")  # should be around 0.35 after 25 epochs
        print("Total time: {0} seconds".format(time.time() - start_time))
##
        # -------------------------
        # Testing
        # -------------------------

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


##
    # -------------------------
    # Save the data for plotting
    # -------------------------

    def data_write_csv(file_name, datas):
        with open(file_name, 'w+', newline='\n') as file_csv:
            writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            for data in datas:
                writer.writerow(data)
        print("save complete")
        print('wrote to' + file_name)

    try:
        os.mkdir('files_to_plot')
    except:
        pass

    for ord in range(1000):
        if os.path.exists('./files_to_plot/lstm' +'_' + cell_type + '_' + controller_type +'_'+multilayer +'_'+ str(ord) + '.csv'):
            continue
        else:
            output_file_name = './files_to_plot/lstm_noMP' +'_' + cell_type + '_' + controller_type +'_'+multilayer +'_' + str(ord) + '.csv'
            break

    data_write_csv(output_file_name, [loss_fn, accuracies, kappa_scors, kappas_test])

##
    # -------------------------
    # Plotting
    # -------------------------
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


if __name__ == '__main__':
    embedding_sizes = [10]
    for embedding_size in embedding_sizes:
        main(embedding_size)