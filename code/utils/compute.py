import numpy as np

def confusion_matrices(sess, feed_iter,
                       tf_true, tf_pred, tf_largest_gate,
                       num_experts, num_classes):
    """
    :return: count {"overall":np[true,pred], "expert":np[i,true,pred], "overall_percent":?}
    """
    print("compute confusion")

    confusion_matrix_experts = np.zeros((num_experts, num_classes, num_classes))
    for feeddict in feed_iter:
        y_trues, y_preds, largest_gates = sess.run((tf_true, tf_pred, tf_largest_gate), feeddict)
        for y_true, y_pred, gate in zip(y_trues, y_preds, largest_gates):
            confusion_matrix_experts[gate, y_true, y_pred] += 1

    confusion_matrix_all = np.sum(confusion_matrix_experts, axis=0)
    confusion_matrix_all_percent = 100 * confusion_matrix_all / np.sum(confusion_matrix_all, axis=1, keepdims=True)


    return {"overall":confusion_matrix_all,
            "overall_percent":confusion_matrix_all_percent,
            "experts":confusion_matrix_experts}

def activation_matrix(sess, feed_iter,
                      tf_activation_label, tf_gate,
                      num_activation_labels, num_experts):

    print("compute activation")

    gate_sum = np.zeros((num_activation_labels, num_experts))
    label_size = np.zeros(num_activation_labels)

    for feeddict in feed_iter:
        b_activation_label, b_gate = sess.run((tf_activation_label, tf_gate),feeddict)
        bool_where = np.equal(  # (s,1) == (1,b) -> (s,b)
            np.arange(num_activation_labels)[:, np.newaxis],
            b_activation_label[np.newaxis, :])

        for super_label in range(num_activation_labels):
            gate_of_label = b_gate[bool_where[super_label]]
            gate_sum[super_label] += np.sum(gate_of_label, axis=0)
        label_size += np.sum(bool_where.astype(np.int32), axis=1)

    activation = gate_sum / label_size[:,np.newaxis]

    return activation
