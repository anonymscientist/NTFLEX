#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
#from sys import settrace

# def my_tracer(frame, event, arg = None): 
#     # extracts frame code 
#     code = frame.f_code 
  
#     # extracts calling function name 
#     func_name = code.co_name 
  
#     # extracts the line number 
#     line_no = frame.f_lineno 
  
#     print(f"A {event} encountered in {func_name}() at line number {line_no} ") 
  
#     return my_tracer 

#settrace(my_tracer)

ll = ctypes.cdll.LoadLibrary
lib = ll("./initEA.so")
test_lib = ll("./test.so")

class Config(object):

    def __init__(self):
        # lib.setInPath("data")
        # test_lib.setInPath("data")
        lib.showPath()
        test_lib.showPath()
        self.testFlag = True
        self.loadFromData = True
        self.L1_flag = True
        self.hidden_size = 50
        self.nbatches = 100
        self.entity = 0
        self.relation = 0
        self.trainTimes = 3000
        self.margin = 4.0
        self.learning_rate = 0.001
        self.attribute = 0
        self.alpha = 0.6

class TransEModel(object):

    def __init__(self, config):

        entity_total = config.entity
        relation_total = config.relation
        batch_size = config.batch_size
        size = config.hidden_size
        margin = config.margin

        attribute_total = config.attribute
        batch_sizeA = config.batch_sizeA
        alpha = config.alpha

        batch_size_attr_head = config.batch_size_attr_head
        batch_size_attr_tail = config.batch_size_attr_tail

        self.pos_h = tf.compat.v1.placeholder(tf.int32, [None])
        self.pos_t = tf.compat.v1.placeholder(tf.int32, [None])
        self.pos_r = tf.compat.v1.placeholder(tf.int32, [None])

        self.neg_h = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_t = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_r = tf.compat.v1.placeholder(tf.int32, [None])

        self.e = tf.compat.v1.placeholder(tf.int32,[None])
        self.a = tf.compat.v1.placeholder(tf.int32,[None])
        self.v = tf.compat.v1.placeholder(tf.float32,[None])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.compat.v1.get_variable(name = "ent_embedding", shape = [entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
            self.rel_embeddings = tf.compat.v1.get_variable(name = "rel_embedding", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
            self.attr_embeddings = tf.compat.v1.get_variable(name = "attr_embedding", shape = [attribute_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
            self.b = tf.compat.v1.get_variable(name = "bias", shape = batch_sizeA, initializer = tf.constant_initializer(0.01))
            self.b_head = tf.compat.v1.get_variable(name="biasTestHead", shape=batch_size_attr_head, initializer=tf.constant_initializer(0.01))
            self.b_tail = tf.compat.v1.get_variable(name="biasTestTail", shape=batch_size_attr_tail, initializer=tf.constant_initializer(0.01))

            ent_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.ent_embeddings), 1, keepdims = True))
            rel_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.rel_embeddings), 1, keepdims = True))
            attr_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.attr_embeddings), 1, keepdims = True))

            self.ent_embeddings = self.ent_embeddings / ent_l2_norm
            self.rel_embeddings = self.rel_embeddings / rel_l2_norm
            self.attr_embeddings = self.attr_embeddings / attr_l2_norm

            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            e_e = tf.nn.embedding_lookup(self.ent_embeddings, self.e)
            a_v = tf.nn.embedding_lookup(self.attr_embeddings, self.a)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims = True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims = True)
            self.predict = pos
            self.predict_attr_head = abs(tf.reduce_sum(a_v * e_e, axis=1) + self.b_head - self.v)
            self.predict_attr_tail = abs(tf.reduce_sum(a_v * e_e, axis=1) + self.b_tail - self.v)
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims = True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims = True)
            self.predict = pos
            self.predict_attr_head = (tf.reduce_sum(a_v * e_e, axis=1) + self.b_head - self.v) ** 2
            self.predict_attr_tail = (tf.reduce_sum(a_v * e_e, axis=1) + self.b_tail - self.v) ** 2

        with tf.name_scope("output"):
            aV = tf.reduce_sum(a_v * e_e, axis = 1)
            target = aV + self.b - self.v
            attr_loss = tf.reduce_sum(abs(target), keepdims = True)
            rel_loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))
            self.loss = alpha * attr_loss + (1 - alpha) * rel_loss


def main(_):
    config = Config()
    if (config.testFlag):
        test_lib.init()
        config.relation = test_lib.getRelationTotal()
        config.entity = test_lib.getEntityTotal()
        config.batch = test_lib.getEntityTotal()
        config.batch_size = config.batch
        config.batch_sizeA = test_lib.getAttpTotal() // config.nbatches
        config.batch_size_attr_head = test_lib.getEntityTotal()
        config.batch_size_attr_tail = test_lib.getValueTotal()
        config.attribute = test_lib.getAttributeTotal()

    else:
        lib.init()
        config.relation = lib.getRelationTotal()
        config.entity = lib.getEntityTotal()
        config.batch_size = lib.getTripleTotal() // config.nbatches
        config.attribute = lib.getAttrTotal()
        config.batch_sizeA = lib.getAttpTotal() // config.nbatches
        config.batch_size_attr_head = lib.getEntityTotal()
        config.batch_size_attr_tail = lib.getValueTotal()

    with tf.Graph().as_default():
        path = './model/WI2/EA/L1_50_4_0.001/'
        sess = tf.compat.v1.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            with tf.compat.v1.variable_scope("model", reuse=None, initializer = initializer):
                trainModel = TransEModel(config = config)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(trainModel.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())

            def train_step(e_batch, a_batch, v_batch, pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
                feed_dict = {
                    trainModel.e: e_batch,
                    trainModel.a: a_batch,
                    trainModel.v: v_batch,
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                    trainModel.neg_h: neg_h_batch,
                    trainModel.neg_t: neg_t_batch,
                    trainModel.neg_r: neg_r_batch
                }
                _, step, loss = sess.run(
                    [train_op, global_step, trainModel.loss], feed_dict)
                return loss

            def test_step(pos_h_batch, pos_t_batch, pos_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                }
                step, predict = sess.run(
                    [global_step, trainModel.predict], feed_dict)
                print("step rel: {0}, predict:{1}".format(step, predict))
                return predict
            
            def test_step_attr_head(pos_e_batch, pos_a_batch, pos_v_batch):
                feed_dict = {
                    trainModel.e: pos_e_batch,
                    trainModel.a: pos_a_batch,
                    trainModel.v: pos_v_batch,
                }
                step, predict = sess.run(
                    [global_step, trainModel.predict_attr_head], feed_dict)
                print("step att: {0}, predict:{1}".format(step, predict))
                return predict
            
            def test_step_attr_tail(pos_e_batch, pos_a_batch, pos_v_batch):
                feed_dict = {
                    trainModel.e: pos_e_batch,
                    trainModel.a: pos_a_batch,
                    trainModel.v: pos_v_batch,
                }
                step, predict = sess.run(
                    [global_step, trainModel.predict_attr_tail], feed_dict)
                print("step att: {0}, predict:{1}".format(step, predict))
                return predict

            ph = np.zeros(config.batch_size, dtype = np.int32)
            pt = np.zeros(config.batch_size, dtype = np.int32)
            pr = np.zeros(config.batch_size, dtype = np.int32)
            nh = np.zeros(config.batch_size, dtype = np.int32)
            nt = np.zeros(config.batch_size, dtype = np.int32)
            nr = np.zeros(config.batch_size, dtype = np.int32)

            ph_addr = ph.__array_interface__['data'][0]
            pt_addr = pt.__array_interface__['data'][0]
            pr_addr = pr.__array_interface__['data'][0]
            nh_addr = nh.__array_interface__['data'][0]
            nt_addr = nt.__array_interface__['data'][0]
            nr_addr = nr.__array_interface__['data'][0]

            e = np.zeros(config.batch_sizeA, dtype = np.int32)
            a = np.zeros(config.batch_sizeA, dtype = np.int32)
            v = np.zeros(config.batch_sizeA, dtype = np.float64)

            e_addr = e.__array_interface__['data'][0]
            a_addr = a.__array_interface__['data'][0]
            v_addr = v.__array_interface__['data'][0]

            e_h = np.zeros(config.batch_size_attr_head, dtype = np.int32)
            a_h = np.zeros(config.batch_size_attr_head, dtype = np.int32)
            v_h = np.zeros(config.batch_size_attr_head, dtype = np.float64)

            e_h_addr = e_h.__array_interface__['data'][0]
            a_h_addr = a_h.__array_interface__['data'][0]
            v_h_addr = v_h.__array_interface__['data'][0]

            e_t = np.zeros(config.batch_size_attr_tail, dtype = np.int32)
            a_t = np.zeros(config.batch_size_attr_tail, dtype = np.int32)
            v_t = np.zeros(config.batch_size_attr_tail, dtype = np.float64)

            e_t_addr = e_t.__array_interface__['data'][0]
            a_t_addr = a_t.__array_interface__['data'][0]
            v_t_addr = v_t.__array_interface__['data'][0]

            lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            lib.getAttrBatch.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
            test_lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.testHead.argtypes = [ctypes.c_void_p]
            test_lib.testTail.argtypes = [ctypes.c_void_p]
            test_lib.getHeadAttrBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.getTailAttrBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.testHeadAttr.argtypes = [ctypes.c_void_p]
            test_lib.testTailAttr.argtypes = [ctypes.c_void_p]

            if not config.testFlag:
                for times in range(config.trainTimes):
                    res = 0.0
                    for batch in range(config.nbatches):
                        lib.getAttrBatch(e_addr, a_addr, v_addr, config.batch_sizeA)
                        lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
                        res += train_step(e,a,v,ph, pt, pr, nh, nt, nr)
                        current_step = tf.compat.v1.train.global_step(sess, global_step)
                    print (times) 
                    print (res)
                if not os.path.exists(path):
                    os.makedirs(path)
                saver.save(sess, path + 'model.vec',global_step = global_step)

            else:
                if (config.loadFromData):
                    ckpt = tf.compat.v1.train.get_checkpoint_state(path)
                    if ckpt and ckpt.model_checkpoint_path:
                        for i in ckpt.all_model_checkpoint_paths:
                            saver.restore(sess,i)
                            total = test_lib.getTestTotal()
                            attrTotal = test_lib.getAttrTestTotal()
                            for times in range(total):
                                test_lib.getHeadBatch(ph_addr, pt_addr, pr_addr)
                                res = test_step(ph, pt, pr)
                                test_lib.testHead(res.__array_interface__['data'][0])

                                test_lib.getTailBatch(ph_addr, pt_addr, pr_addr)
                                res = test_step(ph, pt, pr)
                                test_lib.testTail(res.__array_interface__['data'][0])

                            for times in range(attrTotal):
                                test_lib.getHeadAttrBatch(e_h_addr, a_h_addr, v_h_addr)
                                res = test_step_attr_head(e_h, a_h, v_h)
                                test_lib.testHeadAttr(res.__array_interface__['data'][0])

                                test_lib.getTailAttrBatch(e_t_addr, a_t_addr, v_t_addr)
                                res = test_step_attr_tail(e_t, a_t, v_t)
                                test_lib.testTailAttr(res.__array_interface__['data'][0])
                                # print (times)
                            if (times % 50 == 0):
                                test_lib.test()
                            test_lib.test()

if __name__ == "__main__":
    tf.compat.v1.app.run()
