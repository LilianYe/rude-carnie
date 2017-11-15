from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import cv2
import dlib
import re


RESIZE_FINAL = 227
GENDER_LIST =['M', 'F']
AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
MAX_BATCH_SZ = 128


tf.app.flags.DEFINE_string('model_age_dir', "",
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('model_sex_dir', "",
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('class_type', 'age',
                           'Classification type (age|gender)')

tf.app.flags.DEFINE_string('input', 'video', "video or dir")

tf.app.flags.DEFINE_string('dir_path', '', 'input dir path')
tf.app.flags.DEFINE_string('checkpoint', 'checkpoint', 'Checkpoint basename')

tf.app.flags.DEFINE_string('age_model_type', "inception", 'Type of age convnet (default|bn|inception)')
tf.app.flags.DEFINE_string('sex_model_type', "inception", 'Type of sex convnet (default|bn|inception)')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

tf.app.flags.DEFINE_string('face_detection_type', 'dlib', 'Face detection model type (dlib|yolo_tiny|cascade)')

FLAGS = tf.app.flags.FLAGS


def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])


def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None


def classify(cl_model, label_list, crop_img):
    try:
        image_batch = cv_read(crop_img)

        batch_results = cl_model.run(image_batch)
        output = batch_results[0]
        batch_sz = batch_results.shape[0]

        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)
        nlabels = len(label_list)
        if nlabels > 2:
            output[best] = 0
            second_best = np.argmax(output)
            print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

        return best_choice
    except Exception as e:
        print(e)
        print('Failed to run image ')


class ImportGraph(object):

    def __init__(self, model_dir, class_type):
        self.graph = tf.Graph()
        # config = tf.ConfigProto(allow_soft_placement=True)
        # self.sess = tf.Session(config=config, graph=self.graph)
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # import saved model from loc into local graph
            label_list = AGE_LIST if class_type == 'age' else GENDER_LIST
            nlabels = len(label_list)
            if class_type == "age":
                model_fn = select_model(FLAGS.age_model_type)
            else:
                model_fn = select_model(FLAGS.sex_model_type)

            self.images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, self.images, 1, False)
            # init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            requested_step = FLAGS.requested_step if FLAGS.requested_step else None
            checkpoint_path = '%s' % (model_dir)
            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
            saver.restore(self.sess, model_checkpoint_path)
            self.softmax_output = tf.nn.softmax(logits)

    def run(self, data):
        # with tf.Session().as_default():
        #     data = data.eval()
        return self.sess.run(self.softmax_output, feed_dict={self.images: data})


def frame_classify(frame, face_detect, model_age, model_sex):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectangles = face_detect(gray, 1)
    # change 1 to 0 can make dlib run 4 times faster
    # rectangles = face_detect(gray, 0)
    height, width, _ = frame.shape
    for rectangle in rectangles:
        add_h = int(0.4 * (rectangle.bottom() - rectangle.top()))
        add_w = int(0.4 * (rectangle.right() - rectangle.left()))
        top = rectangle.top() - add_h if (rectangle.top() - add_h) > 0 else 0
        bottom = rectangle.bottom()+add_h if (rectangle.bottom()+add_h) < height else height
        left = rectangle.left()-add_w if (rectangle.left()-add_w) > 0 else 0
        right = rectangle.right()+add_w if (rectangle.right()+add_w) < width else width
        crop_img = frame[top:bottom, left:right]
        cv2.rectangle(frame, (rectangle.left(), rectangle.top()),
                      (rectangle.right(), rectangle.bottom()), (0, 0, 255), 1)

        if model_age:
            start_time = time.time()
            age_label = classify(model_age, AGE_LIST, crop_img)
            print("age %s " % (time.time() - start_time))
            cv2.putText(frame, "age: %s prob: %.2f" % age_label, (rectangle.left(), rectangle.bottom() + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if model_sex:
            start_time = time.time()
            sex_label = classify(model_sex, GENDER_LIST, crop_img)
            print("sex %s " % (time.time() - start_time))
            cv2.putText(frame, "sex: %s prob: %.2f" % sex_label, (rectangle.left(), rectangle.bottom() + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    print("no face")
    cv2.imshow("detect", frame)
    cv2.waitKey(1)


def run_camera():
    if FLAGS.face_detection_type == "dlib":
        face_detect = dlib.get_frontal_face_detector()
    model_age = ImportGraph(FLAGS.model_age_dir, "age")
    model_sex = ImportGraph(FLAGS.model_sex_dir, "sex")

    camera = cv2.VideoCapture(0)
    count = 0
    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)

    while camera.isOpened():
        count += 1
        ret, frame = camera.read()

        if not ret:
            print('camera error')
            break
        if count > 60:
            count = 0
            start_time = time.time()
            frame_classify(frame, face_detect, model_age, model_sex)
            print("total %s " % (time.time() - start_time))


def run_file_dir(file_dir):
    if FLAGS.face_detection_type == "dlib":
        face_detect = dlib.get_frontal_face_detector()

    model_sex = ImportGraph(FLAGS.model_sex_dir, "sex")
    model_age = ImportGraph(FLAGS.model_age_dir, 'age')

    onlyfiles = [f for f in os.listdir(file_dir) if re.search('jpg', f)]
    total = len(onlyfiles)
    print(total)
    for file_n in onlyfiles:
        frame = cv2.imread(os.path.join(file_dir, file_n))
        frame_classify(frame, face_detect, model_age, model_sex)


def main(_):  # pylint: disable=unused-argument
    if FLAGS.input == "video":
        run_camera()
    else:
        if FLAGS.dir_path:
            run_file_dir(FLAGS.dir_path)


if __name__ == '__main__':
    tf.app.run()
