import os
import argparse
import numpy as np
import tensorflow as tf
import cv2

# import keras

classes = {1: "center",
           2: "face",
           3: "right_side",
           4: "left_side"}
colors = {"center": (0, 0, 0),
          "face": (0, 0, 255),
          "right_side": (255, 255, 0),
          "left_side": (255, 0, 0)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize inferenece results"
    )

    parser.add_argument(
        "-p", "--pb_file",
        help="Path to the pb file",
        type=str,
        default="graph.pb"
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        help="Path to the input image folder",
        type=str,
    )
    parser.add_argument(
        "--min_score",
        help="Minimum score of the detected boxes to show",
        type=float,
        default = 0.2
    )

    args = parser.parse_args()

    return args


def writeConfidence(img, val, x, y, class_name):
    pos = (x, y - 20)
    col = colors[class_name]
    str_val = "{:.2f}".format(val)
    cv2.putText(img, str_val, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)


def main():
    args = parse_args()
    pb_file = args.pb_file
    image_dir = args.image_dir
    min_score = args.min_score

    # Read the graph.
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

    with tf.Session(config=config) as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        imgs = [img_file for img_file in os.listdir(image_dir)
                    if os.path.splitext(img_file)[1] in [".png", ".jpg"]]
        for img_file in imgs:
            img_path = os.path.join(image_dir, img_file)

            # Read and preprocess an image.
            img = cv2.imread(img_path)
            cols, rows = img.shape[1], img.shape[0]

            # Run the model
            # input image should be BGR
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                            feed_dict={'image_tensor:0': img[:, :, ::-1].reshape(1, img.shape[0], img.shape[1], 3)})

            num_detections = int(out[0][0])

            found = set()
            for i in range(num_detections):
                class_name = classes[out[3][0][i]]
                if class_name in found:
                    continue
                found.add(class_name)

                score = float(out[1][0][i])
                if score < min_score:  # Already sorted
                    break

                bbox = [float(v) for v in out[2][0][i]]
                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)

                cv2.rectangle(img, (x, y), (right, bottom), colors[class_name], thickness=2)
                writeConfidence(img, score, x, y, class_name)

                if len(found) == len(classes):
                    break

            cv2.imshow("bounding boxes", img)
            k = cv2.waitKey(0)
            cv2.destroyWindow("bounding boxes")

            if k == 27:
                break

if __name__ == '__main__':
    main()
