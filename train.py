import os
import sys
import argparse
import subprocess
import shutil
import glob
import urllib.request
import tarfile
import re
import numpy as np
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Neural Network"
    )

    parser.add_argument(
        "-tr", "--train",
        help="folder containing training data",
        type=str,
        default="./data/images/train"
    )
    parser.add_argument(
        "-te", "--test",
        help="folder containing test data",
        type=str,
        default="./data/images/test"
    )
    parser.add_argument(
        "-ft", "--final",
        help="folder containing final test data",
        type=str,
        default="./data/images/final_test"
    )    
    parser.add_argument(
        "-ns", "--nsteps",
        help="number of steps(epochs)",
        type=int,
        default=5000
    )
    parser.add_argument(
        "-es", "--estep",
        help="evaluation steps",
        type=int,
        default=50
    )        

    args = parser.parse_args()
    return args

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

if __name__ == "__main__":
    args = parse_args()
    train_folder = args.train
    test_folder = args.test
    final_test_folder = args.final
    if os.path.isdir(train_folder) and os.path.isdir(test_folder) and os.path.isdir(final_test_folder):
        print(train_folder)
        print(test_folder)
        print(final_test_folder)
    else:
        print("wrong data folder")

    sys.path.append('/home/opencv/src/models/research')
    sys.path.append('/home/opencv/src/models/research/slim/')
    if 'PYTHONPATH' not in os.environ:
        os.environ['PYTHONPATH'] = ""
    os.environ['PYTHONPATH'] += ':/home/opencv/src/models/research:/home/opencv/src/models/research/slim/'
    os.environ['PYTHONPATH'] += ':/home/opencv/src/models/research:/home/opencv/src/models/research/'
    #num_steps = 500  # A step means using a single batch of data. larger batch, less steps required
    num_steps = args.nsteps
    #Number of evaluation steps.
    #num_eval_steps = 50 #50
    num_eval_steps = args.estep
    print("n Epochs: " + str(num_steps))
    print("n eval steps: " + str(num_eval_steps))
    #Batch size 24 is a setting that generally works well. can be changed higher or lower 
    MODELS_CONFIG = {
            'ssd_mobilenet_v2': {
            'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
            'pipeline_file': 'ssd_mobilenet_v2_coco.config',
            'batch_size': 24
        }
    }
    selected_model = 'ssd_mobilenet_v2'
    # Name of the object detection model to use.
    MODEL = MODELS_CONFIG[selected_model]['model_name']
    # Name of the pipline file in tensorflow object detection API.
    pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']
    # Training batch size fits in Colab's GPU memory for selected model.
    batch_size = MODELS_CONFIG[selected_model]['batch_size']
    subprocess.call("protoc /home/opencv/src/models/research/object_detection/protos/*.proto --python_out=/home/opencv/src/models/research/object_detection/protos/",shell=True)

    #!protoc object_detection/protos/*.proto --python_out=.
    subprocess.call("python ~/src/models/research/object_detection/builders/model_builder_test.py", shell=True)
    # Convert train folder annotation xml files to a single csv file,
    # generate the `label_map.pbtxt` file to `data/` directory as well.
    subprocess.call("python xml_to_csv.py -i " + train_folder +" -o data/annotations/train_labels.csv -l data/annotations", shell=True)
    # Convert test folder annotation xml files to a single csv.
    subprocess.call("python xml_to_csv.py -i " + test_folder + " -o data/annotations/test_labels.csv", shell=True)
    # Generate `train.record`
    subprocess.call("python generate_tfrecord.py --csv_input=data/annotations/train_labels.csv --output_path=data/annotations/train.record --img_path=data/images/train --label_map data/annotations/label_map.pbtxt", shell=True)
    # Generate `test.record`
    subprocess.call("python generate_tfrecord.py --csv_input=data/annotations/test_labels.csv --output_path=data/annotations/test.record --img_path=data/images/test --label_map data/annotations/label_map.pbtxt", shell=True)
    # Set the paths
    test_record_fname = '/home/opencv/src/object_detection_demo_flow/data/annotations/test.record'
    train_record_fname = '/home/opencv/src/object_detection_demo_flow/data/annotations/train.record'
    label_map_pbtxt_fname = '/home/opencv/src/object_detection_demo_flow/data/annotations/label_map.pbtxt'
    print("All paths are set")
    MODEL_FILE = MODEL + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    DEST_DIR = '/home/opencv/src/models/research/pretrained_model'

    if not (os.path.exists(MODEL_FILE)):
        urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

    tar = tarfile.open(MODEL_FILE)
    tar.extractall()
    tar.close()

    os.remove(MODEL_FILE)
    if (os.path.exists(DEST_DIR)):
        shutil.rmtree(DEST_DIR)
    os.rename(MODEL, DEST_DIR)
    print("Pretrained model downloaded")
    fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")
    fine_tune_checkpoint
    pipeline_fname = os.path.join('/home/opencv/src/models/research/object_detection/samples/configs/', pipeline_file)

    assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)
    iou_threshold = 0.50
    num_classes = get_num_classes(label_map_pbtxt_fname)
    with open(pipeline_fname) as f:
        s = f.read()
    with open(pipeline_fname, 'w') as f:
        
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
        
        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
            '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(num_steps), s)
        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(num_classes), s)
        # Set number of classes num_classes.
        s = re.sub('iou_threshold: [0-9].[0-9]+',
                'iou_threshold: {}'.format(iou_threshold), s)
        
        f.write(s)

        #Create training folder
        timestamp = datetime.timestamp(datetime.now())
        dt_object = datetime.fromtimestamp(timestamp)
        folder_timestamp = str(dt_object.year)+"_"+str(dt_object.month)+"_"+str(dt_object.day)+"_"+str(dt_object.hour)+"_"+str(dt_object.minute)
        training_folder = "training_"+folder_timestamp
        print("Training_folder: " + training_folder)
        # necessary
    model_dir = training_folder

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Add tensor board

    model_dir = training_folder

    training_cmd = "python /home/opencv/src/models/research/object_detection/model_main.py --pipeline_config_path="+pipeline_fname+" --model_dir="+model_dir + " --alsologtostderr --num_train_steps="+str(num_steps)+" --num_eval_steps="+str(num_eval_steps)
    print(training_cmd)
    # Train the network
    subprocess.call(training_cmd, shell = True)
    print("Training Complete")
    fg_out_dir = "frozen_" + folder_timestamp
    print("Saving the frozen graphinto: " + fg_out_dir)

    lst = os.listdir(model_dir)
    lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
    steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
    last_model = lst[steps.argmax()].replace('.meta', '')
    last_model_path = os.path.join(model_dir, last_model)
    print(last_model_path)
    frozen_graph_cmd = "python ~/src/models/research/object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=" + pipeline_fname + " --output_directory=" + fg_out_dir + " --trained_checkpoint_prefix=" + last_model_path
    subprocess.call(frozen_graph_cmd, shell = True)
    print("saving end")



        