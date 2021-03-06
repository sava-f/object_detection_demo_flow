As reference follow:
https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_Object_Detection_With_Custom_Data_Demo_Training.ipynb#scrollTo=CjDHjhKQofT5

## Steps
1. Train the neural network
2. Verify the result
3. Convert the network to a blob format

### Environment setup

Log into opecv machine:
        - ssh opencv@10.81.1.118  psw: opencv
source conda environment:
        - su_conda
activate the proper environment
        - conda activate ssd
### Data-set organization
Images are in 1920X1080 format. The annotation is made using: http://www.robots.ox.ac.uk/~vgg/software/via/. The output is in Json format. Since for training we need xml format we have scripts for handling the conversion. Refer to section Data Conversion.
The class to detect are:
-right hole -left side -right side -center

Dataset is organized in the following way:
       object_detection_demo_flow

```
├── data  

    ├── images  

        ├── train 

        ├── test 

        ├── final_test 
```

Train folder contains the 80% of the dataset while test and final_test the remaining 20%. In final_test folder only images and not xml files are saved.



### Training the neural network
launch the training notebook:
Use the train.py script.
        
        python train.py

Arguments (not mandatory) are:
- "-tr", "--train" = folder containing training data (default = ./data/images/train)
- "-te", "--test" = folder containing test data (default = ./data/images/test)
- "-ft", "--final" = folder containing final test data ( default = ./data/images/final_test)
-  "-ns", "--nsteps" = number of steps/epochs (default = 5000)
- "-es", "--estep" = evaluation steps (default = 50)

To visualize results use Tensorboard. 
To launch it run: 
        
        tensorbod --logdir=absolute_path_to_training_folder

Please launch it without activate any conda environment (this will be fixed in the future)
### Run the inference to test the result
Run the local python script (it's not a python notebook):

        python run_inference.py -p [pb file path] -i [input image dir]

pb file is referred to frozen_model_graph

### Conversion to blob for OAK-D board
## Convert th model to an intermediate format IR.
        
        cd to_folder_where_frozen_inference_graph.pb
    
        mkdir convertedFiles
        
execute the python command:

        python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
        --input_model frozen_inference_graph.pb \
        --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
        --tensorflow_object_detection_api_pipeline_config pipeline.config \
        --reverse_input_channels \
        --output_dir ./convertedFiles\
        --data_type FP16

To generate the blob goes to the folder where you saved the xml at prev step (e.g. frozen_inference_graph.xmlconvertedFiles)

        /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -m ./frozen_inference_graph.xml -o frozenGraph.blob -ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4


### Data Conversion
Annotation data have to be converted from json format to xml. Under the folder utils use the json2xml.py script:

        python3 via2coco.py -i [path_to__annotation_file.json] -p [destination_folder]
    

If more images have to be merged in the same folder for the train and test process it may be possible they share the same name. To solve this problem use the script updateFilename.py in dataConversionScriptFolder. Launch this script from the folder where mages and xml files are saved. This will update image and xml filename attaching the current folder name to the actual name plus updates the xml file content.

### Utils folder
Contain useful script for data manipulation:
- json2xml.py = convert the annotation data from json to xml
- resize_images.py = change image size
- updateFilename.py = concatenate folder name and image name and update also the xml annotation file