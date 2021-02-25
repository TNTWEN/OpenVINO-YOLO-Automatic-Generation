# Tips

Support OpenVINO2020R4,OpenVINO2021.1,,OpenVINO2021.2 or newer

The following script commands take openvino2020R4 as an example

If you use other versions of OpenVINO, just change the relevant file path

# yolov4-tiny 

```
python parse_config.py --cfg cfg/yolov4-tiny.cfg
```

copy all the output and paste [here](https://github.com/TNTWEN/OpenVINO-YOLO-Automatic-Generation/blob/4d4ae3361a33bb4552562295273242e557f2673b/yolov4tiny/yolo_v4_tiny.py#L81):

Then we can convert .weights -> .pb -> OpenVINO



```
#OpenVINO 2020R4 WIN

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny.weights --data_format NHWC --tiny

"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov3_model.pb --transformations_config yolo_v4_tiny.json --batch 1 --reverse_input_channels
```



# yolov4-tiny-3l

```
python parse_config.py --cfg cfg/yolov4-tiny-3l.cfg --threel
```

copy all the output and paste [here](https://github.com/TNTWEN/OpenVINO-YOLO-Automatic-Generation/blob/4d4ae3361a33bb4552562295273242e557f2673b/yolov4tiny/yolo_v4_tiny.py#L81)

Then we can convert .weights -> .pb -> OpenVINO

```
#OpenVINO 2020R4 WIN 

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny-3l.weights --data_format NHWC --tiny

"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov3_model.pb --transformations_config yolo_v4_tiny_3l.json --batch 1 --reverse_input_channels
```

