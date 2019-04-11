build\tools\Release\caffe train -solver models\SSD\solver.prototxt --weights=models\SSD\VGG_ILSVRC_16_layers_fc_reduced.caffemodel
::--weights=models\SSD\VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
::--weights=models\MobileNetV2\deploy_voc.caffemodel
::--snapshot=models\yolov2\MobileNetYOLO-V2_deploy_iter_25000.solverstate
