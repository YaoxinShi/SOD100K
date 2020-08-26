import os
import math
import argparse
import importlib
import time

import numpy as np
import torch
import skimage
from configs import cfg
from skimage import io
from skimage.transform import resize
from model.utils.simplesum_octconv import simplesum

parser = argparse.ArgumentParser(description='PyTorch SOD')

parser.add_argument(
    "--config",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
args = parser.parse_args()
assert os.path.isfile(args.config)
cfg.merge_from_file(args.config)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU)
if cfg.TASK == '':
    cfg.TASK = cfg.MODEL.ARCH

print(cfg)


def main():
    global cfg
    model_lib = importlib.import_module("model." + cfg.MODEL.ARCH)
    predefine_file = cfg.TEST.MODEL_CONFIG
    model = model_lib.build_model(predefine=predefine_file)
    #model.cuda()
    prams, flops = simplesum(model, inputsize=(3, 224, 224), device=0)
    print('  + Number of params: %.4fM' % (prams / 1e6))
    print('  + Number of FLOPs: %.4fG' % (flops / 1e9))
    this_checkpoint = cfg.TEST.CHECKPOINT
    if os.path.isfile(this_checkpoint):
        print("=> loading checkpoint '{}'".format(this_checkpoint))
        checkpoint = torch.load(this_checkpoint, map_location=torch.device('cpu'))
        loadepoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            this_checkpoint, checkpoint['epoch']))
        convert_onnx_openvino = False
        if convert_onnx_openvino:
            print(">>> convert pytorch to onnx")
            x = torch.randn(1, 3, 224, 224, requires_grad=True) # Input to the model. (batch,channel,width,height)
            torch.onnx.export(model,             # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "out.onnx",                # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      verbose=False)             # print out a human-readable representation of the network
            #### Here use opset_version=10. pytorch->onnx warning and onnx->openvino pass.
            # C:\Python36\lib\site-packages\torch\onnx\symbolic_helper.py:243: UserWarning:
            # You are trying to export the model with onnx:Resize for ONNX opset version 10.
            # This operator might cause results to not match the expected results by PyTorch.
            # ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11.
            # Attributes to determine how to transform the input were added in onnx:Resize in
            # opset 11 to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).
            # We recommend using opset 11 and above for models using this operator.
            #### if use opset_version=11. pytorch->onnx pass and onnx->openvino fail.
            # [ ERROR ]  Exception occurred during running replacer "REPLACEMENT_ID" (<class 'extensions.load.onnx.loader.ONNXLoader'>):
            # Unexpected exception happened during extracting attributes for node Resize_51.
            # Original exception message: ONNX Resize operation from opset 11 is not supported.
            #### from https://blog.csdn.net/github_28260175/article/details/105704337
            # F.interpolate(mode=nearest)                       ==> torch.onnx opset10 pass
            # F.interpolate(mode=bilinear, align_corners=False) ==> torch.onnx opset10 warning (may cause inference mismatch)
            # F.interpolate(mode=bilinear, align_corners=True)  ==> torch.onnx opset10 fail
            print(">>> verify onnx model")
            import onnx
            onnx_model = onnx.load("out.onnx")
            onnx.checker.check_model(onnx_model)
            print(">>> convert onnx to OpenVINO")
            os.system('python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" --input_model out.onnx')
            #print(">>> convert onnx to TF")
            #from onnx_tf.backend import prepare
            #tf_rep = prepare(onnx_model, strict=False)
            #tf_rep.export_graph("out.pb")
            #print(">>> convert TF to OpenVINO")
            #os.system('python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" --input_model out.pb')
            #   Take care of the version of tensorflow/onnx/onnx-tf
            #   pip install tensorflow==2.2.0
            #   pip install tensorflow-addons==0.11.1
            #   pip install onnx==1.7.0
            #   pip install --user https://github.com/onnx/onnx-tensorflow/archive/master.zip, will get 1.6.0
            #   "C:\Python36\Scripts\onnx-tf.exe convert -i out.onnx -o out.pb"

        test(model, cfg.TEST.DATASETS, loadepoch)
        eval(cfg.TASK, loadepoch)
    else:
        print(this_checkpoint, "Not found.")


def test(model, test_datasets, epoch):
    model.eval()
    print("Start testing.")
    for dataset in test_datasets:
        sal_save_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK,
                                    dataset + '_' + str(epoch))
        os.makedirs(sal_save_dir, exist_ok=True)
        img_dir = os.path.join(cfg.TEST.DATASET_PATH, dataset, 'images')
        img_list = os.listdir(img_dir)
        count = 0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        time_s = time.time()
        with torch.no_grad():
            for img_name in img_list:
                img = skimage.img_as_float(
                    io.imread(os.path.join(img_dir, img_name)))
                h, w = img.shape[:2]
                if cfg.TEST.IMAGE_H != 0 and cfg.TEST.IMAGE_W != 0:
                    img = resize(img, (cfg.TEST.IMAGE_H, cfg.TEST.IMAGE_W),
                                 mode='reflect',
                                 anti_aliasing=False)
                else:
                    if h % 16 != 0 or w % 16 != 0:
                        img = resize(
                            img,
                            (math.ceil(h / 16) * 16, math.ceil(w / 16) * 16),
                            mode='reflect',
                            anti_aliasing=False)
                img = np.transpose((img - mean) / std, (2, 0, 1))
                img = torch.unsqueeze(torch.FloatTensor(img), 0)
                input_var = torch.autograd.Variable(img)
                #input_var = input_var.cuda()
                predict = model(input_var)

                print("@@@@@@")
                print(predict.shape)
                print(predict)
                print("@@@@@@")

                predict = predict[0]
                predict = torch.sigmoid(predict.squeeze(0).squeeze(0))

                print("@@@@@@")
                print(predict.shape)
                print(predict)
                print("@@@@@@")

                predict = predict.data.cpu().numpy()
                predict = (resize(
                    predict, (h, w), mode='reflect', anti_aliasing=False) *
                           255).astype(np.uint8)
                save_file = os.path.join(sal_save_dir, img_name[0:-4] + '.png')
                io.imsave(save_file, predict)
                count += 1
        print('Dataset: {}, {} images'.format(dataset, len(img_list)))
        time_e = time.time()
        print('Speed: %f FPS' % (len(img_list)/(time_e-time_s)))


def eval(method, epoch):
    evalcmd = "python eval.py --method " + method + " --range " + str(
        epoch) + "," + str(epoch + 1)
    print(evalcmd)
    content = os.popen(evalcmd).read()
    print(content)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
