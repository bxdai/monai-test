import argparse
import torch
import numpy as np
import os
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activations, AddChannel, AsDiscrete, Compose, 
    LoadImage,SaveImage, ScaleIntensity, EnsureType,ScaleIntensityRanged,
    EnsureType,SqueezeDim,CropForeground
)


#%%
def main():
    """
    main function

    """

       #解析参数
    parser = argparse.ArgumentParser()
    parser.description = 'please enter two parameters input and outputfile ...'
    parser.add_argument("-i", "--input", help="this is parameter dicom filename",
                        dest="dcmfile", type=str, default=None)
    parser.add_argument("-o", "--output", help="this is parameter labelMap filepath",
                        dest="labelmap", type=str, default=None)
    args = parser.parse_args()
    if args.dcmfile == None:
        print("please input dcm filename")
        exit()
    if args.labelmap == None:
        print("please input label filepath")
        exit()
    # print("parameter dcmfile is :",args.dcmfile)
    # print("parameter labelfile is :",args.labelmap)
    dcmfile_abs = os.path.realpath(args.dcmfile)
    labelmap_abs = os.path.realpath(args.labelmap)
    

    loader = LoadImage()
    data,meta = loader(dcmfile_abs)
    dcm_data = np.asarray(data)

    print(f"dcm file: {dcmfile_abs}")
    #cpath = os.path.dirname(__file__)
    #print(f"currnet_file:{cpath}")
    # define transforms for image and segmentation
    # infer_transforms = Compose(
    #     [            
    #         LoadImaged(keys="img"),
    #         SqueezeDimd(keys="img", dim=-1),
    #         AddChanneld(keys="img"),
    #         #ScaleIntensityRanged(keys="img",
    #         # a_min =0,a_max=65535.0,
    #         # b_min=0.0,b_max=1.0,
    #         # clip=True
    #         # ),
    #         ScaleIntensityd(keys="img"),
    #         CropForegroundd(keys="img", source_key="img"),
    #         EnsureTyped(keys="img"),
    #     ]
    # )

    # define transforms for image and segmentation
    infer_transforms = Compose(
        [            
            SqueezeDim(dim=-1),
            AddChannel(),         
            ScaleIntensity(),
            CropForeground(),
            EnsureType(),
        ]
    )


    img_tensor = infer_transforms(dcm_data)
    
    #后处理Compose
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    #定义保存文件对象
    saver = SaveImage(output_dir=labelmap_abs, output_ext=".nii.gz", output_postfix="segmnet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),"model_weights_dict.pth")))

    model.eval()
    with torch.no_grad():
        image_data = img_tensor.to(device)
        image_data = torch.unsqueeze(image_data, dim=0)  # 在第一维度添加维度N 因为sliding_window_inference需要 NCHW格式
        #print(f"image type:{type(image_data)}\n image shape:{image_data.shape}")
        # define sliding window size and batch size for windows inference
        roi_size = (192, 192)
        sw_batch_size = 4
        infer_output = sliding_window_inference(
            image_data, roi_size, sw_batch_size, model)
        #print(f"infer_output type:{type(infer_output)}\ninfer_output shape:{infer_output.shape}")
        infer_output = post_trans(infer_output)
        #print(f"infer_output type:{type(infer_output)}\ninfer_output shape:{infer_output.shape}")
        infer_output = infer_output.permute(0,2,3,1)
        #print(f"infer_output type:{type(infer_output)}\ninfer_output shape:{infer_output.shape}")
        saver(infer_output, meta)

#%%
if __name__ ==  "__main__":
    main()
