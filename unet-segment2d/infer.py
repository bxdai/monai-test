import argparse
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activations, AddChanneld, AsDiscrete, Compose, 
    LoadImaged,SaveImage, ScaleIntensityd, EnsureTyped,
    EnsureType,SqueezeDimd,CropForegroundd
)


#%%
def main():
    """
    main function

    """


    dcm_data = {"img": args.dcmfile}
    # define transforms for image and segmentation
    infer_transforms = Compose(
        [            
            LoadImaged(keys="img"),
            SqueezeDimd(keys="img", dim=-1),
            AddChanneld(keys="img"),
            ScaleIntensityd(keys="img"),
            CropForegroundd(keys="img", source_key="img"),
            EnsureTyped(keys="img"),
        ]
    )

    img_tensor = infer_transforms(dcm_data)
    
    #后处理Compose
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    #定义保存文件对象
    saver = SaveImage(output_dir=args.labelmap, output_ext=".nii.gz", output_postfix="segmnet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    model.load_state_dict(torch.load("best4_metric_model_segmentation2d_dict.pth"))

    model.eval()
    with torch.no_grad():
        image_data = img_tensor["img"].to(device)
        image_data = torch.unsqueeze(image_data, dim=0)  # 在第一维度添加维度N 因为sliding_window_inference需要 NCHW格式
        #print(f"image type:{type(image_data)}\n image shape:{image_data.shape}")
        # define sliding window size and batch size for windows inference
        roi_size = (192, 192)
        sw_batch_size = 4
        infer_output = sliding_window_inference(
            image_data, roi_size, sw_batch_size, model)
        #print(f"infer_output type:{type(infer_output)}\ninfer_output shape:{infer_output.shape}")
        infer_output = post_trans(infer_output)
        saver(infer_output)

#%%
if __name__ ==  "__main__":
    
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
        print("please input label filename")
        exit()
    # print("parameter dcmfile is :",args.dcmfile)
    # print("parameter labelfile is :",args.labelmap)
    main()
