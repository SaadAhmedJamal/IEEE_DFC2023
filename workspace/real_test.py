import os
import timeit
import argparse

from tqdm import tqdm
from torch.utils import data

from utils.eval_utils import *
#from utils.eval_utils_modified2 import *
#from networks.baseline import Res_Deeplab
#from networks.baseline_modified1 import Res_Deeplab
from networks.baseline_modified1_StackSkip import Res_Deeplab
#from networks.baseline_modified1_StackSkip_4 import Res_Deeplab

#from dataset.dataset_modified1 import ISPRSDataValSet
from dataset.dataset_modified1 import ISPRSDataTestSet
from utils.metrics import AverageMeter, Result
from PIL import Image

import rasterio


start = timeit.default_timer()
#IMG_MEAN = np.array((120.47595769, 81.79931481, 81.19268267), dtype=np.float32)
IMG_MEAN = np.array((120.47595769, 81.79931481, 81.19268267,0.25), dtype=np.float32)


DATA_DIRECTORY = 'workspace/dataset/data/'
#DATA_LIST_PATH = 'workspace/dataset/val_imgs.txt'
#TEST_DATA_LIST_PATH = 'workspace/dataset/val_imgs.txt'
DATA_LIST_PATH = 'workspace/dataset/test.txt'
#TEST_DATA_LIST_PATH = 'workspace/dataset/test.txt'



'''
DATA_DIRECTORY = 'workspace/dataset/data'
DATA_LIST_PATH = 'workspace/dataset/train.txt'
TEST_DATA_LIST_PATH = 'workspace/dataset/train.txt'
'''
NUM_STEPS = 17
INPUT_SIZE = '512,512'

'''
DATA_DIRECTORY = '/workspace/dataset/data/'
DATA_LIST_PATH = '/workspace/dataset/test.txt'
RESTORE_FROM = '/workspace/baseline_results/best.pth'
'''



#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\my_resnet_model.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\my_model.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_59993.pth'


#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\resnet50-imagenet.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_97.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_293.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_294.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_1469.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_4017.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_9897.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_9998.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_9999.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_5000.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_39995.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_39999.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_49994.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_4999.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_59993.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_69992.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_70999.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_78999.pth
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_79991.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_79799.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_80000.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_19997_3.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_79991_3.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_80000_3.pth'

#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_19997_5.pth'

#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_9998_6.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_19997_8.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_89990_8.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_119987.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_9998_9.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_160000_9.pth'

#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_1077_10.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_39199_10.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_80000_10.pth'

#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_47519_11.pth'

#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_4899_13.pth'

#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_16000_18.pth'
#RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_160000.pth'
RESTORE_FROM = r'C:\Users\saada\Desktop\Drive\UBS\Workshop_Contest\track2\HeightEstimation\workspace\pretrain\ISPRS_lr1e-2_93059_18.pth'


IGNORE_LABEL = 255



NUM_CLASSES = 1




def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    #parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
    #                    help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--whole", type=bool, default=False,
                        help="use whole input size.")
    return parser.parse_args()




def main():
    """Create the model and start the evaluation process."""
    average_meter = AverageMeter()
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    h, w = map(int, args.input_size.split(','))

    if args.whole:
        input_size = (2000, 2000)
    else:
        input_size = (h, w)

    model = Res_Deeplab(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    testdataset = ISPRSDataTestSet(args.data_dir, args.data_list, mean=IMG_MEAN, scale=False, mirror=False)
    testloader = data.DataLoader(testdataset,batch_size=1, shuffle=False, pin_memory=True)


    print('----------------------------------------------')
    print('Testing:')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    #Myaddition to metrics
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    for index, batch in enumerate(tqdm(testloader)):
        image, size, name = batch

        sar_images = image[:,3:4,:,:]
        #print(sar_images.shape)
        images = torch.cat((image,sar_images,sar_images),dim =1)   


        with torch.no_grad():
            if args.whole:
                print("Wrong")
                output = predict_multiscale(model, image, (image.shape[2], image.shape[3]), [0.75, 1.0, 1.25], args.num_classes, True)
            else:
                output = predict_sliding(model, image[:,:,:,:].numpy(), input_size, 1, True)    
                #output = predict_sliding(model, images[:,:,:,:].numpy(), input_size, 1, True)
        

        output = output * 183.17412
        #output = output * 18.17412

        seg_pred = output
        seg_pred = np.squeeze(seg_pred)

        #dsm = np.array(ndsm)
        #dsm = np.squeeze(dsm)
        result = Result()
        #result.evaluate(seg_pred, dsm)
        average_meter.update(result, image.size(0))

        # Creating output lines as follows by myself (By Saad):
        print(np.shape(output))
        print("Notcool  ")
        '''     
        #Image Recreation using Pillow
        myout_image = Image.fromarray(seg_pred)
        #myout_image.scale = img_scale
        #left, upper, right, lower = (0.0000000000000000,-256.0000000000000000,256.0000000000000000,0.0000000000000000)
        #myout_image = myout_image.transform((256, 256), Image.EXTENT, (left, upper, right, lower) )
        sname = str(name)[2:-2]
        myout_image.save("outputs/"+str(sname)+".tif")
        #Image.save(fp, format=None, **params)'outputs/'+
        '''


        
        # Image Recreation using Rasterio
        # Create an array with random values
        array = np.random.rand(512, 512)

        # Define the raster profile
        profile = {
            'driver': 'GTiff', 
            'height': array.shape[0], 
            'width': array.shape[1], 
            'count': 1, 
            'dtype': 'float32', 
            #'crs': 'EPSG:32650', 
            'transform': rasterio.transform.from_origin(0, 0, 0.5, 0.5),
        }
        sname = str(name)[2:-2]
        # Write the array to a raster file
        with rasterio.open("outputs/"+str(sname)+".tif", "w", **profile) as dst:
            dst.write(seg_pred, 1)








        output= seg_pred
        #target= dsm
        output[output == 0] = 0.00001
        output[output < 0] = 999
        #target[target <= 0] = 0.00001
        #valid_mask = ((target>0) + (output>0)) > 0

        #output = output[valid_mask]
        #target = target[valid_mask]
        #abs_diff = np.abs(output - target)

        #self.mse = np.mean(abs_diff ** 2)
        #mse=np.mean(abs_diff ** 2)
        #print(mse)



        # Calculate MAPE
        #mape = mean_absolute_percentage_error(output, target)
        #print("MAPE: ", mape)




    avg = average_meter.average()
    end = timeit.default_timer()


    print('----------------------------------------------')
    #print('DSM Estimate metrics:')
    print('**********************************************')
    #print('MAE={average.mae:.3f}, MSE={average.mse:.3f},    RMSE={average.rmse:.3f}\n'
    #      'Delta1={average.delta1:.3f}, Delta2={average.delta2:.3f}, Delta3={average.delta3:.3f}'.format(average=avg))
    print('**********************************************')
    # print('Model inference time:', end - start, 'seconds')
    print('----------------------------------------------')





if __name__ == '__main__':
    main()
