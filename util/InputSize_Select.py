import torchvision
from torchvision import transforms
import torch.nn.functional as F

def Transform_Select(args):
    if args.model_select =='VGGFace2':
        Input = 224
        transform = transforms.Compose([torchvision.transforms.Resize(Input),
                                        transforms.ToTensor()])
                                        # transforms.Normalize((0.5, 0.5, 0.5),
                                        #                      (0.5, 0.5, 0.5))])
    elif args.model_select == 'Light_CNN_9' or args.model_select == 'Light_CNN_29' or args.model_select == 'Light_CNN_29_v2':
        Input = 128
        transform = transforms.Compose([torchvision.transforms.Resize(Input),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
                                        # transforms.Normalize([0.5],[0.5])
    elif args.model_select == 'IR-50':
        Input = 112
        transform = transforms.Compose([torchvision.transforms.Resize(Input),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])
    return transform

def TrainingSize_Select(img, args):
    # Input range [-1, 1]

    if args.model_select =='VGGFace2':
        # Input range [0, 255]
        Input = 224
        transform = F.interpolate((img + 1) * 127.5, size=Input)

    elif args.model_select == 'Light_CNN_9' or args.model_select == 'Light_CNN_29' or args.model_select == 'Light_CNN_29_v2':
        # Input range [0, 1]
        Input = 128
        transform = F.interpolate((img * 0.5) + 0.5, size=Input)

    elif args.model_select == 'IR-50':
        # Input range [-1, 1]
        Input = 112
        transform = F.interpolate(img, size=Input)

    return transform

def TestingSize_Select(input, args):

    # [0, 255]
    if args.model_select =='VGGFace2':
        input = F.interpolate(input, 224, mode='bilinear', align_corners=False)
        input = (input + 1) / 2 * 255
        input = input[:, 0, :, :].unsqueeze(1)

    # [0, 1]
    elif args.model_select == 'Light_CNN_9' or args.model_select == 'Light_CNN_29' or args.model_select == 'Light_CNN_29_v2':
        input = F.interpolate(input, 128, mode='bilinear', align_corners=False)
        input = (input + 1) / 2
        input = input[:, 0, :, :].unsqueeze(1)

    # [-1, 1]
    elif args.model_select == 'IR-50':
        input = F.interpolate(input, 112, mode='bilinear', align_corners=False)
        input = input

    else:
        print('Please select valid pretrained model !')
        exit()

    return input
