import argparse
import random

import clip
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torchvision import utils as vutils
from torchvision.transforms.functional import adjust_contrast

import StyleNet
import utils
from template import imagenet_templates


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# set pined seed
set_random_seed(1)

parser = argparse.ArgumentParser()

parser.add_argument('--content_path', type=str, default="./face.jpg",
                    help='Image resolution')
parser.add_argument('--content_name', type=str, default="face",
                    help='Image resolution')
parser.add_argument('--exp_name', type=str, default="exp1",
                    help='Image resolution')
parser.add_argument('--text', type=str, default="Fire",
                    help='Image resolution')
parser.add_argument('--lambda_tv', type=float, default=2e-3,
                    help='total variation loss parameter')
parser.add_argument('--lambda_patch', type=float, default=9000,
                    help='PatchCLIP loss parameter')
parser.add_argument('--lambda_dir', type=float, default=500,
                    help='directional loss parameter')
parser.add_argument('--lambda_c', type=float, default=150,
                    help='content loss parameter')
parser.add_argument('--crop_size', type=int, default=128,
                    help='cropped image size')
parser.add_argument('--num_crops', type=int, default=64,
                    help='number of patches')
parser.add_argument('--img_width', type=int, default=512,
                    help='size of images')
parser.add_argument('--img_height', type=int, default=512,
                    help='size of images')
parser.add_argument('--max_step', type=int, default=200,
                    help='Number of domains')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Number of domains')
parser.add_argument('--thresh', type=float, default=0.7,
                    help='Number of domains')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert (args.img_width % 8) == 0, "width must be multiple of 8"
assert (args.img_height % 8) == 0, "height must be multiple of 8"

# 冻结VGG【backbone】
VGG = models.vgg19(pretrained=True).features
VGG.to(device)

for parameter in VGG.parameters():
    parameter.requires_grad_(False)


def img_denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = image * std + mean
    return image


def img_normalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize(image, device):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


content_path = args.content_path
content_image = utils.load_image2(content_path, img_height=args.img_height, img_width=args.img_width)
content = args.content_name
exp = args.exp_name

content_image = content_image.to(device)

# 将图片直接送入VGG【backbone】得到特征
content_features = utils.get_features(img_normalize(content_image), VGG)
# content_image.shape = torch.Size([1, 3, 1024, 2048]) [B, C, H, W]
# conv1_1: torch.Size([1, 64, 1024, 2048])
# conv2_1: torch.Size([1, 128, 512, 1024])
# conv3_1: torch.Size([1, 256, 256, 512])
# conv4_1: torch.Size([1, 512, 128, 256])
# conv4_2: torch.Size([1, 512, 128, 256])
# conv5_1: torch.Size([1, 512, 64, 128])
# conv5_2: torch.Size([1, 512, 64, 128])


# 获得风格转换网络
style_net = StyleNet.UNet()
style_net.to(device)

content_weight = args.lambda_c

show_every = 100
optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
steps = args.max_step

content_loss_epoch = []
style_loss_epoch = []
total_loss_epoch = []

cropper = transforms.Compose([
    transforms.RandomCrop(args.crop_size)
])
augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
    transforms.Resize(224)
])
device = 'cuda'
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

prompt = args.text

source = "a Photo"

with torch.no_grad():
    template_text = compose_text_with_templates(prompt, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)
    source_features = clip_model.encode_image(clip_normalize(content_image, device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

# save content image
out_path = './outputs/' + content + '_' + exp + '_origin.jpg'
output_image = content_image.clone()
output_image = torch.clamp(output_image, 0, 1)
output_image = adjust_contrast(output_image, 1.5)
vutils.save_image(
    output_image,
    out_path,
    nrow=1,
    normalize=True)
print(f'origin saved at [{out_path}]')

num_crops = args.num_crops
for epoch in range(0, steps + 1):

    scheduler.step()
    target = style_net(content_image, use_sigmoid=True).to(device)
    # content_image.shape=torch.Size([1, 3, 1024, 2048])
    # target.shape=torch.Size([1, 3, 1024, 2048])

    # 将target转换为可学习的参数
    target.requires_grad_(True)

    target_features = utils.get_features(img_normalize(target), VGG)

    # 内容损失，相当于就是对像素进行l2损失【L_c】
    content_loss = 0
    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

    # patch的损失【L_patch】
    loss_patch = 0
    img_proc = []
    for n in range(num_crops):
        target_crop = cropper(target)
        target_crop = augment(target_crop)
        img_proc.append(target_crop)

    img_proc = torch.cat(img_proc, dim=0)
    img_aug = img_proc

    image_features = clip_model.encode_image(clip_normalize(img_aug, device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

    img_direction = (image_features - source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
    loss_temp[loss_temp < args.thresh] = 0
    loss_patch += loss_temp.mean()

    # 全局的方向损失【L_dir】
    glob_features = clip_model.encode_image(clip_normalize(target, device))
    glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

    glob_direction = (glob_features - source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

    loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

    # 正则项【L_tv】
    reg_tv = args.lambda_tv * get_image_prior_losses(target)

    total_loss = args.lambda_patch * loss_patch + content_weight * content_loss + reg_tv + args.lambda_dir * loss_glob
    total_loss_epoch.append(total_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("After %d criterions:" % epoch)
        print('Total loss: ', total_loss.item())
        print('Content loss: ', content_loss.item())
        print('patch loss: ', loss_patch.item())
        print('dir loss: ', loss_glob.item())
        print('TV loss: ', reg_tv.item())

    if epoch % 50 == 0:
        out_path = './outputs/' + prompt + '_' + content + '_' + exp + '.jpg'
        output_image = target.clone()
        output_image = torch.clamp(output_image, 0, 1)
        output_image = adjust_contrast(output_image, 1.5)
        vutils.save_image(
            output_image,
            out_path,
            nrow=1,
            normalize=True)
        print(f'saved at [{out_path}]')
