import sys
import os
import torch
import yaml

sys.path.append('../../../../')
from models.KGFSL import FRN
from utils import util
from trainers.eval import meta_test
from statistics import mean

with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path, 'CUB_fewshot_cropped/test')
# model_path = './model_ResNet-12.pth'
model_path = 'F:\FRN-main\experiments\CUB_fewshot_cropped\FRN\ResNet-12\model_ResNet-12.pth'

gpu = 0
torch.cuda.set_device(gpu)

model = FRN(resnet=True)
model.cuda()
checkpoint = torch.load(model_path, map_location=util.get_device_map(gpu))
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
# model.load_state_dict(checkpoint, strict=True)
model.eval()
with torch.no_grad():
    way = 5
    for shot in [1, 5]:
        mean1, interval1, mean2, interval2 = meta_test(data_path=test_path,
                                                       model=model,
                                                       way=way,
                                                       shot=shot,
                                                       pre=False,
                                                       transform_type=0,
                                                       trial=2000)
        # print('%d-way-%d-shot acc: %.3f\t%.3f' % (way, shot, mean, interval))
        print('%d-way-%d-shot acc1: %.3f\t%.3f' % (way, shot, mean1, interval1))
        print('%d-way-%d-shot acc1: %.3f\t%.3f' % (way, shot, mean2, interval2))
# with torch.no_grad():

