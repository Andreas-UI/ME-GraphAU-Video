from .utils import *
from .conf import get_config,set_logger,set_outdir,set_env
import pkg_resources
from MEGraphAU.download_checkpoints import download_checkpoints
import os

conf = get_config()
conf.evaluate = True
set_env(conf)
set_outdir(conf)
set_logger(conf)

def predict(img, stage=2, arc="resnet50", resume=pkg_resources.resource_filename("MEGraphAU", "OpenGraphAU/checkpoints/OpenGprahAU-ResNet50_second_stage.pth")):
    if "resnet50-19c8e357.pth" not in os.listdir(pkg_resources.resource_filename("MEGraphAU", "OpenGraphAU/checkpoints")) or "OpenGprahAU-ResNet50_second_stage.pth" not in os.listdir(pkg_resources.resource_filename("MEGraphAU", "OpenGraphAU/checkpoints")) :
        print("Checkpoints are not downloaded")
        download_checkpoints()

    dataset_info = hybrid_prediction_infolist

    if stage == 1:
        from .model.ANFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
    else:
        from .model.MEFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=arc)
    
    # resume
    if resume != '':
        net = load_state_dict(net, resume)


    net.eval()
    img_transform = image_eval()
    img_ = img_transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        net = net.cuda()
        img_ = img_.cuda()

    with torch.no_grad():
        pred = net(img_)
        pred = pred.squeeze().cpu().numpy()

    infostr_probs,  infostr_aus = dataset_info(pred, 0.5)

    return infostr_aus, pred