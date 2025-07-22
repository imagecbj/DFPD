import config
import networks.DWT.Unet_common as common
from networks.HiNetmodel import *
from networks.fs_model_256 import fsModel
from networks.network_fs.AEI_Net import AEI_Net
from InsightFace_Pytorch.model import Backbone
from networks.StarGAN import Generator as StarG
from networks.fgan import Generator as Fgan
import torchvision.utils as vutils

class Models(nn.Module):
    def __init__(self, device, opts):
        super(Models, self).__init__()
        self.dwt = common.DWT()
        self.iwt = common.IWT()
        ## HiNet
        self.net = Model()
        self.net = torch.nn.DataParallel(self.net)
        self.net.to(device)
        init_model(self.net)
        model_path = config.Hinet_path  # 原始pth
        self.load(model_path)
        self.net.eval()

        ###  attack model prepare ###
        ## 1.simswap
        self.simswap_net = fsModel(perturb_wt=config.perturb_wt, attack_loss_type=config.loss_type)
        self.simswap_net.eval().to(device)
        ## 2.faceswap
        self.G = AEI_Net(c_id=512)
        self.G.eval()
        self.G.load_state_dict(torch.load(config.fs_path,
                                     map_location=torch.device('cpu')))
        self.G = self.G.to(device)

        self.arcface = Backbone(50, 0.6, 'ir_se').to(device)
        self.arcface.eval()
        self.arcface.load_state_dict(
            torch.load(config.arcface_path,
                       map_location=device), strict=False)
        ## 3.stargan
        self.stargan = StarG(conv_dim=config.conv_dim, c_dim=config.c_dim, repeat_num=config.repeat_num).to(config.device)
        ckpt = torch.load(config.starganPath, map_location=lambda storage, loc: storage)
        self.stargan.load_state_dict(ckpt)

        ## 4.fgan
        self.fgan = Fgan(64, 5, 6).to(device)
        self.fgan.load_state_dict(torch.load(config.fgan_path, map_location=lambda storage, loc: storage))

        self.selected_attrs = opts.selected_attrs

    def load(self, name):
        state_dicts = torch.load(name)
        network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
        self.net.load_state_dict(network_state_dict)

    def create_labels(self,c_org):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.

        hair_color_indices = []
        for i, attr_name in enumerate(self.selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(config.c_dim):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(config.device))
        return c_trg_list

    def model_out(self, x, c_trg_list):
        outs = [x.data]
        with torch.no_grad():
            for i, c_trg in enumerate(c_trg_list):
                gen, _ = self.stargan(x, c_trg)
                outs.append(gen.data)
        return outs

    def model_fgan_out(self, x, c_trg_list):
        outs = [x.data]
        with torch.no_grad():
            for i, c_trg in enumerate(c_trg_list):
                gen = torch.tanh(x + self.fgan(x, c_trg)[0])
                # gen, _ = stargan(x, c_trg)
                outs.append(gen.data)
        return outs

    def save_grid_img(self, imgs, save_root_path, epoch):
        out_file = save_root_path + '/' + str(epoch) + '_result.png'
        x_concat = torch.cat(imgs, dim=-2)
        vutils.save_image(x_concat.data, out_file, nrow=14, normalize=True, range=(-1., 1.))

    def tensor2im(self, var, fs=False):
        var = var.permute(1, 2, 0).cpu().detach().numpy()
        if not fs:
            var = ((var + 1) / 2)
        var[var < 0] = 0
        var[var > 1] = 1
        var = var * 255
        return var