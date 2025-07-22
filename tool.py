import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out
def create_labels(device, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        elif dataset == 'RaFD':
            c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

        c_trg_list.append(c_trg.to(device))
    return c_trg_list
def get_lpips_models(device):
    lpips_model = lpips.LPIPS(net="alex").to(device)
    # loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    lpips_model2 = lpips.LPIPS(net='vgg').to(device)
    return lpips_model, lpips_model2
def compute_metrics2(x_ori, x_adv, lpips_model, lpips_model2):
    img_ori = (x_ori[0] / 2 + 0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img_adv = (x_adv[0] / 2 + 0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    psnr_value = psnr(img_ori, img_adv)
    ssim_value = ssim(img_ori, img_adv, channel_axis = 2)
    lpips_alex = lpips_model(x_ori, x_adv).item()
    lpips_vgg = lpips_model2(x_ori, x_adv).item()
    return psnr_value, ssim_value, lpips_alex, lpips_vgg
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G
def get_perceptual_loss(vgg, input, adv):
    # get vgg features
    x_features = vgg(input)
    adv_features = vgg(adv)
    # calculate style loss
    loss_mse = torch.nn.MSELoss()
    x_gram = [gram(fmap) for fmap in x_features]
    adv_gram = [gram(fmap) for fmap in adv_features]
    style_loss = 0.0
    for j in range(4):
        style_loss += loss_mse(x_gram[j], adv_gram[j])
    style_loss = style_loss

    # calculate content loss (h_relu_2_2)
    xcon = x_features[1]
    acon = adv_features[1]
    content_loss = loss_mse(xcon, acon)
    return style_loss, content_loss