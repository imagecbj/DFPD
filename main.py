import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils
from argparse import ArgumentParser
from color_space import rgb2ycbcr_np, ycbcr_to_tensor, ycbcr_to_rgb
from logger import setup_logger
from metrics_compute import compute_metrics1, prepare_lpips
from data import CelebA
import torch.utils.data as data
from networks.models import Models
import torch
import config as c
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    config_parser = ArgumentParser()
    config_parser.add_argument('--iteration', default=20, type=int, help='Iteration Nums')
    config_parser.add_argument('--epsilon', default=0.02, type=float, help='Perturbation Thresholds')
    config_parser.add_argument('--step_size', default=0.002, type=float, help='Step Size')
    config_parser.add_argument('--logger_path', default='./results', type=str, help='Logger Path')
    config_parser.add_argument('--batch_size', default=1, type=int, help='Dataset Batch Size')
    config_parser.add_argument('--iter', default=10, type=int, help='Iteration')
    config_parser.add_argument('--lr', default=0.0001, type=float, help='Learning Rate')
    config_parser.add_argument('--dataset_path', default='/home/lab/workspace/works/hyt/img_test_256/',
                               type=str, help='Dataset Path')
    config_parser.add_argument('--attribute_txt_path',
                               default='/home/lab/workspace/works/hyt/attribute_crop.txt',
                               type=str, help='Attribute Txt Path')
    config_parser.add_argument('--selected_attrs', default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
                               type=list, help='Attribute Selection')
    config_parser.add_argument('--img_size', default=256, type=float, help='Image Size')
    opts = config_parser.parse_args()

    torch.nn.Module.dump_patches = True

    test_dataset = CelebA(opts.dataset_path, opts.attribute_txt_path,
                          opts.img_size, 'train', c.attrs,
                          opts.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=opts.batch_size, num_workers=4,
        shuffle=False
    )
    
    print("prepared datasets:"+str(len(test_dataloader)))

    ################## logger setting ###################
    logger = setup_logger(opts.logger_path, 'result.log', 'result_logger')
    logger.info(f'Loading model.')
    models = Models(device, opts)

    lpips_model, lpips_model2 = prepare_lpips()
    criterion = torch.nn.MSELoss()

    pil2tensor_transform = transforms.Compose([transforms.ToTensor()])
    tensor2pil_transform = transforms.ToPILImage()
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    ############## Forward Pass ######################
    n_dist_stargan, n_dist_fgan, n_dist_simswap, n_dist_fs = 0.0, 0.0, 0.0, 0.0
    psnr_adv_cln, psnr_adv_stego = 0.0, 0.0
    psnr_value, ssim_value, lpips_alexs, lpips_vggs = 0.0, 0.0, 0.0, 0.0
    psnr_value_fgan, ssim_value_fgan, lpips_alexs_fgan, lpips_vggs_fgan = 0.0, 0.0, 0.0, 0.0
    succ_num, total_num, n_dist = 0.0, 0.0, 0.0
    succ_num_attgan = 0.0
    l1_error, l2_error = 0.0, 0.0
    l1_error_fgan = 0.0
    l1_star, l2_star = 0.0, 0.0
    l1_hifi, l1_sw, l1_fs, l2_hifi, l2_sw, l2_fs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    vgg_sum = 0.0
    vgg_sum_attgan = 0.0
    vgg_sum_hifi, vgg_sum_sw, vgg_sum_fs = 0.0,0.0,0.0
    id_sum = 0.0
    id_hififace, id_simswap, id_fs = 0.0, 0.0, 0.0
    ssim_id_hifi, ssim_id_simswap, ssim_id_fs = 0.0, 0.0, 0.0
    psnr_adv, ssim_adv, lpips_alexs_adv, lpips_vggs_adv = 0.0, 0.0, 0.0, 0.0
    psnr_hifi, psnr_sw, psnr_fs, ssim_hifi, ssim_sw, ssim_fs = 0.0, 0.0, 0.0, 0.0, 0.0,0.0
    lpips_alexs_hifi, lpips_alexs_sw, lpips_alexs_fs, lpips_vggs_hifi, lpips_vggs_sw, lpips_vggs_fs = 0.0, 0.0, 0.0, 0.0, 0.0,0.0
    for ii, (img_a, att_a, c_org, filename) in enumerate(tqdm(test_dataloader, desc='')):
        c_trg_list = models.create_labels(c_org)

        pic_a = c.dataset_path + str(ii+1) + '.png'
        # The person who provides id information
        target_img = Image.open(pic_a)
        ###### simswap transformer
        target_simswap_img = transformer_Arcface(target_img)
        target_simswap_id = target_simswap_img.view(-1, target_simswap_img.shape[0], target_simswap_img.shape[1], target_simswap_img.shape[2]).to(device)

        target_fs_img = test_transform(target_img).unsqueeze(0).cuda()
        target_img = pil2tensor_transform(target_img).unsqueeze(0).to(device)
        id_ori_fs = models.arcface(
            F.interpolate(target_fs_img[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))

        ###################### watermark embedding #####################
        pic_a = c.dataset_path + str(ii) + '.png'
        cover_img = Image.open(pic_a)
        cln_img = cover_img
        secrect_img = Image.fromarray(np.uint8(np.zeros((np.array(cover_img)).shape)))

        cover_img = transforms.ToTensor()(cover_img).unsqueeze(0).to(device)
        secrect_img = transforms.ToTensor()(secrect_img).unsqueeze(0).to(device)

        cover_input = models.dwt(cover_img)
        secret_input = models.dwt(secrect_img)
        input_img = torch.cat((cover_input, secret_input), 1)

        output = models.net(input_img)
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        steg_img = models.iwt(output_steg)  # 【0-1】
        source_img = Image.fromarray(steg_img[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())

        img_source_fs = test_transform(source_img).unsqueeze(0).to(device)

        s_img = np.array(source_img)  # 512
        source_img_ori = pil2tensor_transform(source_img).unsqueeze(0).to(device)

        b_ori_img = pil2tensor_transform(cln_img)
        b_ori_img = b_ori_img.unsqueeze(0).cuda()

        ## simswap数据注入
        models.simswap_net.real_A1 = source_img_ori.clone()
        models.simswap_net.real_A2 = target_simswap_id.clone()  # A2 加扰动

        with torch.no_grad():
            old_simswap = models.simswap_net.swap_face_single2(models.simswap_net.real_A1, models.simswap_net.real_A2)
            old_fs = models.G(img_source_fs, id_ori_fs)[0] / 2 + 0.5
            ori_outs = models.model_out(img_source_fs, c_trg_list)
            ori_outs_fgan = models.model_fgan_out(img_source_fs, c_trg_list)

        ###########################define Y_mask#######################################
        mask_y = np.ones(source_img_ori.detach().clone().cpu().size())
        mask_y[:, 0, :, :] = 0
        mask_y = torch.Tensor(mask_y).cuda()
        mask_y.requires_grad = False
        #####################RGB to YCbCr################################
        x_y = rgb2ycbcr_np(s_img.astype('float'))  # [224,224,3]
        x_y = ycbcr_to_tensor(x_y).to(device)
        z1 = x_y
        z1.requires_grad = False

        pbar = tqdm(range(1, opts.iteration + 1), leave=True)
        X = z1.clone().detach_() + torch.tensor(
            np.random.uniform(-opts.epsilon, opts.epsilon, source_img_ori.shape).astype('float32')).to(device)
        for step in pbar:
            loss = 0.0
            X.requires_grad = True
            G_normgrad = torch.zeros_like(X)
            adv_A1 = ycbcr_to_rgb(X)  # tensor类型,(224,224,3) #BGR int

            ## simswap
            adv_out_simswap = models.simswap_net.swap_face_single2(adv_A1, models.simswap_net.real_A2)
            grad_simswap = \
                torch.autograd.grad(outputs=criterion(adv_out_simswap,old_simswap), inputs=X, create_graph=True,
                                    retain_graph=True)[0]
            grad_normalized = grad_simswap / grad_simswap.norm(2)
            G_normgrad += (2*grad_normalized)

            ## fs
            adv_A2 = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(adv_A1.contiguous())
            adv_out_fs, _ = models.G(adv_A2, id_ori_fs)
            grad_fs = \
                torch.autograd.grad(outputs=criterion(adv_out_fs,old_fs), inputs=X, create_graph=True,
                                    retain_graph=True)[0]
            grad_normalized = grad_fs / grad_fs.norm(2)
            G_normgrad += grad_normalized

            ## stargan
            loss_stargan = 0.0
            x_adv = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(adv_A1.contiguous())
            for i, c_trg in enumerate(c_trg_list):
                gen, _ = models.stargan(img_source_fs, c_trg)
                gen_adv, _ = models.stargan(x_adv, c_trg)
                loss_stargan += criterion(gen, gen_adv)
            grad_stargan = \
                torch.autograd.grad(outputs=loss_stargan, inputs=X, create_graph=True,
                                    retain_graph=True)[0]
            grad_normalized = grad_stargan / grad_stargan.norm(2)
            G_normgrad += grad_normalized

            # fgan
            loss_fgan = 0.0
            for i, c_trg in enumerate(c_trg_list):
                gen = torch.tanh(img_source_fs + models.fgan(img_source_fs, c_trg)[0])
                gen_adv = torch.tanh(x_adv + models.fgan(x_adv, c_trg)[0])
                loss_fgan += criterion(gen, gen_adv)
            grad_fgan = \
                torch.autograd.grad(outputs=loss_fgan, inputs=X, create_graph=True,
                                    retain_graph=True)[0]
            grad_normalized = grad_fgan / grad_fgan.norm(2)
            G_normgrad += grad_normalized

            X_adv = X + opts.step_size * torch.sign(G_normgrad)
            eta1 = torch.clamp(X_adv - z1, min=-opts.epsilon, max=opts.epsilon) * mask_y
            X = (z1 + eta1).detach_()
        pertubation = X - z1
        x_adv = ycbcr_to_rgb(z1 + pertubation)

        # compute psnr between cln imgs and defensed imgs
        psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics1(b_ori_img, x_adv, lpips_model, lpips_model2)
        psnr_adv += psnr_temp
        ssim_adv += ssim_temp
        lpips_alexs_adv += lpips_alex
        lpips_vggs_adv += lpips_vgg
        log_message = f', psnr_adv: {psnr_temp:.3f}'
        log_message += f', ssim_adv: {ssim_temp:.4f}'
        log_message += f', lpips_adv: {(lpips_alex+lpips_vgg)/2:.4f}'

        with torch.no_grad(): ## adv outs
            x_adv_norm = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())
            adv_out_simswap = models.simswap_net.swap_face_single2(x_adv, models.simswap_net.real_A2)
            adv_out_fs = models.G(x_adv_norm, id_ori_fs)[0] / 2 + 0.5
            adv_outs = models.model_out(x_adv_norm,c_trg_list)
            adv_outs_fgan = models.model_fgan_out(x_adv_norm,c_trg_list)

        ## simswap
        psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics1(old_simswap, adv_out_simswap, lpips_model,
                                                                       lpips_model2)
        l1 = torch.nn.functional.l1_loss(old_simswap, adv_out_simswap).item()
        l1_sw += l1
        psnr_sw += psnr_temp
        lpips_alexs_sw += lpips_alex
        lpips_vggs_sw += lpips_vgg

        log_message += f', psnr_simswap: {psnr_temp:.4f}'
        log_message += f', lpips_simswap: {(lpips_alex+lpips_vgg)/2:.4f}'
        log_message += f', l1_simswap: {l1:.4f}'


        ## faceshifter
        psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics1(old_fs, adv_out_fs, lpips_model,
                                                                       lpips_model2)
        l1 = torch.nn.functional.l1_loss(old_fs, adv_out_fs).item()
        l1_fs += l1
        psnr_fs += psnr_temp
        lpips_alexs_fs += lpips_alex
        lpips_vggs_fs += lpips_vgg

        log_message += f', psnr_fs: {psnr_temp:.4f}'
        log_message += f', lpips_fs: {(lpips_alex+lpips_vgg)/2:.4f}'
        log_message += f', l1_fs: {l1:.4f}'

        ## stargan
        temp_psnr, temp_lpips, temp_l1 = 0.0, 0.0, 0.0
        for i in range(len(adv_outs) - 1):
            psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics1(ori_outs[i + 1], adv_outs[i + 1], lpips_model,
                                                                           lpips_model2)
            temp_psnr += psnr_temp
            temp_lpips += ((lpips_alex+lpips_vgg)/2)
            l1 = torch.nn.functional.l1_loss(ori_outs[i + 1], adv_outs[i + 1]).item()
            l1_error += l1
            temp_l1 += l1
            l2_error += torch.nn.functional.mse_loss(ori_outs[i + 1], adv_outs[i + 1]).item()
            psnr_value += psnr_temp
            lpips_alexs += lpips_alex
            lpips_vggs += lpips_vgg

        log_message += f', psnr_stargan: {temp_psnr/5:.4f}'
        log_message += f', lpips_stargan: {temp_lpips/5:.4f}'
        log_message += f', l1_stargan: {temp_l1/5:.4f}'

        # fgan
        temp_psnr, temp_lpips, temp_l1 = 0.0, 0.0, 0.0
        for i in range(len(adv_outs_fgan) - 1):
            psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics1(ori_outs_fgan[i + 1], adv_outs_fgan[i + 1],
                                                                           lpips_model,
                                                                           lpips_model2)
            temp_psnr += psnr_temp
            temp_lpips += ((lpips_alex+lpips_vgg)/2)
            l1 = torch.nn.functional.l1_loss(ori_outs_fgan[i + 1], adv_outs_fgan[i + 1]).item()
            l1_error_fgan += l1
            temp_l1 += l1
            psnr_value_fgan += psnr_temp
            lpips_alexs_fgan += lpips_alex
            lpips_vggs_fgan += lpips_vgg

        log_message += f', psnr_fgan: {temp_psnr/5:.4f}'
        log_message += f', lpips_fgan: {temp_lpips/5:.4f}'
        log_message += f', l1_fgan: {temp_l1/5:.4f}'

        logger.debug(f'Step: {ii:05d}, '
                          f'{log_message}')
        total_num = total_num + 1

        result_imgs = []
        images1 = torch.cat((b_ori_img, x_adv, x_adv - b_ori_img, target_img), -1)
        images2 = torch.cat((adv_out_simswap, old_simswap, adv_out_fs, old_fs), -1)
        horizontal_grid1 = models.tensor2im(vutils.make_grid(images1), fs=True)
        horizontal_grid2 = models.tensor2im(vutils.make_grid(images2), fs=True)
        horizontal_grid1 = cv2.cvtColor(horizontal_grid1, cv2.COLOR_BGR2RGB)
        horizontal_grid2 = cv2.cvtColor(horizontal_grid2, cv2.COLOR_BGR2RGB)

        results = []
        results.append(torch.cat(ori_outs, dim=0))
        results.append(torch.cat(adv_outs, dim=0))
        results.append(torch.cat(ori_outs_fgan, dim=0))
        results.append(torch.cat(adv_outs_fgan, dim=0))
        models.save_grid_img(results, opts.logger_path, ii)

        cv2.imwrite(os.path.join(opts.logger_path, f"{ii}_train.png"), horizontal_grid1)
        cv2.imwrite(os.path.join(opts.logger_path, f"{ii}_swap.png"), horizontal_grid2)
        vutils.save_image(x_adv, opts.logger_path + "/{}_adv.png".format(ii))


    psnr_adv /= total_num
    ssim_adv /= total_num
    lpips_alexs_adv /= total_num
    lpips_vggs_adv /= total_num

    ## simswap:
    psnr_sw /= total_num
    lpips_alexs_sw /= total_num
    lpips_vggs_sw /= total_num
    l1_sw /= total_num

    ## faceshifter
    psnr_fs /= total_num
    lpips_alexs_fs /= total_num
    lpips_vggs_fs /= total_num
    l1_fs /= total_num

    ## stargan
    len_crg = 5
    psnr_value /= total_num * len_crg
    lpips_alexs /= total_num * len_crg
    lpips_vggs /= total_num * len_crg
    l1_error /= total_num * len_crg

    ## attgan
    psnr_value_fgan /= total_num * len_crg
    lpips_alexs_fgan /= total_num * len_crg
    lpips_vggs_fgan /= total_num * len_crg
    l1_error_fgan /= total_num * len_crg


    log_message = "\nThe Average Metrics between clean and defensed images:\n"
    log_message += f'psnr_adv: {psnr_adv:.3f}'
    log_message += f', ssim_adv: {ssim_adv:.4f}'
    log_message += f', lpips(alex)_adv: {lpips_alexs_adv:.5f}'
    log_message += f', lpips(vgg)_adv: {lpips_vggs_adv:.5f}'

    log_message += "\nThe Average Metrics between clean outputs and defensed outputs (simswap):\n"
    log_message += f'psnr: {psnr_sw:.3f}'
    log_message += f', lpips(alex): {lpips_alexs_sw:.5f}'
    log_message += f', lpips(vgg): {lpips_vggs_sw:.5f}'
    log_message += f', l1_error: {l1_sw:.5f}'
    log_message += f', vgg: {vgg_sum_sw:.5f}'

    log_message += "\nThe Average Metrics between clean outputs and defensed outputs (fs):\n"
    log_message += f'psnr: {psnr_fs:.3f}'
    log_message += f', lpips(alex): {lpips_alexs_fs:.5f}'
    log_message += f', lpips(vgg): {lpips_vggs_fs:.5f}'
    log_message += f', l1_error: {l1_fs:.5f}'

    log_message += "\nThe Average Metrics between clean outputs and defensed outputs (stargan):\n"
    log_message += f'psnr: {psnr_value:.3f}'
    log_message += f', lpips(alex): {lpips_alexs:.5f}'
    log_message += f', lpips(vgg): {lpips_vggs:.5f}'
    log_message += f', l1_error: {l1_error:.5f}'

    log_message += "\nThe Average Metrics between clean outputs and defensed outputs (fgan):\n"
    log_message += f'psnr: {psnr_value_fgan:.3f}'
    log_message += f', lpips(alex): {lpips_alexs_fgan:.5f}'
    log_message += f', lpips(vgg): {lpips_vggs_fgan:.5f}'
    log_message += f', l1_error: {l1_error_fgan:.5f}'
    logger.debug(f'{log_message}')

    print(log_message)


