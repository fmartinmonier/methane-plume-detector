import torch
import torchmetrics
import random
import time
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import lightning.pytorch as pl
from PIL import Image

class double_conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = double_conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = double_conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class CH4UNet(pl.LightningModule):
    def __init__(self, in_channels, n_classes, base_filters, lr=1e-3, data_module=None):
        super(CH4UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.data_module = data_module
        self.base_filters = base_filters
        self.lr = lr
        self.loss_min = 100
        self.bce_min = 100
        self.iou_min = 100
        random.seed(int(time.time()))
        
        # Encoder
        self.e1 = encoder_block(self.in_channels, self.base_filters)
        self.e2 = encoder_block(self.base_filters, self.base_filters * 2)
        self.e3 = encoder_block(self.base_filters * 2, self.base_filters * 4)
        self.e4 = encoder_block(self.base_filters * 4, self.base_filters * 8)

        # Bottleneck
        self.b = double_conv_block(self.base_filters * 8, self.base_filters * 16)

        # Decoder
        self.d1 = decoder_block(self.base_filters * 16, self.base_filters * 8)
        self.d2 = decoder_block(self.base_filters * 8, self.base_filters * 4)
        self.d3 = decoder_block(self.base_filters * 4, self.base_filters * 2)
        self.d4 = decoder_block(self.base_filters * 2, self.base_filters)

        # Classifier
        self.outputs = nn.Conv2d(self.base_filters, self.n_classes, kernel_size=1, padding=0)
        
        self.save_hyperparameters()
        
    def forward(self, inputs):
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck
        b = self.b(p4)

        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Classifier
        outputs = self.outputs(d4)
        
        proba = torch.sigmoid(outputs)
        proba = proba[:, 0, :, :] # Squeezing out channel dimension
        mask = torch.where(proba > 0.5, 1, 0)
        
        #return proba
        return proba, mask
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        inputs, targets, pixel_mask = batch['image'], batch['plume_mask'], batch["pixel_mask"]
        proba, mask = self(inputs)

        targets = targets * pixel_mask
        proba = proba * pixel_mask
        mask = mask * pixel_mask
        
        loss, binary_ce, iou = self._jack_loss(proba, mask, targets)
        self.log('train_loss', loss)
        self.log('train_loss_binary_ce', binary_ce)
        self.log('train_loss_iou', 1-iou)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets, pixel_mask = batch['image'], batch['plume_mask'], batch["pixel_mask"]
        proba, mask = self(inputs)

        targets = targets * pixel_mask
        proba = proba * pixel_mask
        mask = mask * pixel_mask
        
        loss, binary_ce, iou = self._jack_loss(proba, mask, targets)
        self.log('validation_loss', loss)
        self.log('validation_loss_binary_ce', binary_ce)
        self.log('validation_loss_iou', 1-iou)
        
        #self.log_summary_metrics(loss, binary_ce, iou)
        
    def on_validation_epoch_end(self):
        val_dataloader = self.data_module.val_dataloader()
        random_batch_idx = random.randint(0, len(val_dataloader) - 1)

        # Retrieve the random batch
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx == random_batch_idx:
                inputs = batch["image"].to(self.device)
                targets = batch["plume_mask"].to(self.device)
                pixel_mask = batch["pixel_mask"].to(self.device)
                source_rate = batch["source_rate"][0]
                scene_id = batch["scene_id"][0]
                break
                
        proba, mask = self(inputs)
        targets = targets * pixel_mask
        proba = proba * pixel_mask
        mask = mask * pixel_mask

        gt_mask = targets[0, :, :].squeeze().detach().cpu().numpy().astype(np.uint8) * 255
        pred_mask = mask[0, :, :].squeeze().detach().cpu().numpy().astype(np.uint8) * 255
        pred_proba = proba[0, :, :].detach().squeeze().cpu().numpy() * 255
        pred_proba = pred_proba.astype(np.uint8)
        image = inputs[0, 0, :, :].detach().cpu().numpy() # this should show the ndmi channel

        self.visualize_prediction(image, gt_mask, pred_mask, pred_proba, source_rate, scene_id)

    def _jack_loss(self, proba, mask, targets):
        binary_ce = F.binary_cross_entropy(proba, targets)
        iou = self.jaccard_index(mask, targets)
        #dice = self.dice_coef(mask, targets)

        loss = binary_ce - torch.log(iou)
        #loss = binary_ce - torch.log(dice)
        return loss, binary_ce, iou
    
    def jaccard_index(self, outputs, targets):
        smooth = 1
        targets_f = torch.flatten(targets)
        outputs_f = torch.flatten(outputs)
        intersection = torch.sum(targets_f * outputs_f)
        return (intersection + smooth) / (torch.sum(targets_f) + torch.sum(outputs_f) - intersection + smooth)
    
    def dice_coef(self, outputs, targets):
        smooth = 1
        targets_f = torch.flatten(targets)
        outputs_f = torch.flatten(outputs)
        intersection = torch.sum(targets_f * outputs_f)
        return (2 * intersection + smooth) / (torch.sum(targets_f) + torch.sum(outputs_f) + smooth)


    def visualize_prediction(self, image, ground_truth, pred_mask, pred_proba, source_rate, scene_id):
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        scaled_image = (normalized_image * 255).astype(np.uint8)
        image = Image.fromarray(scaled_image, mode='L')
        ground_truth = TF.to_pil_image(ground_truth, mode='L')
        pred_mask = TF.to_pil_image(pred_mask, mode='L')
        pred_proba = TF.to_pil_image(pred_proba, mode='L')

        self.logger.log_image(key="Visualization", images=[image, ground_truth, pred_mask, pred_proba], caption=[f"NDMI - {scene_id}", f"Ground Truth Mask - {int(source_rate)} t/h", f"Predicted Mask", "Probabilities"])
        
    def log_summary_metrics(self, loss, binary_ce, iou):
        if (loss < self.loss_min):
            wandb.run.summary["val_loss"] = loss
            self.loss_min = loss
        if (binary_ce < self.bce_min):
            wandb.run.summary["val_bce"] = binary_ce
            self.bce_min = loss
        if (iou < self.iou_min):
            wandb.run.summary["val_iou"] = iou
            self.iou_min = iou