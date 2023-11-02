import torch
import torch.nn as nn
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl
from functools import partial

class classificationModel(pl.LightningModule):
    def __init__(self,  
                 number_classes, 
                 loss, 
                 lr, 
                 model):
        
        super(classificationModel, self).__init__()
        
        self.model = model

        self.n_outputs = number_classes

        self.loss = loss
        self.lr = lr
        
        # Evaluation metrics
        if self.n_outputs == 1:
            self.evaluation_metric_train = torchmetrics.AUROC(task='binary')
            self.evaluation_metric_val = torchmetrics.AUROC(task='binary')

    def forward(self, x):

        x = self.model(x)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr) 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        if self.n_outputs == 1:
            loss = self.loss(y_hat[:,0], y[:,0])

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.evaluation_metric_train(y_hat[:,0], y[:,0])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        
        if self.n_outputs == 1:
            loss = self.loss(y_hat[:,0], y[:,0])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.evaluation_metric_val(y_hat[:,0], y[:,0])

    def on_training_epoch_end(self):
        auc = self.evaluation_metric_train.compute()
        self.log('train_auc', auc)
        self.log('step', self.current_epoch)
        self.evaluation_metric_train.reset()


    def on_validation_epoch_end(self):
        auc = self.evaluation_metric_val.compute()
        self.log('val_auc', auc)
        self.log('step', self.current_epoch)
        self.evaluation_metric_val.reset()


""" UTIL FUNCTIONS """
class Attention_block(nn.Module):
    def __init__(self, F_g, F_1, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_1, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


""" MODEL CLASSES """
class ResNet(pl.LightningModule):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):

        super(ResNet, self).__init__()


        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.in_planes_Vasc = block_inplanes[0]
        self.no_max_pool = no_max_pool


        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear( block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 

        x = self.conv1(xOriginal) 
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x) 

        x = self.layer1(x) 
        x = self.layer2(x) 

        x = self.layer3(x) 
        x = self.layer4(x) 
 
        # Average Pooling
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 

        return x


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        if new_layer:
            self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)


    def _make_layer_Vasc(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes_Vasc != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes_Vasc, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes_Vasc,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        
        self.in_planes_Vasc = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes_Vasc, planes))

        return nn.Sequential(*layers)

class ResNet_multichannel(pl.LightningModule):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):

        super(ResNet_multichannel, self).__init__()


        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.in_planes_Vasc = block_inplanes[0]
        self.no_max_pool = no_max_pool


        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear( block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xOriginal = x[:,0,:,...] 

        x = self.conv1(xOriginal) 
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x) 

        x = self.layer1(x) 
        x = self.layer2(x)

        x = self.layer3(x) 
        x = self.layer4(x) 
 
        # Average Pooling
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 

        return x

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        if new_layer:
            self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)


    def _make_layer_Vasc(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes_Vasc != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes_Vasc, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes_Vasc,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        
        self.in_planes_Vasc = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes_Vasc, planes))

        return nn.Sequential(*layers)

class ResNet_attention_strategy_1(pl.LightningModule):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):

        super(ResNet_attention_strategy_1, self).__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.in_planes_Vasc = block_inplanes[0]
        self.no_max_pool = no_max_pool


        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)



        self.conv1_Vasc = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1_Vasc = nn.BatchNorm3d(self.in_planes)
        self.relu_Vasc = nn.ReLU(inplace=True)
        self.maxpool_Vasc = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        

        self.layer1_Vasc = self._make_layer_Vasc(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2_Vasc = self._make_layer_Vasc(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3_Vasc = self._make_layer_Vasc(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)               
        self.layer4_Vasc = self._make_layer_Vasc(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        
        self.AttFinal = Attention_block(F_g=512, F_1=512, F_int=256)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear( block_inplanes[3] * block.expansion * 2, n_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 
        xOriginalVasc = torch.unsqueeze(x[:,0,1,...],1) 

        """ BRAIN CTA """
        x = self.conv1(xOriginal)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x) 

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
 
        """ BRAIN VASCULATURE """
        xVasc = xOriginalVasc
        xVasc = self.conv1_Vasc(xOriginal) 
        xVasc = self.bn1_Vasc(xVasc)
        xVasc = self.relu_Vasc(xVasc)
        if not self.no_max_pool:
            xVasc = self.maxpool_Vasc(xVasc) 

        xVasc = self.layer1_Vasc(xVasc) 
        xVasc = self.layer2_Vasc(xVasc)

        xVasc = self.layer3_Vasc(xVasc)
        xVasc = self.layer4_Vasc(xVasc) 
 

        x_all = self.AttFinal(g=x, x=xVasc)
        x_all = torch.cat((x, x_all), dim=1)

        # Average Pooling
        x = self.avgpool(x_all)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        return x

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        if new_layer:
            self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _make_layer_Vasc(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes_Vasc != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes_Vasc, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes_Vasc,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        
        self.in_planes_Vasc = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes_Vasc, planes))

        return nn.Sequential(*layers)

class ResNet_attention_strategy_2(pl.LightningModule):
    
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):

        super(ResNet_attention_strategy_2, self).__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.in_planes_Vasc = block_inplanes[0]
        self.no_max_pool = no_max_pool


        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)


        self.conv1_Vasc = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1_Vasc = nn.BatchNorm3d(self.in_planes)
        self.relu_Vasc = nn.ReLU(inplace=True)
        self.maxpool_Vasc = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)

        self.in_planes = self.in_planes * 2
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)

        self.in_planes = self.in_planes * 2
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)

        self.in_planes = self.in_planes * 2
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        

        self.layer1_Vasc = self._make_layer_Vasc(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2_Vasc = self._make_layer_Vasc(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3_Vasc = self._make_layer_Vasc(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)               
        self.layer4_Vasc = self._make_layer_Vasc(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.AttLayer1 = Attention_block(F_g=64, F_1=64, F_int=32)
        self.AttLayer2 = Attention_block(F_g=128, F_1=128, F_int=64)
        self.AttLayer3 = Attention_block(F_g=256, F_1=256, F_int=128)
        self.AttLayer4 = Attention_block(F_g=512, F_1=512, F_int=256)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear( block_inplanes[3] * block.expansion * 2, n_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1)
        xOriginalVasc = torch.unsqueeze(x[:,0,1,...],1) 

        """ BRAIN VASCULATURE """
        xVasc = xOriginalVasc
        xVasc = self.conv1_Vasc(xOriginal) 
        xVasc = self.bn1_Vasc(xVasc)
        xVasc = self.relu_Vasc(xVasc)
        if not self.no_max_pool:
            xVasc = self.maxpool_Vasc(xVasc) 

        xVasc_l1 = self.layer1_Vasc(xVasc) 
        xVasc_l2 = self.layer2_Vasc(xVasc_l1) 
        xVasc_l3 = self.layer3_Vasc(xVasc_l2) 
        xVasc_l4 = self.layer4_Vasc(xVasc_l3)

        """ BRAIN CTA """
        x = self.conv1(xOriginal) 
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x) 

        x = self.layer1(x) 
        x_attL1 = self.AttLayer1(g=x, x=xVasc_l1)
        x = torch.cat((x, x_attL1), dim=1)

        x = self.layer2(x) 
        x_attL2 = self.AttLayer2(g=x, x=xVasc_l2)
        x = torch.cat((x, x_attL2), dim=1)

        x = self.layer3(x) 
        x_attL3 = self.AttLayer3(g=x, x=xVasc_l3)
        x = torch.cat((x, x_attL3), dim=1)

        x = self.layer4(x) 
        x_attL4 = self.AttLayer4(g=x, x=xVasc_l4)

        x_all = torch.cat((x, x_attL4), dim=1)
    
        """ AVERAGE POOLING """
        x = self.avgpool(x_all) 
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 

        return x

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        if new_layer:
            self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _make_layer_Vasc(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes_Vasc != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes_Vasc, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes_Vasc,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        
        self.in_planes_Vasc = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes_Vasc, planes))

        return nn.Sequential(*layers)

