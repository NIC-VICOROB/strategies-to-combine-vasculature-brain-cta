import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

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
class vggNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(vggNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
        self.maxPooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        xMp = self.maxPooling(x3)

        return xMp

class encoder_1VGG(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(encoder_1VGG, self).__init__()
        self.encoder_first = vggNet(in_channels, out_channels, stride=stride, padding=padding)

    def forward(self, x):
        x = self.encoder_first(x) 
        return x
        
class encoder_2VGG(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(encoder_2VGG, self).__init__()
        self.encoder_first = vggNet(in_channels, out_channels, stride=stride, padding=padding)
        self.encoder = vggNet(out_channels, out_channels, stride=stride, padding=padding)

    def forward(self, x):
        x = self.encoder_first(x) 
        x = self.encoder(x)

        return x

class encoder_3VGG(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(encoder_3VGG, self).__init__()
        self.encoder_first = vggNet(in_channels, out_channels, stride=stride, padding=padding)
        self.encoder1 = vggNet(out_channels, out_channels, stride=stride, padding=padding)
        self.encoder2 = vggNet(out_channels, out_channels, stride=stride, padding=padding)

    def forward(self, x):
        x = self.encoder_first(x) 
        x = self.encoder1(x)
        x = self.encoder2(x)

        return x

class encoder_x_xFlip(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding='same'):
        super(encoder_x_xFlip, self).__init__()
        self.encoder = encoder_3VGG(in_channels, out_channels, stride=stride, padding=padding)

    def forward(self, x, xFlip):
        x1 = self.encoder(x)
        x2 = self.encoder(xFlip)

        return x1, x2

class merge_process(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, padding=2, depth_processing=2):
        super(merge_process, self).__init__()
        self.encoder = encoder_2VGG(in_channels, out_channels, stride=1, padding=padding)

        self.act = nn.LeakyReLU(inplace=True)
        self.nclasses = n_classes

        self.ap3 = nn.AvgPool3d(kernel_size=(2,5,4))
        self.lin = nn.Linear(in_features=24, out_features=n_classes)

    def forward(self, x, x_flip):
        merged_layer = torch.abs(x - x_flip) 
        encoding = self.encoder(merged_layer)
        gavgp = torch.flatten(self.ap3(encoding),start_dim=1)
        dense = self.lin(gavgp)

        return dense

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



""" MODEL CLASSES """
class DeepSymNetv3(pl.LightningModule):
    def __init__(self, 
                 input_channels=1, 
                 n_filters_vgg=24, 
                 number_classes=1, 
                 loss=nn.BCEWithLogitsLoss(), 
                 lr=1e-4, 
                 return_activated_output=False):
        
        super(DeepSymNetv3, self).__init__()
        
        self.n_outputs = number_classes

        self.loss = loss
        self.lr = lr
        
        self.return_activated_output = return_activated_output

        self.encoder = encoder_x_xFlip(input_channels, out_channels=n_filters_vgg, stride=1, padding=1)
        self.merge_proc = merge_process(in_channels=n_filters_vgg, out_channels=n_filters_vgg, n_classes=number_classes, padding=1, depth_processing=2)
        
        if self.n_outputs == 1:
            self.activation = nn.Sigmoid()
        elif self.n_outputs > 1:
            self.activation = nn.Softmax()

        # Evaluation metrics
        if self.n_outputs == 1:
            self.evaluation_metric_train = torchmetrics.AUROC(task='binary')
            self.evaluation_metric_val = torchmetrics.AUROC(task='binary')

    def forward(self, x): 
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 
        xFlip = torch.unsqueeze(x[:,0,1,...],1)    
 
        """ Encoding path """
        xEnc, xFlipEnc = self.encoder(xOriginal, xFlip) 
        xMerged = self.merge_proc(xEnc, xFlipEnc)

        if self.return_activated_output:
            xMerged = self.activation(xMerged)

        return xMerged

class DeepSymNetv3_multichannel(pl.LightningModule):
    def __init__(self,  
                 input_channels=1, 
                 n_filters_vgg=24, 
                 number_classes=1, 
                 loss=nn.BCEWithLogitsLoss(), 
                 lr=1e-4, 
                 return_activated_output=False):
        
        super(DeepSymNetv3_multichannel, self).__init__()
        
        self.n_outputs = number_classes

        self.loss = loss
        self.lr = lr
        
        self.return_activated_output = return_activated_output

        self.encoder = encoder_x_xFlip(input_channels, out_channels=n_filters_vgg, stride=1, padding=1)
        self.merge_proc = merge_process(in_channels=n_filters_vgg, out_channels=n_filters_vgg, n_classes=number_classes, padding=1, depth_processing=2)
        
        if self.n_outputs == 1:
            self.activation = nn.Sigmoid()
        elif self.n_outputs > 1:
            self.activation = nn.Softmax()

        # Evaluation metrics
        if self.n_outputs == 1:
            self.evaluation_metric_train = torchmetrics.AUROC(task='binary')
            self.evaluation_metric_val = torchmetrics.AUROC(task='binary')

    def forward(self, x): 
        x_original = torch.unsqueeze(x[:,0,0,...],1) 
        x_flip = torch.unsqueeze(x[:,0,1,...],1)       
        xOriginal_Vasc = torch.unsqueeze(x[:,0,2,...],1)
        xFlip_Vasc = torch.unsqueeze(x[:,0,3,...],1)    
   
        xOriginal = torch.cat([x_original,xOriginal_Vasc],dim=1)
        xFlip = torch.cat([x_flip,xFlip_Vasc],dim=1)

        """ Encoding path """
        xEnc, xFlipEnc = self.encoder(xOriginal, xFlip) 
        xMerged = self.merge_proc(xEnc, xFlipEnc)

        if self.return_activated_output:
            xMerged = self.activation(xMerged)

        return xMerged

class DeepSymNetv3_attention_strategy_1(pl.LightningModule):
    def __init__(self,  
                 input_channels=1, 
                 n_filters_vgg=24, 
                 number_classes=1, 
                 loss=nn.BCEWithLogitsLoss(), 
                 lr=1e-4, 
                 return_activated_output=False):
        
        super(DeepSymNetv3_attention_strategy_1, self).__init__()
        
        self.n_outputs = number_classes

        self.loss = loss
        self.lr = lr
        
        self.return_activated_output = return_activated_output

        # Encoders
        self.encoder_first_Vasc = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder_first = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1 = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2 = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)

        # Attention Gates
        self.AttLayerAfterEncoder = Attention_block(F_g=24, F_1=24, F_int=12)
        self.AttLayerAfterEncoder_flip = Attention_block(F_g=24, F_1=24, F_int=12)

        # Merging hemispheres features
        self.merge_proc = merge_process(in_channels=n_filters_vgg*2, out_channels=n_filters_vgg, n_classes=number_classes, padding=1, depth_processing=2)
        
        if self.n_outputs == 1:
            self.activation = nn.Sigmoid()
        elif self.n_outputs > 1:
            self.activation = nn.Softmax()

        # Evaluation metrics
        if self.n_outputs == 1:
            self.evaluation_metric_train = torchmetrics.AUROC(task='binary')
            self.evaluation_metric_val = torchmetrics.AUROC(task='binary')

    def forward(self, x): 
        # Brain CTA
        x_original = torch.unsqueeze(x[:,0,0,...],1) 
        x_flip = torch.unsqueeze(x[:,0,1,...],1)    
        # Vasculature   
        xOriginal_Vasc = torch.unsqueeze(x[:,0,2,...],1)
        xFlip_Vasc = torch.unsqueeze(x[:,0,3,...],1)    
   

        """ Encoding Vasculature """
        """ One hemisphere """
        x_Vasc_first = self.encoder_first_Vasc(xOriginal_Vasc) 
        x_Vasc_1 = self.encoder1_Vasc(x_Vasc_first) 
        x_Vasc_2 = self.encoder2_Vasc(x_Vasc_1) 

        """ The other hemisphere """
        x_Vasc_Flip_first = self.encoder_first_Vasc(xFlip_Vasc) 
        x_Vasc_Flip_1 = self.encoder1_Vasc(x_Vasc_Flip_first) 
        x_Vasc_Flip_2 = self.encoder2_Vasc(x_Vasc_Flip_1) 


        """ Encoding CTA """
        """ One hemisphere """
        x_Original = self.encoder_first(x_original) 
        x_first = self.encoder1(x_Original)

        x_1 = self.encoder2(x_first)
        x_att_2 = self.AttLayerAfterEncoder(x_1,x_Vasc_2)
        x_2 = torch.cat([x_1,x_att_2],dim=1)

        """ The other hemisphere """
        x_Flip = self.encoder_first(x_flip) 
        x_Flip_first = self.encoder1(x_Flip)

        x_Flip_1 = self.encoder2(x_Flip_first)
        x_Flip_att_2 = self.AttLayerAfterEncoder_flip(x_Flip_1,x_Vasc_Flip_2)
        x_Flip_2 = torch.cat([x_Flip_1,x_Flip_att_2],dim=1)

        xMerged = self.merge_proc(x_2, x_Flip_2)

        if self.return_activated_output:
            xMerged = self.activation(xMerged)
            
        return xMerged

class DeepSymNetv3_attention_strategy_2(pl.LightningModule):
    def __init__(self,  
                 input_channels=1, 
                 n_filters_vgg=24, 
                 number_classes=1, 
                 loss=nn.BCEWithLogitsLoss(), 
                 lr=1e-4, 
                 return_activated_output=False):
        
        super(DeepSymNetv3_attention_strategy_2, self).__init__()
        
        self.n_outputs = number_classes

        self.loss = loss
        self.lr = lr
        
        self.return_activated_output = return_activated_output

        # Encoders
        self.encoder_first_Vasc = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder_first = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1 = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2 = vggNet(n_filters_vgg*2, n_filters_vgg, stride=1, padding=1)

        # Attention Gates
        self.AttLayer1 = Attention_block(F_g=24, F_1=24, F_int=12)
        self.AttLayer2 = Attention_block(F_g=24, F_1=24, F_int=12)
        self.AttLayer1_flip = Attention_block(F_g=24, F_1=24, F_int=12)
        self.AttLayer2_flip = Attention_block(F_g=24, F_1=24, F_int=12)

        # Merging hemispheres features
        self.merge_proc = merge_process(in_channels=n_filters_vgg*2, out_channels=n_filters_vgg, n_classes=number_classes, padding=1, depth_processing=2)
        
        if self.n_outputs == 1:
            self.activation = nn.Sigmoid()
        elif self.n_outputs > 1:
            self.activation = nn.Softmax()

        # Evaluation metrics
        if self.n_outputs == 1:
            self.evaluation_metric_train = torchmetrics.AUROC(task='binary')
            self.evaluation_metric_val = torchmetrics.AUROC(task='binary')

    def forward(self, x): 
        # Brain CTA
        x_original = torch.unsqueeze(x[:,0,0,...],1) 
        x_flip = torch.unsqueeze(x[:,0,1,...],1)    
        # Vasculature   
        xOriginal_Vasc = torch.unsqueeze(x[:,0,2,...],1)
        xFlip_Vasc = torch.unsqueeze(x[:,0,3,...],1)    
   

        """ Encoding Vasculature """
        """ One hemisphere """
        x_Vasc_first = self.encoder_first_Vasc(xOriginal_Vasc) 
        x_Vasc_1 = self.encoder1_Vasc(x_Vasc_first) 
        x_Vasc_2 = self.encoder2_Vasc(x_Vasc_1) 

        """ The other hemisphere """
        x_Vasc_Flip_first = self.encoder_first_Vasc(xFlip_Vasc) 
        x_Vasc_Flip_1 = self.encoder1_Vasc(x_Vasc_Flip_first) 
        x_Vasc_Flip_2 = self.encoder2_Vasc(x_Vasc_Flip_1) 


        """ Encoding CTA """
        """ One hemisphere """
        x_Original = self.encoder_first(x_original) 

        x_first = self.encoder1(x_Original)
        x_att_1 = self.AttLayer1(x_first,x_Vasc_1)
        x_1 = torch.cat([x_first,x_att_1],dim=1)

        x_1 = self.encoder2(x_1)
        x_att_2 = self.AttLayer2(x_1,x_Vasc_2)
        x_2 = torch.cat([x_1,x_att_2],dim=1)

        """ The other hemisphere """
        x_Flip = self.encoder_first(x_flip) 

        x_Flip_first = self.encoder1(x_Flip)
        x_Flip_att_1 = self.AttLayer1_flip(x_Flip_first,x_Vasc_Flip_1)
        x_Flip_1 = torch.cat([x_Flip_first,x_Flip_att_1],dim=1)

        x_Flip_1 = self.encoder2(x_Flip_1)
        x_Flip_att_2 = self.AttLayer2_flip(x_Flip_1,x_Vasc_Flip_2)
        x_Flip_2 = torch.cat([x_Flip_1,x_Flip_att_2],dim=1)

        xMerged = self.merge_proc(x_2, x_Flip_2)

        if self.return_activated_output:
            xMerged = self.activation(xMerged)
            
        return xMerged

class DeepSymNetv3_attention_strategy_3(pl.LightningModule):
    def __init__(self,  
                 input_channels=1, 
                 n_filters_vgg=24, 
                 number_classes=1, 
                 loss=nn.BCEWithLogitsLoss(), 
                 lr=1e-4, 
                 return_activated_output=False):
        
        super(DeepSymNetv3_attention_strategy_3, self).__init__()
        
        self.n_outputs = number_classes

        self.loss = loss
        self.lr = lr
        
        self.return_activated_output = return_activated_output

        # Encoders
        self.encoder_first_Vasc = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder_first = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1 = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2 = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)

        # Attention Gates 
        self.AttLayer = Attention_block(F_g=24, F_1=24, F_int=12)

        # After merging hemispheres features
        self.encoder = encoder_2VGG(in_channels=n_filters_vgg*2, out_channels=n_filters_vgg, stride=1, padding=1)
        self.ap3 = nn.AvgPool3d(kernel_size=(2,5,4))
        self.lin = nn.Linear(in_features=24, out_features=number_classes)

        if self.n_outputs == 1:
            self.activation = nn.Sigmoid()
        elif self.n_outputs > 1:
            self.activation = nn.Softmax()

        # Evaluation metrics
        if self.n_outputs == 1:
            self.evaluation_metric_train = torchmetrics.AUROC(task='binary')
            self.evaluation_metric_val = torchmetrics.AUROC(task='binary')

    def forward(self, x): 
        # Brain CTA
        x_original = torch.unsqueeze(x[:,0,0,...],1) 
        x_flip = torch.unsqueeze(x[:,0,1,...],1)    
        # Vasculature   
        xOriginal_Vasc = torch.unsqueeze(x[:,0,2,...],1)
        xFlip_Vasc = torch.unsqueeze(x[:,0,3,...],1)    
   

        """ Encoding Vasculature """
        """ One hemisphere """
        x_Vasc_first = self.encoder_first_Vasc(xOriginal_Vasc) 
        x_Vasc_1 = self.encoder1_Vasc(x_Vasc_first) 
        x_Vasc_2 = self.encoder2_Vasc(x_Vasc_1) 

        """ The other hemisphere """
        x_Vasc_Flip_first = self.encoder_first_Vasc(xFlip_Vasc) 
        x_Vasc_Flip_1 = self.encoder1_Vasc(x_Vasc_Flip_first) 
        x_Vasc_Flip_2 = self.encoder2_Vasc(x_Vasc_Flip_1) 

        # L1 layer
        merged_layer_vasc = torch.abs(x_Vasc_2 - x_Vasc_Flip_2)


        """ Encoding CTA """
        """ One hemisphere """
        x_Original = self.encoder_first(x_original) 
        x_first = self.encoder1(x_Original)
        x_1 = self.encoder2(x_first)

        """ The other hemisphere """
        x_Flip = self.encoder_first(x_flip) 
        x_Flip_first = self.encoder1(x_Flip)
        x_Flip_1 = self.encoder2(x_Flip_first)

        # L1 layer
        merged_layer = torch.abs(x_1 - x_Flip_1)


        """ Attention after L1 layer """
        x_att = self.AttLayer(merged_layer,merged_layer_vasc)

        x_marged = torch.cat([merged_layer,x_att],dim=1)
        encoding = self.encoder(x_marged)
        gavgp = torch.flatten(self.ap3(encoding),start_dim=1)
        xMerged = self.lin(gavgp)


        if self.return_activated_output:
            xMerged = self.activation(xMerged)
            
        return xMerged

class DeepSymNetv3_attention_strategy_4(pl.LightningModule):
    def __init__(self,  
                 input_channels=1, 
                 n_filters_vgg=24, 
                 number_classes=1, 
                 loss=nn.BCEWithLogitsLoss(), 
                 lr=1e-4, 
                 return_activated_output=False):
        
        super(DeepSymNetv3_attention_strategy_4, self).__init__()
        
        self.n_outputs = number_classes

        self.loss = loss
        self.lr = lr
        
        self.return_activated_output = return_activated_output

        # Encoders
        self.encoder_first_Vasc = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder_first = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1 = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2 = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)

        # Attention Gates 
        self.AttLayerLast = Attention_block(F_g=24, F_1=24, F_int=12)

        # After merging hemispheres features
        self.encoder = encoder_2VGG(in_channels=n_filters_vgg, out_channels=n_filters_vgg, stride=1, padding=1)
        self.encoder_vasc = encoder_2VGG(in_channels=n_filters_vgg, out_channels=n_filters_vgg, stride=1, padding=1)
        self.ap3 = nn.AvgPool3d(kernel_size=(2,5,4))
        self.lin = nn.Linear(in_features=24*2, out_features=number_classes)

        if self.n_outputs == 1:
            self.activation = nn.Sigmoid()
        elif self.n_outputs > 1:
            self.activation = nn.Softmax()

        # Evaluation metrics
        if self.n_outputs == 1:
            self.evaluation_metric_train = torchmetrics.AUROC(task='binary')
            self.evaluation_metric_val = torchmetrics.AUROC(task='binary')

    def forward(self, x): 
        # Brain CTA
        x_original = torch.unsqueeze(x[:,0,0,...],1) 
        x_flip = torch.unsqueeze(x[:,0,1,...],1)    
        # Vasculature   
        xOriginal_Vasc = torch.unsqueeze(x[:,0,2,...],1)
        xFlip_Vasc = torch.unsqueeze(x[:,0,3,...],1)    
   

        """ Encoding Vasculature """
        """ One hemisphere """
        x_Vasc_first = self.encoder_first_Vasc(xOriginal_Vasc) 
        x_Vasc_1 = self.encoder1_Vasc(x_Vasc_first) 
        x_Vasc_2 = self.encoder2_Vasc(x_Vasc_1) 

        """ The other hemisphere """
        x_Vasc_Flip_first = self.encoder_first_Vasc(xFlip_Vasc) 
        x_Vasc_Flip_1 = self.encoder1_Vasc(x_Vasc_Flip_first) 
        x_Vasc_Flip_2 = self.encoder2_Vasc(x_Vasc_Flip_1) 

        # L1 layer
        merged_layer_vasc = torch.abs(x_Vasc_2 - x_Vasc_Flip_2)

        # Process
        encoding_vasc = self.encoder_vasc(merged_layer_vasc)


        """ Encoding CTA """
        """ One hemisphere """
        x_Original = self.encoder_first(x_original) 
        x_first = self.encoder1(x_Original)
        x_1 = self.encoder2(x_first)

        """ The other hemisphere """
        x_Flip = self.encoder_first(x_flip) 
        x_Flip_first = self.encoder1(x_Flip)
        x_Flip_1 = self.encoder2(x_Flip_first)

        # L1 layer
        merged_layer = torch.abs(x_1 - x_Flip_1)

        # Process
        encoding = self.encoder(merged_layer)

        """ Attention after processing """
        encoding_att = self.AttLayerLast(encoding, encoding_vasc)
        encoding_att = torch.cat([encoding, encoding_att], dim=1)

        gavgp = torch.flatten(self.ap3(encoding_att),start_dim=1)
        xMerged = self.lin(gavgp)

        if self.return_activated_output:
            xMerged = self.activation(xMerged)
            
        return xMerged

class DeepSymNetv3_attention_strategy_5(pl.LightningModule):
    def __init__(self,  
                 input_channels=1, 
                 n_filters_vgg=24, 
                 number_classes=1, 
                 loss=nn.BCEWithLogitsLoss(), 
                 lr=1e-4, 
                 return_activated_output=False):
        
        super(DeepSymNetv3_attention_strategy_5, self).__init__()
        
        self.n_outputs = number_classes

        self.loss = loss
        self.lr = lr
        
        self.return_activated_output = return_activated_output

        # Encoders
        self.encoder_first_Vasc = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2_Vasc = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder_first = vggNet(input_channels, n_filters_vgg, stride=1, padding=1)
        self.encoder1 = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)
        self.encoder2 = vggNet(n_filters_vgg, n_filters_vgg, stride=1, padding=1)

        # Attention Gates 
        self.AttLayer = Attention_block(F_g=24, F_1=24, F_int=12)
        self.AttLayerLast = Attention_block(F_g=24, F_1=24, F_int=12)

        # After merging hemispheres features
        self.encoder = encoder_2VGG(in_channels=n_filters_vgg*2, out_channels=n_filters_vgg, stride=1, padding=1)
        self.encoder_vasc = encoder_2VGG(in_channels=n_filters_vgg, out_channels=n_filters_vgg, stride=1, padding=1)
        self.ap3 = nn.AvgPool3d(kernel_size=(2,5,4))
        self.lin = nn.Linear(in_features=24*2, out_features=number_classes)

        if self.n_outputs == 1:
            self.activation = nn.Sigmoid()
        elif self.n_outputs > 1:
            self.activation = nn.Softmax()

        # Evaluation metrics
        if self.n_outputs == 1:
            self.evaluation_metric_train = torchmetrics.AUROC(task='binary')
            self.evaluation_metric_val = torchmetrics.AUROC(task='binary')

    def forward(self, x): 
        # Brain CTA
        x_original = torch.unsqueeze(x[:,0,0,...],1) 
        x_flip = torch.unsqueeze(x[:,0,1,...],1)    
        # Vasculature   
        xOriginal_Vasc = torch.unsqueeze(x[:,0,2,...],1)
        xFlip_Vasc = torch.unsqueeze(x[:,0,3,...],1)    
   

        """ Encoding Vasculature """
        """ One hemisphere """
        x_Vasc_first = self.encoder_first_Vasc(xOriginal_Vasc) 
        x_Vasc_1 = self.encoder1_Vasc(x_Vasc_first) 
        x_Vasc_2 = self.encoder2_Vasc(x_Vasc_1) 

        """ The other hemisphere """
        x_Vasc_Flip_first = self.encoder_first_Vasc(xFlip_Vasc) 
        x_Vasc_Flip_1 = self.encoder1_Vasc(x_Vasc_Flip_first) 
        x_Vasc_Flip_2 = self.encoder2_Vasc(x_Vasc_Flip_1) 

        # L1 layer
        merged_layer_vasc = torch.abs(x_Vasc_2 - x_Vasc_Flip_2)

        # Process
        encoding_vasc = self.encoder_vasc(merged_layer_vasc)


        """ Encoding CTA """
        """ One hemisphere """
        x_Original = self.encoder_first(x_original) 
        x_first = self.encoder1(x_Original)
        x_1 = self.encoder2(x_first)

        """ The other hemisphere """
        x_Flip = self.encoder_first(x_flip) 
        x_Flip_first = self.encoder1(x_Flip)
        x_Flip_1 = self.encoder2(x_Flip_first)

        # L1 layer
        merged_layer = torch.abs(x_1 - x_Flip_1)

        """ Attention after L1 layer """
        x_att = self.AttLayer(merged_layer,merged_layer_vasc)
        x_marged = torch.cat([merged_layer,x_att],dim=1)

        # Process
        encoding = self.encoder(x_marged)

        """ Attention after processing """
        encoding_att = self.AttLayerLast(encoding, encoding_vasc)
        encoding_att = torch.cat([encoding, encoding_att], dim=1)

        gavgp = torch.flatten(self.ap3(encoding_att),start_dim=1)
        xMerged = self.lin(gavgp)

        if self.return_activated_output:
            xMerged = self.activation(xMerged)
            
        return xMerged





