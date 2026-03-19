import math
from torch.nn import init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# configure logging
from elements.utils import LoggerSingleton
logger = LoggerSingleton.get_logger()

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def lstm_unit(input_size, hidden_size=None, num_layers = 2, batch_first = True):
    if hidden_size is None:
        hidden_size = input_size*2

    return nn.LSTM(
        input_size,
        hidden_size,
        num_layers=num_layers,
        batch_first = batch_first
    )


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if not np.array_equal(from_up.data.shape, from_down.data.shape):
            from_up = F.upsample(from_up, from_down.data.shape[2:], mode="bilinear")
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net.txt is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filters=64, up_mode='transpose',
                 merge_mode='concat', return_latent=False, latent_vector_size=1024):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.txt.
            start_filters: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
            return_latent:
                If True the forward method also returns the latent vector
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filters
        self.depth = depth
        self.return_latent = return_latent

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        if latent_vector_size != start_filters * (2**(depth-1)):
            self.conv_latent = conv1x1(start_filters * (2**(depth-1)), latent_vector_size)
        else:
            self.conv_latent = None
        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        if self.conv_latent is not None:
            latent_tensor = self.conv_latent(x)
        else:
            latent_tensor = x

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        x = F.log_softmax(x, dim=1)
        if self.return_latent:
            return x, latent_tensor
        else:
            return x

class DynamicCNN1D(nn.Module):
    def __init__(self, in_channels, num_classes, input_length,conv_block_type,
                 num_conv_layers=3, num_fc_layers=2, start_filters=32, dropout=0.3,
                 final_activation='logsoftmax'):
        """
        Arguments:
            in_channels: int, number of input channels for the convolutional layers.
            num_classes: int, number of output classes.
            input_length: int, the length of the input sequence.
            conv_block_type: string, type of convolutional block. Choices: 'A' for maxpool downsizing, 'B' for strided conv downsizing , 'C' for both.
            num_conv_layers: int, number of convolutional layers.
            num_fc_layers: int, number of fully connected layers.
            start_filters: int, number of output channels for the first convolutional layer.
            dropout: float, dropout probability for the first fully connected layer.
            final_activation: str, final activation layer to use. Options:
                'logsoftmax', 'softmax', 'sigmoid', or 'none' (no activation).
        """
        super(DynamicCNN1D, self).__init__()

        # convolutional layers with relu and pooling
        conv_layers = []
        current_channels = in_channels

        for i in range(num_conv_layers):
            out_channels = start_filters * (2 ** i)
            conv_layers.append(nn.Conv1d(current_channels, out_channels, kernel_size=3, stride= 1 if conv_block_type == 'A' else 2, padding=1))
            conv_layers.append(nn.ReLU())
            if conv_block_type != 'B':                  # there is max pooling in block type 'A' and 'C'
                conv_layers.append(nn.MaxPool1d(2))
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # calculating the flattened size after convolutional layers
        sample_input = torch.randn(1, in_channels, input_length)
        conv_output = self.conv_layers(sample_input)
        self.flat_dim = conv_output.shape[1] * conv_output.shape[2]

        # fully connected layers with one dropout layer after the first layer
        fc_layers = []
        current_dim = self.flat_dim
        for i in range(num_fc_layers - 1):
            # next_dim = max([x for x in [2**i for i in range(16)] if x <= current_dim//2])
            next_dim = 2 ** math.floor(math.log2(current_dim / 2))
            fc_layers.append(nn.Linear(current_dim, next_dim))
            fc_layers.append(nn.ReLU())
            if i == 0:
                fc_layers.append(nn.Dropout(dropout))
            current_dim = next_dim

        # final fully connected layer
        fc_layers.append(nn.Linear(current_dim, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)

        # the final activation layer
        if final_activation == 'logsoftmax':
            self.final_activation = nn.LogSoftmax(dim=1)
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'none':
            self.final_activation = nn.Identity()  # No activation layer
        else:
            raise ValueError("Invalid final activation. Choose from 'logsoftmax', 'softmax', 'sigmoid', or 'none'.")

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        #return self.final_activation(x)
        res = self.final_activation(x)
        #print(res)
        return res

class DynamicCNN3D(nn.Module):
    def __init__(self, in_channels: int, num_classes,
                 num_conv_layers=3, num_fc_layers=2, start_filters=4, dropout=0.3,
                 final_activation='logsoftmax', tile_size=5, tile_reduce_mode='avg'):
        """
        Arguments:
            in_channels: int, number of input channels (needs to be multiple of 2 ** num_conv_layers)
            num_classes: int, number of output classes.
            num_conv_layers: int, number of convolutional layers.
            num_fc_layers: int, number of fully connected layers.
            start_filters: int, number of output channels for the first convolutional layer.
            dropout: float, dropout probability for the first fully connected layer.
            final_activation: str, final activation layer to use. Options:
                'logsoftmax', 'softmax', 'sigmoid', or 'none' (no activation).
            tile_size: int, size of tile inputs during training
            tile_reduce_mode: which mode to use to reduce input tile to a single pixel. Options:
                'avg', 'max', or 'conv'

        """
        super(DynamicCNN3D, self).__init__()
        self.in_channels = in_channels

        # tile reduction to single pixel
        match tile_reduce_mode:
            case 'avg':
                self.reduce_layer = nn.AvgPool2d((tile_size, tile_size), stride=(1, 1),padding=(2,2))
            case 'max':
                self.reduce_layer = nn.MaxPool2d((tile_size, tile_size), stride=(1, 1),padding = (2,2))
            case 'conv':
                self.reduce_layer = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, (tile_size, tile_size)),
                    nn.ReLU(inplace=True),
                )
            case _:
                raise ValueError("Invalid value for tile_reduce_mode. Use one of 'avg', 'max', or 'conv'.")

        # convolutional layers with relu and pooling
        # note: these are 3D convolutions (spatially-aware equivalent of Conv1D when kernel size is 1 for height/width dims)
        # input for these layers is 5D, with a fake additional
        conv_layers = []
        current_channels = 1    # fake channel dim starts at size 1

        for i in range(num_conv_layers):
            out_channels = start_filters * (2 ** i)
            conv_layers.append(nn.Conv3d(current_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool3d((2, 1, 1)))
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # fully connected layers with one dropout layer after the first layer
        # note: Conv2D with kernel size of (1, 1) is functionally equivalent to fully connected layer
        # but supports spatial input for direct segmentation
        current_channels *= int(math.floor(in_channels / 2 ** num_conv_layers))
        fc_layers = []
        for i in range(num_fc_layers - 1):
            next_channels = 2 ** math.floor(math.log2(current_channels / 2))   # find next lower power-of-2
            fc_layers.append(nn.Conv2d(in_channels=current_channels, out_channels=next_channels, kernel_size=(1, 1)))
            fc_layers.append(nn.ReLU(inplace=True))
            if i == 0:
                fc_layers.append(nn.Dropout(dropout))
            current_channels = next_channels

        # final fully connected layer
        fc_layers.append(nn.Conv2d(in_channels=current_channels, out_channels=num_classes, kernel_size=(1, 1)))
        self.fc_layers = nn.Sequential(*fc_layers)

        # the final activation layer
        if final_activation == 'logsoftmax':
            self.final_activation = nn.LogSoftmax(dim=1)
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'none':
            self.final_activation = nn.Identity()  # No activation layer
        else:
            raise ValueError("Invalid final activation. Choose from 'logsoftmax', 'softmax', 'sigmoid', or 'none'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: PyTorch Tensor of shape (N, C, H, W)

        As the 5x5 averaging is now done internally in the network, the input shape during training should be (N, C, 5, 5)
        The output will also be 4D (N, n_classes, H, W). During training with individual tiles the height and width of the result will be 1.

        During inferencing, the entire image can be passed in directly. The output height and with are reduced by (tile_size - 1).
        """
        # apply tile reduction
        x = self.reduce_layer(x)

        # add additional 'fake' channel dim to make Conv3D work
        x = x.unsqueeze(1)
        x = self.conv_layers(x)

        # remove fake channel dim
        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))

        x = self.fc_layers(x)
        return self.final_activation(x)

class CnnLstm(nn.Module):
    def __init__(self, in_channels, num_classes, input_length=224, conv_block_type='A',
                 num_conv_layers=None, num_fc_layers=2, start_filters=32, dropout = 0.3, final_activation='logsoftmax',
                 num_lstm_blocks=None, num_lstm_layers=None):
        """
        Arguments:
            in_channels: int, number of input channels for the convolutional layers.
            num_classes: int, number of output classes.
            input_length: int, the length of the input sequence.
            conv_block_type: string, type of convolutional block. Choices: 'A' for maxpool downsizing, 'B' for strided conv downsizing , 'C' for both.
            num_conv_layers: int, number of convolutional layers.
            num_fc_layers: int, number of fully connected layers.
            start_filters: int, number of output channels for the first convolutional layer.
            dropout: float, dropout probability for the first fully connected layer.
            final_activation: str, final activation layer to use. Options:
                'logsoftmax', 'softmax', 'sigmoid', or 'none' (no activation).
            num_lstm_blocks: int, number of LSTM units.
            num_lstm_layers: int, number of LSTM hidden layers.
        """
        super(CnnLstm, self).__init__()

        #convolutional layers with relu and pooling
        conv_layers = []
        current_channels = in_channels
        #conv_layers.append( nn.BatchNorm1d(num_features=in_channels) )

        for i in range(num_conv_layers):
            out_channels = start_filters * (2 ** i)
            conv_layers.append(
                nn.Conv1d(current_channels, out_channels, kernel_size=3, stride=1 if conv_block_type == 'A' else 2,
                          padding=1))
            conv_layers.append(nn.ReLU())
            if conv_block_type != 'B':  # there is max pooling in block type 'A' and 'C'
                conv_layers.append(nn.MaxPool1d(2))
                input_length = input_length // 2
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        self.num_lstm_blocks = num_lstm_blocks
        bn = [nn.BatchNorm1d(num_features=current_channels)]
        lstm = []
        self.tanh = nn.Tanh()
        for i in range(num_lstm_blocks):
            lstm.append(lstm_unit(current_channels, num_layers = num_lstm_layers))
            current_channels = current_channels * 2
            bn.append(nn.BatchNorm1d(num_features=current_channels))
        self.lstm = nn.ModuleList(lstm)
        self.bn = nn.ModuleList(bn)

        #fully connected layers
        fc_layers = []
        flat_dim = current_channels * input_length
        for i in range(num_fc_layers - 1):
            next_dim = 2 ** math.floor(math.log2(flat_dim / 2))
            fc_layers.append(nn.Linear(flat_dim, next_dim))
            fc_layers.append(nn.ReLU())
            if i == 0:
                fc_layers.append(nn.Dropout(dropout))
            flat_dim = next_dim

        # final fully connected layer
        fc_layers.append(nn.Linear(flat_dim, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)

        # the final activation layer
        if final_activation == 'logsoftmax':
            self.final_activation = nn.LogSoftmax(dim=1)
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'none':
            self.final_activation = nn.Identity()  # No activation layer
        else:
            raise ValueError("Invalid final activation. Choose from 'logsoftmax', 'softmax', 'sigmoid', or 'none'.")

    def forward(self, x):
        x = self.conv_layers(x)
        # x = self.bn[0](x)
        # output of 1D-CNN: (B, C, L)

        # input required for LSTM: (B, L, C)
        for blocks in range(self.num_lstm_blocks):
            x = x.transpose(1, 2)
            x,_ = self.lstm[blocks](x)
            x = self.tanh(x)
            x = x.transpose(1, 2)
            #x = self.bn[blocks+1](x)

        # back to (B, C, L)
        #flatten
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return self.final_activation(x)


class LstmCnn(nn.Module):
    """
    LSTM-CNN Model: First applies LSTM to capture temporal/spectral dependencies,
    then applies CNN to extract high-level features.
    
    Architecture: Input -> LSTM blocks -> CNN layers -> FC layers -> Output
    """
    def __init__(self, in_channels, num_classes, input_length=224, conv_block_type='A',
                 num_conv_layers=None, num_fc_layers=2, start_filters=32, dropout=0.3, 
                 final_activation='logsoftmax', num_lstm_blocks=None, num_lstm_layers=None):
        """
        Arguments:
            in_channels: int, number of input channels (spectral bands).
            num_classes: int, number of output classes.
            input_length: int, the length of the input sequence.
            conv_block_type: string, type of convolutional block. Choices: 'A' for maxpool downsizing, 'B' for strided conv downsizing, 'C' for both.
            num_conv_layers: int, number of convolutional layers.
            num_fc_layers: int, number of fully connected layers.
            start_filters: int, number of output channels for the first convolutional layer.
            dropout: float, dropout probability for the first fully connected layer.
            final_activation: str, final activation layer to use. Options:
                'logsoftmax', 'softmax', 'sigmoid', or 'none' (no activation).
            num_lstm_blocks: int, number of LSTM blocks.
            num_lstm_layers: int, number of LSTM hidden layers per block.
        """
        super(LstmCnn, self).__init__()
        
        # LSTM layers first - process sequence to capture temporal/spectral dependencies
        self.num_lstm_blocks = num_lstm_blocks
        lstm = []
        bn_lstm = []
        self.tanh = nn.Tanh()
        
        # Input to LSTM: (B, L, C) where L is sequence length, C is channels
        # For spectral data: L = input_length (224), C = in_channels (1)
        current_features = in_channels
        
        for i in range(num_lstm_blocks):
            lstm.append(lstm_unit(current_features, num_layers=num_lstm_layers))
            current_features = current_features * 2  # LSTM doubles the feature size
            bn_lstm.append(nn.BatchNorm1d(num_features=input_length))  # Normalize over sequence length
        
        self.lstm = nn.ModuleList(lstm)
        self.bn_lstm = nn.ModuleList(bn_lstm)
        
        # After LSTM, we have (B, L, C) where C has been expanded
        # Convert to (B, C, L) for Conv1D
        lstm_output_channels = current_features
        
        # Convolutional layers - extract high-level features from LSTM output
        conv_layers = []
        current_channels = lstm_output_channels
        
        for i in range(num_conv_layers):
            out_channels = start_filters * (2 ** i)
            conv_layers.append(
                nn.Conv1d(current_channels, out_channels, kernel_size=3, 
                         stride=1 if conv_block_type == 'A' else 2, padding=1))
            conv_layers.append(nn.ReLU())
            if conv_block_type != 'B':  # there is max pooling in block type 'A' and 'C'
                conv_layers.append(nn.MaxPool1d(2))
                input_length = input_length // 2
            current_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Fully connected layers
        fc_layers = []
        flat_dim = current_channels * input_length
        
        for i in range(num_fc_layers - 1):
            next_dim = 2 ** math.floor(math.log2(flat_dim / 2))
            fc_layers.append(nn.Linear(flat_dim, next_dim))
            fc_layers.append(nn.ReLU())
            if i == 0:
                fc_layers.append(nn.Dropout(dropout))
            flat_dim = next_dim
        
        # Final fully connected layer
        fc_layers.append(nn.Linear(flat_dim, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Final activation layer
        if final_activation == 'logsoftmax':
            self.final_activation = nn.LogSoftmax(dim=1)
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'none':
            self.final_activation = nn.Identity()
        else:
            raise ValueError("Invalid final activation. Choose from 'logsoftmax', 'softmax', 'sigmoid', or 'none'.")
    
    def forward(self, x):
        """
        Forward pass: LSTM -> CNN -> FC
        
        Input shape: (B, C, L) where B=batch, C=channels (1), L=sequence length (224)
        """
        # Input: (B, C, L) = (B, 1, 224)
        # LSTM expects: (B, L, C)
        x = x.transpose(1, 2)  # (B, L, C)
        
        # Apply LSTM blocks
        for i in range(self.num_lstm_blocks):
            x, _ = self.lstm[i](x)  # (B, L, C) -> (B, L, 2*C)
            x = self.tanh(x)
            x = self.bn_lstm[i](x)  # Normalize over sequence dimension
        
        # Convert back to (B, C, L) for Conv1D
        x = x.transpose(1, 2)  # (B, C, L)
        
        # Apply CNN layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Apply FC layers
        x = self.fc_layers(x)
        
        return self.final_activation(x)


class DynamicCNN2D(nn.Module):
    """
    A flexible 2D CNN model for spatial image classification.
    
    Architecture:
    - Multiple 2D convolutional blocks with batch normalization and ReLU
    - Max pooling for spatial downsampling
    - Fully connected layers for classification
    - Optional dropout for regularization
    
    Designed for 64x64 tiles with spectral channels (e.g., 224 channels for hyperspectral).
    
    Args:
        in_channels (int): Number of input channels (e.g., 224 for hyperspectral)
        num_classes (int): Number of output classes
        num_conv_layers (int): Number of convolutional layers (default: 3)
        num_fc_layers (int): Number of fully connected layers (default: 2)
        start_filters (int): Number of filters in the first conv layer (default: 32)
        dropout (float): Dropout rate (default: 0.3)
        final_activation (str): Final activation function ('logsoftmax', 'softmax', 'none')
    """
    
    def __init__(self, in_channels, num_classes, num_conv_layers=3, num_fc_layers=2,
                 start_filters=32, dropout=0.3, final_activation='logsoftmax'):
        super(DynamicCNN2D, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers if num_conv_layers is not None else 3
        self.num_fc_layers = num_fc_layers if num_fc_layers is not None else 2
        self.start_filters = start_filters
        self.dropout_rate = dropout
        self.final_activation = final_activation
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(self.num_conv_layers):
            out_ch = start_filters * (2 ** i)
            
            # Conv layer
            self.conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(out_ch))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_ch = out_ch
        
        # Calculate flattened size after convolutions and pooling
        # For 64x64 input with num_conv_layers pooling operations:
        # 64 -> 32 -> 16 -> 8 (for 3 layers)
        spatial_size = 64 // (2 ** self.num_conv_layers)
        self.flattened_size = in_ch * spatial_size * spatial_size
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_bn_layers = nn.ModuleList()
        
        fc_in = self.flattened_size
        fc_hidden = max(256, self.flattened_size // 4)
        
        for i in range(self.num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(fc_in, fc_hidden))
            self.fc_bn_layers.append(nn.BatchNorm1d(fc_hidden))
            fc_in = fc_hidden
            fc_hidden = max(128, fc_hidden // 2)
        
        # Output layer
        self.fc_layers.append(nn.Linear(fc_in, num_classes))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.pool_layers[i](x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for i in range(self.num_fc_layers - 1):
            x = self.fc_layers[i](x)
            x = self.fc_bn_layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.fc_layers[-1](x)
        
        # Final activation
        if self.final_activation == 'logsoftmax':
            x = F.log_softmax(x, dim=1)
        elif self.final_activation == 'softmax':
            x = F.softmax(x, dim=1)
        
        return x
