import torch
import torch.utils
import torch.utils.data
NoneType = type(None)
from elements.model_wrappers import UNet, DynamicCNN1D, DynamicCNN2D, DynamicCNN3D, CnnLstm, LstmCnn
from elements.common.RGB_models.RGBViT import RGBViT, RGBResNet, RGBHybrid
from elements.common.HSI_models.SSFTT.SSFTTnet import SSFTTnet
from elements.common.HSI_models.SSFTT.SSFTTnet_unet import SSFTTnet_UNet
from elements.common.HSI_models.SSFTT.SSFTTnet_lstm import SSFTTnet_LSTM
from elements.common.HSI_models.SSFTT.SSFTTnet_fusion import (
    SpectralRGBFusion3DCNN, SpectralRGBFusion2Plus1D,
    SpectralRGBFusionHybrid, SpectralRGBFusionAttention
)
from elements.common.HSI_models.Data_Fusion.SSFTTnet_fusion_transformer import SSFTTFusion, SSFTTFusionDeep

# configure logging
from elements.utils import LoggerSingleton
logger = LoggerSingleton.get_logger()

def load_weights_cnn1d_to_cnn3d(cnn1d_model_path, cnn3d_model):
    """
    Load weights from a trained DynamicCNN1D model into a DynamicCNN3D model.

    :param cnn1d_model_path: Path to the trained DynamicCNN1D state dictionary.
    :param cnn3d_model: Instance of DynamicCNN3D to which the weights will be loaded.
    :return: The updated DynamicCNN3D model.
    """
    # load cnn1d state
    cnn1d_state_dict = torch.load(cnn1d_model_path)

    # get cnn3d state
    cnn3d_state_dict = cnn3d_model.state_dict()

    for name, param in cnn1d_state_dict.items():
        if name in cnn3d_state_dict:
            if "bias" in name:
                cnn3d_state_dict[name] = param # bis dimenstions are the same
            else:
                cnn3d_state_dict[name] = param.unsqueeze(-1).unsqueeze(-1)  # add two dimensions for weights

    # load the revised cnn1d state to cnn3d
    cnn3d_model.load_state_dict(cnn3d_state_dict, strict=False)

    return cnn3d_model

def initialize_model(model_type, in_channels, out_classes,start_filters, cnn_input_length, cnn_conv_block_type, cnn_conv_layers=None,
                     cnn_fc_layers=5,cnn_final_activation='logsoftmax', cnn_dropout=0.3, unet_depth=4,  best_state_path=None, num_lstm_layers=None, num_lstm_blocks=None,
                     rgb_image_size=64, rgb_patch_size=8, rgb_dim=256, rgb_depth=6, rgb_heads=8, rgb_mlp_dim=512,
                     num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
    """
    Initialize and return a model based on the specified model type.

    Args:
        model_type (str): Type of model to create ('cnn1d', 'cnn2d', 'cnn3d', 'unet', 'cnn_lstm', 'lstm_cnn', 'rgb_vit', 'rgb_resnet', 'rgb_hybrid').
        in_channels (int): Number of input channels.
        out_classes (int): Number of output classes.
        cnn_input_length (int): Length of the input vector (for 'cnn1d' model).
        cnn_conv_block_type (str): Type of convolutional block to use. Choices: 'A' for maxpool downsizing, 'B' for strided conv downsizing , 'C' for both.
        cnn_conv_layers (int, optional): Number of convolutional layers (for 'cnn1d' model). Defaults to 2.
        cnn_fc_layers (int, optional): Number of fully connected layers (for 'cnn1d' model). Defaults to 3.
        cnn_final_activation (str): The final activation function.
        cnn_dropout (float, optional): Dropout rate for the first fully connected layer (for 'cnn1d' model). Defaults to 0.3.
        unet_depth (int, optional): Depth parameter for U-Net.txt (for 'unet' model). Defaults to 4.
        start_filters (int, optional): Starting number of filters for U-Net.txt (for 'unet' model). Defaults to 64.
        best_state_path (str, optional): Path to load a saved model state. If provided, loads the saved state.
        num_lstm_blocks (int, optional): Number of LSTM units. Defaults to 2.
        num_lstm_layers (int, optional): Number of LSTM layers. Defaults to 2.
        rgb_image_size (int, optional): Image size for RGB models. Defaults to 64.
        rgb_patch_size (int, optional): Patch size for RGB ViT. Defaults to 8.
        rgb_dim (int, optional): Embedding dimension for RGB ViT. Defaults to 256.
        rgb_depth (int, optional): Number of transformer layers for RGB ViT. Defaults to 6.
        rgb_heads (int, optional): Number of attention heads for RGB ViT. Defaults to 8.
        rgb_mlp_dim (int, optional): MLP dimension for RGB ViT. Defaults to 512.

    Returns:
        nn.Module: Initialized model loaded to the specified device.
    """

    if model_type == 'cnn1d':
        model = DynamicCNN1D(in_channels=in_channels,num_classes=out_classes,input_length=cnn_input_length, conv_block_type=cnn_conv_block_type,
            num_conv_layers=cnn_conv_layers,num_fc_layers=cnn_fc_layers,final_activation=cnn_final_activation,
            start_filters=start_filters,dropout=cnn_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'cnn2d':
        model = DynamicCNN2D(in_channels=in_channels, num_classes=out_classes, num_conv_layers=cnn_conv_layers,
            num_fc_layers=cnn_fc_layers, final_activation=cnn_final_activation, start_filters=start_filters,
            dropout=cnn_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'cnn3d':
        model = DynamicCNN3D(in_channels=in_channels,num_classes=out_classes,num_conv_layers=cnn_conv_layers,
            num_fc_layers=cnn_fc_layers,final_activation=cnn_final_activation,start_filters=start_filters,
            dropout=cnn_dropout)
        if best_state_path is not None:
            try:
                model.load_state_dict(torch.load(best_state_path))
            except RuntimeError:
                load_weights_cnn1d_to_cnn3d(best_state_path, model)

    elif model_type == 'unet':
        model = UNet(in_channels=in_channels, num_classes=out_classes, depth=unet_depth, start_filters=start_filters)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'cnn_lstm':
        model = CnnLstm(in_channels=in_channels, num_classes=out_classes, num_conv_layers=cnn_conv_layers,
                        input_length=cnn_input_length,
                    start_filters=start_filters,
                    num_lstm_layers=num_lstm_layers, num_lstm_blocks=num_lstm_blocks)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'lstm_cnn':
        model = LstmCnn(in_channels=in_channels, num_classes=out_classes, num_conv_layers=cnn_conv_layers,
                        input_length=cnn_input_length, conv_block_type=cnn_conv_block_type,
                        start_filters=start_filters, dropout=cnn_dropout,
                        num_lstm_layers=num_lstm_layers, num_lstm_blocks=num_lstm_blocks,
                        num_fc_layers=cnn_fc_layers, final_activation=cnn_final_activation)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'rgb_vit':
        model = RGBViT(image_size=rgb_image_size, patch_size=rgb_patch_size, num_classes=out_classes,
                      dim=rgb_dim, depth=rgb_depth, heads=rgb_heads, mlp_dim=rgb_mlp_dim,
                      channels=3, dropout=cnn_dropout, emb_dropout=cnn_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'rgb_resnet':
        model = RGBResNet(num_classes=out_classes, channels=3)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'rgb_hybrid':
        model = RGBHybrid(image_size=rgb_image_size, num_classes=out_classes,
                         dim=rgb_dim, depth=rgb_depth, heads=rgb_heads, mlp_dim=rgb_mlp_dim,
                         channels=3, dropout=cnn_dropout, emb_dropout=cnn_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'fusion_3dcnn':
        model = SpectralRGBFusion3DCNN(num_classes=out_classes, start_filters=start_filters,
                                       num_conv_layers=cnn_conv_layers or 3, dropout=cnn_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'fusion_2plus1d':
        model = SpectralRGBFusion2Plus1D(num_classes=out_classes, start_filters=start_filters,
                                         num_blocks=cnn_conv_layers or 3, dropout=cnn_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'fusion_hybrid':
        model = SpectralRGBFusionHybrid(num_classes=out_classes, spectral_filters=start_filters,
                                        rgb_filters=start_filters // 2, dropout=cnn_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'fusion_attention':
        model = SpectralRGBFusionAttention(num_classes=out_classes, base_filters=start_filters,
                                           dropout=cnn_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'ssftt':
        model = SSFTTnet(in_channels=in_channels, num_classes=out_classes,
                        num_tokens=num_tokens, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                        dropout=dropout, emb_dropout=emb_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'ssftt_unet':
        model = SSFTTnet_UNet(in_channels=in_channels, num_classes=out_classes,
                             num_tokens=num_tokens, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                             dropout=dropout, emb_dropout=emb_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'ssftt_lstm':
        model = SSFTTnet_LSTM(in_channels=in_channels, num_classes=out_classes,
                             num_tokens=num_tokens, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                             dropout=dropout, emb_dropout=emb_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'ssftt_fusion':
        model = SSFTTFusion(in_channels=224, num_classes=out_classes,
                           num_tokens=num_tokens, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                           dropout=dropout, emb_dropout=emb_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'ssftt_fusion_deep':
        model = SSFTTFusionDeep(in_channels=224, num_classes=out_classes,
                               num_tokens=num_tokens, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                               dropout=dropout, emb_dropout=emb_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    else:
        raise ValueError("Invalid model type. Choose 'cnn1d', 'cnn2d', 'cnn3d', 'unet', 'cnn_lstm', 'lstm_cnn', 'rgb_vit', 'rgb_resnet', 'rgb_hybrid', 'fusion_3dcnn', 'fusion_2plus1d', 'fusion_hybrid', 'fusion_attention', 'ssftt_fusion', or 'ssftt_fusion_deep'.")

    return model
