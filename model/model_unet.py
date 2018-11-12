from model_helper import *
from misc import SCSEBlock

class UNet11(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super(UNet11,self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.num_classes = num_classes

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[3],
                                   self.relu)

        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            self.encoder[8],
            self.relu,
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            self.encoder[13],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.relu,
        )

        self.center = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv=True)
        self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv=True)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=True)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig


class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super(UNet16,self).__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlockLinkNet,self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super(LinkNet34, self).__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig


class Conv3BN(nn.Module):
    def __init__(self, in_, out, bn=False):
        super(Conv3BN).__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_, out):
        super(UNetModule).__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class UNet(nn.Module):
    """
    Vanilla UNet.

    Implementation from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """
    output_downscaled = 1
    module = UNetModule

    def __init__(self,
                 input_channels = 3,
                 filters_base = 32,
                 down_filter_factors=(1, 2, 4, 8, 16),
                 up_filter_factors=(1, 2, 4, 8, 16),
                 bottom_s=4,
                 num_classes=1,
                 add_output=True):
        super(UNet).__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(self.module(
                down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i]))
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))

        if self.add_output:
            x_out = self.conv_final(x_out)
            if self.num_classes > 1:
                x_out = F.log_softmax(x_out, dim=1)
        return x_out


class AlbuNet(nn.Module):
    def __init__(self, num_classes=1, num_filters=32,
                 pretrained=False, is_deconv=False,
                 is_Refine = False, is_Freeze = False,
                 is_SCSEBlock = False, norm_type = None):

        super(AlbuNet, self).__init__()
        self.num_classes = num_classes
        self.is_refine = is_Refine
        self.is_SE = is_SCSEBlock

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        if is_Freeze:
            print('Freeze!!!!!!!!!!!!!!!!!!!!')
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        if self.is_SE:
            # self.center_se = SCSEBlock(512, reduction=1)
            self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)

            self.dec5_se = SCSEBlock(512 + num_filters * 8, reduction=1)
            self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)

            self.dec4_se = SCSEBlock(256 + num_filters * 8, reduction=1)
            self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)

            self.dec3_se = SCSEBlock(128 + num_filters * 8, reduction=1)
            self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv, norm_type)

            self.dec2_se = SCSEBlock(64 + num_filters * 2, reduction=1)
            self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv, norm_type)

            self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv, norm_type)
            self.dec0 = ConvRelu(num_filters, num_filters)
        else:
            self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
            self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
            self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
            self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv, norm_type)
            self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv, norm_type)
            self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv, norm_type)
            self.dec0 = ConvRelu(num_filters, num_filters)

        if self.is_refine:
            self.refine_net = RefineNet(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)


    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32 8

        center = self.center(self.pool(conv5))

        if self.is_SE:
            dec5_se = self.dec5_se(torch.cat([center, conv5], 1))
            dec5 = self.dec5(dec5_se)
            dec4_se = self.dec4_se(torch.cat([dec5, conv4], 1))
            dec4 = self.dec4(dec4_se)
            dec3_se = self.dec3_se(torch.cat([dec4, conv3], 1))
            dec3 = self.dec3(dec3_se)
            dec2_se = self.dec2_se(torch.cat([dec3, conv2], 1))
            dec2 = self.dec2(dec2_se)
        else:
            dec5 = self.dec5(torch.cat([center, conv5], 1))
            dec4 = self.dec4(torch.cat([dec5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))

        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.is_refine:
            dec0 = self.refine_net(dec0)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig


class AlbuNet50(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False, is_Freeze = False, norm_type = None):
        super(AlbuNet50, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)

        resnet = models.resnet50(pretrained=True)

        if is_Freeze:
            print('Freeze!!!!!!!!!!!!!!!!!!!!')
            for param in resnet.parameters():
                param.requires_grad = False


        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = DecoderBlock(512*4, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)

        self.dec5 = DecoderBlock(512*4 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
        self.dec4 = DecoderBlock(256*4 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
        self.dec3 = DecoderBlock(128*4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv, norm_type)
        self.dec2 = DecoderBlock(64*4 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv, norm_type)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv, norm_type)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)


    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        conv2 = self.encoder1(x)
        conv3 = self.encoder2(conv2)
        conv4 = self.encoder3(conv3)
        conv5 = self.encoder4(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig


class AlbuNet101(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False, is_Freeze=False, norm_type = None):
        super(AlbuNet101, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)

        resnet = models.resnet101(pretrained=True)
        if is_Freeze:
            print('Freeze!!!!!!!!!!!!!!!!!!!!')
            for param in resnet.parameters():
                param.requires_grad = False

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = DecoderBlock(512*4, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)

        self.dec5 = DecoderBlock(512*4 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
        self.dec4 = DecoderBlock(256*4 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
        self.dec3 = DecoderBlock(128*4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv, norm_type)
        self.dec2 = DecoderBlock(64*4 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv, norm_type)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv, norm_type)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)



    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        conv2 = self.encoder1(x)
        conv3 = self.encoder2(conv2)
        conv4 = self.encoder3(conv3)
        conv5 = self.encoder4(conv4)

        # Center
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig


class RefineNet(nn.Module):
    def __init__(self, channel, out):
        super(RefineNet, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate5 = nn.Conv2d(channel, out, kernel_size=3, dilation=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.activation(self.dilate1(x))
        dilate2_out = self.activation(self.dilate2(dilate1_out))
        dilate3_out = self.activation(self.dilate3(dilate2_out))
        dilate4_out = self.activation(self.dilate4(dilate3_out))
        dilate5_out = self.activation(self.dilate5(dilate4_out))
        return dilate5_out
