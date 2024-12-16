import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from torchvision.ops import DeformConv2d


class MorphicConvolution(nn.Module):
    def __init__(self, dim_in, dim_out, k, deform_k=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim_in, dim_out, kernel_size=k,dilation=dilation, padding='same', groups=dim_in) 
        self.pwconv = nn.Conv2d(dim_out, 2* deform_k * deform_k, kernel_size=1)  
        self.norm_layer = nn.BatchNorm2d(dim_in)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.deform_conv = DeformConv2d(dim_in, dim_out, kernel_size=deform_k, padding=1)
    def forward(self, x):
        residual = x 
        x_dw = self.norm_layer(self.dwconv(x))
        x_offset = self.pwconv(x_dw)
        x_deform = self.deform_conv(x, x_offset)
        out = x_deform + x_dw + residual * self.skip_scale
        return out
        
            
        
class MorphoSpectralHarmonicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.fourier = SpectralAttention(input_dim,input_dim)
        
        self.morpho_conv1 = MorphicConvolution(input_dim//4, input_dim//4, k=3)
        self.morpho_conv2 = MorphicConvolution(input_dim//4, input_dim//4, k=5)
        self.morpho_conv3 = MorphicConvolution(input_dim//4, input_dim//4, k=7)
        self.morpho_conv4 = MorphicConvolution(input_dim//4, input_dim//4, k=9)

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        
        
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_fourier = self.fourier(x_norm.transpose(-1, -2).reshape(B, self.input_dim, *img_dims))

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        x1 = x1.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        x2 = x2.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        x3 = x3.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        x4 = x4.transpose(-1, -2).reshape(B, self.input_dim // 4, *img_dims)
        
        x_morpho1 = self.morpho_conv1(x1)
        x_morpho2 = self.morpho_conv2(x2)
        x_morpho3 = self.morpho_conv3(x3)
        x_morpho4 = self.morpho_conv4(x4)
       

        x_morpho = torch.cat([x_morpho1, x_morpho2, x_morpho3, x_morpho4], dim=1)

        x_morpho = x_morpho.reshape(B, -1, n_tokens).transpose(-1, -2)
        x_fourier = x_fourier.reshape(B, -1, n_tokens).transpose(-1, -2)
        x_conbined = self.norm(x_morpho+x_fourier)
        x_conbined = self.proj(x_conbined )
        out = x_conbined .transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        
        return out 


class DBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DBAM, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.channel_att_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(in_channels // 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels // 2, bias=False),
        )
        self.channel_att_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Linear(in_channels // 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels // 2, bias=False),
        )

        self.spatial_att2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.alpha= nn.Parameter(torch.ones(1))

        self.dw_conv = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, groups=in_channels // 2, padding=1, bias=False)

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)

        #CAB
        avg_ca1 = self.channel_att_avg[0](x1).view(x1.size(0), -1)  # Avg pooling
        max_ca1 = self.channel_att_max[0](x1).view(x1.size(0), -1)  # Max pooling

        avg_ca1 = self.channel_att_avg[1:](avg_ca1).view(x1.size(0), x1.size(1), 1, 1)
        max_ca1 = self.channel_att_max[1:](max_ca1).view(x1.size(0), x1.size(1), 1, 1)

        ca1 = avg_ca1 + max_ca1
        ca1 = torch.sigmoid(ca1)
        cx1 = self.alpha * x1 * ca1  

        #SAB
        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)
        sa2 = self.spatial_att2(torch.cat([avg_out2, max_out2], dim=1))
        sx2 = self.alpha * x2 * sa2
        #FEB
        x1_enhanced = F.gelu(self.dw_conv(x1))
        x2_enhanced = F.gelu(self.dw_conv(x2))

        x1 = cx1 + x1_enhanced
        x2 = sx2 + x2_enhanced

        x = torch.cat((x1, x2), dim=1)

        x = self.channel_shuffle(x, 2)

        return x

class SpectralAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpectralAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        fft_real = fft.real
        fft_imag = fft.imag
        fft_real = self.conv(fft_real)
        fft_imag = self.conv(fft_imag)
        fft_complex = torch.complex(fft_real, fft_imag)
        ifft = torch.fft.irfft2(fft_complex, s=x.shape[-2:], dim=(-2, -1), norm="ortho")
        return self.bn(ifft)

class PinNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64],
                bridge=True,):
        super().__init__()

        self.bridge = bridge
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),

        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            MorphoSpectralHarmonicLayer(input_dim=c_list[2], output_dim=c_list[3]),
            #nn.Conv2d(c_list[2], c_list[3], 1),
        )
        self.encoder5 = nn.Sequential(
            MorphoSpectralHarmonicLayer(input_dim=c_list[3], output_dim=c_list[4]),
            #nn.Conv2d(c_list[3], c_list[4],1),
        )
        self.encoder6 = nn.Sequential(
            MorphoSpectralHarmonicLayer(input_dim=c_list[4], output_dim=c_list[5]),
            #nn.Conv2d(c_list[4], c_list[5],1),
        )

        if bridge: 
            self.ab1 = DBAM(c_list[0],reduction=4)
            self.ab2 = DBAM(c_list[1],reduction=8)
            self.ab3 = DBAM(c_list[2],reduction=16)
            self.ab4 = DBAM(c_list[3],reduction=16)
            self.ab5 = DBAM(c_list[4],reduction=32)
            print('Att_Bridge was used')
        self.decoder1 = nn.Sequential(
            MorphoSpectralHarmonicLayer(input_dim=c_list[5], output_dim=c_list[4]),
            #nn.Conv2d(c_list[5], c_list[4],1),
        )
        self.decoder2 = nn.Sequential(
            MorphoSpectralHarmonicLayer(input_dim=c_list[4], output_dim=c_list[3]),
            #nn.Conv2d(c_list[4], c_list[3],1),
        )
        self.decoder3 = nn.Sequential(
            MorphoSpectralHarmonicLayer(input_dim=c_list[3], output_dim=c_list[2]),
            #nn.Conv2d(c_list[3], c_list[2], 1),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),

        )
        
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)
        


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2
        
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4
        

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8
        

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16
        

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32
        
        
        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        
            
        
        if self.bridge: 
            t1= self.ab1(t1)
            t2= self.ab2(t2)
            t3= self.ab3(t3)
            t4= self.ab4(t4)
            t5= self.ab5(t5)
        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)





from thop import profile
# 创建一个简单模型实例
model = PinNet()
# 使用thop计算FLOPs和参数量
input = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(input, ))

# 输出结果
print(f"Total parameters: {params / 1e6:.6f}M")
print(f"Total FLOPs: {flops / 1e9:.6f}G")

