## 1.MODNet原理学习

MODNet网络用来预测显著性区域（可以简单理解为人像，但不局限于人像，比如动物前景也没问题），以便从图像中分离出前后景。

![](https://img.simoniu.com/基于MODNet实现证件照更换背景颜色01.jpg)


MODNet 本质上是一个卷积神经网络（Convolutional Neural Network, CNN），MODNet 是一个专门为实时人像抠图任务设计和优化的卷积神经网络（CNN）架构。它巧妙地利用了多分支CNN结构、特征融合技术和注意力机制（在CNN框架内）来实现高精度和高效率的抠图效果。因此，将其归类为CNN是准确的。可以说它是CNN在特定任务（实时人像抠图）上的一个成功应用实例和架构创新。

### 主要有以下几块组成

1. 低分辨率分支（Semantic Estimation），通过对图片的卷积缩小图像尺寸，预测大致的语义信息，即图像中哪些区域是人物。为后续的边缘细节预测以及融合提供上下文。该分支还有一个e-ASPP辅助模块，它是一个空洞空间金字塔池化结构，用于捕捉不同尺度上的语义信息，有助于处理图像人物不同部分的尺度变化
2. 高分辨率分支（Deatil Prediction），专注于预测图像中的细节，特别是边缘区域，一边产生更精准的分割效果。它利用了低分辨率分支的输出，对该输出进行上采样（即增加分辨率）以及结合下采样后的原始图像来获取更清晰的边缘分割。该分支也有一个辅助模块Skp Link，它将网络早期层的特征传递到后面的层，因为早期层通常包含更多的原始信息，这样做更有助于恢复细节。
3. 融合分支（Semantic-Detail Fusion），结合了低分辨率分支和高分辨率分支，通过融合语义信息以及边缘细节信息来提高分割的准确度。
4. 输出：上面三个分支分别输出：语义sp(表示图像中人像区域)、细节dp（表示人像的精细边缘）以及融合alpha matter alpha_p（显示了任务和背景明确的分离）。
5. 后处理：transition region md，表示分割过程中可能会出现的过度区域，例如头发的边缘。

###  网络结构代码实现

```python
class MODNet(nn.Module):
    """ MODNet架构
    """
    #模型初始化
    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=True):
        super(MODNet, self).__init__()


        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained


        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)


        #初始化 低分辨率分支
        self.lr_branch = LRBranch(self.backbone)
        #初始化 高分辨率分支
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        #初始化 融合分支
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)


        #加载预训练模型
        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()                


    #前向传播
    def forward(self, img, inference):
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(img, inference)
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)

        #需要的就是融合分支处理后的结果 即pred_matte
        return pred_semantic, pred_detail, pred_matte
```

## 2.基于MODNet实现证件照更换背景颜色案例

代码实现：

```python
import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src.models.modnet import MODNet
import numpy as np
import cv2

color_classes = ['red', 'green', 'blue', 'white']
#  预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
])

modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

#加载模型
ckpt_path = "./pretrained/modnet_photographic_portrait_matting.ckpt"

# 加载模型
if torch.cuda.is_available():
    modnet = modnet.cuda()
    weights = torch.load(ckpt_path)
else:
    weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
modnet.load_state_dict(weights)
modnet.eval()

#  更换颜色
def change_color(srcImg, color):
    # 这里做什么？
    #print(srcImg)
    print("要更换的颜色：", color)
    # 读取并预处理图像
    # original_image = cv2.imread("./images/personal02.jpg")
    # original_image = cv2.imread(srcImg)
    original_image = srcImg.astype(np.uint8)
    original_height, original_width = original_image.shape[:2]  # 获取原始高宽

    # 预处理流程（缩放到512x512）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # 转为RGB
    # image = Resize((512, 512))(ToTensor()(image)).unsqueeze(0)  # 调整尺寸并转为Tensor
    image = transform(image).unsqueeze(0)

    # 推理
    with torch.no_grad():
        _, _, matte = modnet(image, True)

    # 后处理：生成Alpha遮罩
    matte = matte.squeeze().cpu().numpy()
    matte = cv2.resize(matte, (image.shape[3], image.shape[2]))  # 恢复原始尺寸

    # 替换背景颜色（例如：红色背景）
    new_background = np.zeros_like(image.squeeze().permute(1, 2, 0).cpu().numpy())
    match color:
        case "red":
            # 换成红色背景
            new_background[:, :] = [208, 0, 0]
        case "blue":
            # 换成蓝色背景
            new_background[:, :] = [0, 143, 213]
        case "green":
            # 换成绿色背景
            new_background[:, :] = [0, 134, 134]
        case "white":
            # 换成白色背景
            new_background[:, :] = [255, 255, 255]

    # 合成新图像
    foreground = srcImg.astype(np.uint8)
    foreground = cv2.resize(foreground, (512, 512))
    alpha = np.repeat(matte[:, :, np.newaxis], 3, axis=2)
    result = alpha * foreground + (1 - alpha) * new_background

    # 保存结果
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    result = cv2.resize(result, (original_width, original_height))
    cv2.imwrite("result.jpg", result)

    result_img = './result.jpg'
    return result_img


demo = gr.Interface(
    fn=change_color,
    title='OpenCV实现更换证件照背景测案例',
    inputs=[gr.Image(label='源图片'), gr.Dropdown(color_classes, value='red', label='背景颜色')],
    outputs=[gr.Image(show_label=False)],
    examples=[['./images/personal01.jpg', 'blue'], ['./images/personal02.jpg', 'red'],
              ['./images/personal04.jpg', 'white']]
)

if __name__ == "__main__":
    # 定义端口号
    gradio_port = 8080
    gradio_url = f"http://127.0.0.1:{gradio_port}"
    demo.launch(
        server_name="127.0.0.1",
        server_port=gradio_port,
        debug=True,
        #auth=("admin", "123456"),
        #auth_message="请输入账号信息访问此应用。测试账号：admin,密码：123456",
        #inbrowser=False,
        #prevent_thread_lock=True,
        #share=True
    )
```

模型下载地址：
|地址|提取码|
|-|-|
|https://pan.baidu.com/s/1JK4bkVD8i5OBTvkP0UmTUg?pwd=9527|9527 |

运行效果：
![](https://img.simoniu.com/基于MODNet实现证件照更换背景颜色02.jpg)




