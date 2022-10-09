from io import BytesIO
import torch
from fast_style_transfer import TransNet
import pywebio
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def app():
    file_info = pywebio.input.file_upload(
        label='Choose your picture',
        accept=['.jpg', '.png'],
        placeholder='请上传jpg或png类型的图片'
    )

    img_content = file_info['content']
    pil_img = Image.open(BytesIO(img_content))
    pywebio.output.put_image(pil_img)
    action = pywebio.input.actions(label='是否开始生成图片？', buttons=['confirm', 'cancel'])
    if action == 'cancel':
        exit(0)

    g_net = TransNet()
    model_data = torch.load("./fst.pth")
    g_net.load_state_dict(model_data)

    image = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    image = cv2.resize(image, (512, 512))
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)

    pywebio.output.put_text("******正在生成图片******")

    demo = g_net(image)[0].detach()
    demo = demo.numpy()
    demo = np.transpose(demo, (1,2,0))
    demo = cv2.resize(demo, dsize=(width, height))
    demo *= 255.0
    
    img = demo.astype(np.uint8)
    # print(np.average(demo), np.min(demo), np.max(demo))
    # plt.imshow(demo)
    # plt.show()

    img = Image.fromarray(img)
    pywebio.output.put_image(img)


if __name__ == "__main__":
    app()