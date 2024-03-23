# TODO 检测使用PIL

from PIL import Image
import torchvision.transforms as transforms

image_path = "C:/Users/jiawenduo/Pictures/Camera Roll/th.jpg"

image = Image.open(image_path)
# image.show()

print(image.size)

# 调整image的大小
image = image.resize((500, 500))
# image.show()

# 图像的保存
image.save("../data/test/demo1.jpg")

# 将PIL图像转换为tensor张量
transform = transforms.ToTensor()
tensor_image = transform(image)

print(tensor_image)
print(tensor_image.shape)
