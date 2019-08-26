from pathlib import Path
import numpy as np
from PIL import Image
from keras.models import load_model


model_path = 'logdir_bekos/model_file.hdf5'
images_folder = 'bekos/predict'

# load model
model = load_model(model_path)
image_shape = (32, 32, 3)


# load images
def crop_resize(image_path):
    image = Image.open(image_path)
    length = min(image.size)
    crop = image.crop((0, 0, length, length))
    resized = crop.resize(image_shape[:2])  # use width x height
    img = np.array(resized).astype('float32')
    img /= 255
    return img


folder = Path(images_folder)
image_paths = [str(f) for f in folder.glob('*.jpg')]
images = [crop_resize(p) for p in image_paths]
images = np.asarray(images)

predicted = model.predict_classes(images)

# 正解表示用の辞書
dic = {
    '0':'ベコサス',
    '1':'キマイラベコ',
    '2':'ケルベコス',
    '3':'ミノタウベコ',
    '4':'ユニベコーン'
}

# 結果表示
ok = 0
for i , item in enumerate(predicted):
    beko_name = dic[str(item)]
    print(image_paths[i] + 'は' + beko_name + '!')
    if beko_name in image_paths[i]:
        ok += 1

print('正解率：{}/{}({}%)'.format(ok, len(image_paths), round(ok / len(image_paths) * 100)))
print('当たってた？')