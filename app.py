from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
from fastai.vision.all import *
from translate import Translator
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Загрузка обученной модели
model_path = 'models/res18-unfine.pth'
state = torch.load(model_path, map_location='cpu')
model = state['model']

# Получение пути к вашим данным
path_to_data = 'poisonous_plants_dataset' 

# Создание DataLoader
item_tfms = Resize(460)
batch_tfms = [*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
dls = ImageDataLoaders.from_folder(path_to_data, bs=64, item_tfms=item_tfms, batch_tfms=batch_tfms)

# Инициализация Learner
learn = cnn_learner(dls, resnet18, metrics=accuracy)
learn.model.load_state_dict(state['model'])

# Явное преобразование модели к устройству (device)
device = torch.device('cpu')
learn.model = learn.model.to(device)

# Словарь с координатами для каждого растения
plant_locations = {
    "Visteria": {
        "lat": 51.1694, 
        "lng": 71.4491, 
        "description": "Вистерия – бұл ұзын гүл шоғыры бар ағаш тәрізді өсімдік. Оның көктемгі гүлденуі көптеген бағбандарды таң қалдырады."
    },
    "Алқаптың лалагүлі": {
        "lat": 43.2389, 
        "lng": 76.8897, 
        "description": "Тюльпан – көктемгі мезгілде гүлдейтін сүйкімді өсімдік. Олар әр түрлі түстерде болады және Казахстанда кең таралған."
    },
    "Диеффенбахия": {
        "lat": 50.2839, 
        "lng": 57.1669, 
        "description": "Диеффенбахия – ішкі мекенде өсірілетін танымал өсімдік. Оның жапырақтары үлкен және әдемі, бірақ улы, сондықтан абай болған жөн."
    },
    "Кастор майы зауыты": {
        "lat": 49.8075, 
        "lng": 73.0877, 
        "description": "Касторовое растение пайдаланылатын май өндіретін өсімдік. Оның даны улы, сондықтан өсімдікпен жұмыс істегенде сақ болу қажет."
    },
    "Лалагүлдер": {
        "lat": 44.8528, 
        "lng": 65.5092, 
        "description": "Роза – бақта өсірілетін ең танымал гүлдердің бірі. Олардың түрлері өте көп, ал гүлдері әртүрлі түсті болып келеді."
    },
    "Наперстянка": {
        "lat": 54.8740, 
        "lng": 69.1633, 
        "description": "Наперстянка – жүрек ауруларын емдеуде пайдаланылатын дәрілік өсімдік. Бірақ оның барлық бөліктері улы, сондықтан мұқият болу қажет."
    },
    "Олеандр": {
        "lat": 51.7250, 
        "lng": 75.3144, 
        "description": "Олеандр – қатты улы гүлді өсімдік. Оның гүлдері әсем болғанымен, улану қаупі жоғары."
    },
    "Ревень": {
        "lat": 52.2826, 
        "lng": 76.9547, 
        "description": "Ревень – асқа пайдаланылатын, көктемде өсетін өсімдік. Оның тамыры мен сабақтары пісірілген кезде ғана жейді."
    }
}


# Функция для предсказания объекта на изображении
def predict_image(image):
    img_np = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_pil = Image.fromarray(img_cv)

    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    pred, pred_idx, probs = learn.predict(img_pil)
    probability_percent = probs[pred_idx].item() * 100

    if probability_percent < 60:
        return ("Бұл өсімдік емес немесе анықталмады.", probability_percent, img_str, None, None)
    else:
        plant_name = pred.capitalize()
        translator = Translator(to_lang="kk")
        plant_name_translation = translator.translate(plant_name)
        location = plant_locations.get(plant_name_translation, {"lat": 0, "lng": 0, "description": "Сипаттама жоқ"})
        return (plant_name_translation, probability_percent, img_str, location, location['description'])


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_bytes = file.read()
            image = Image.open(BytesIO(image_bytes))
            plant_name_translation, probability_percent, img_str, location, description = predict_image(image)
            return render_template('result.html', plant_name=plant_name_translation, probability=probability_percent, image=img_str, location=location, description=description)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
