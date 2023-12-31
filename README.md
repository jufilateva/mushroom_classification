# mushroom_classification

Программа написана для использования в colab.research.google.com

Суть программы - классификация грибов на основе двух изображений - с ракурсов "сбоку" и "сверху". Изображения с каждого ракурса распознаются отдельной сверточной нейросетью. 

Данные выгружаются с Google Drive, по 3600 изображений для каждого ракурса. Всего классов 15, количество фотографий в классах одинаковое. Тренировочный датасет составляет 90% от общего, тестовый - 10%.

Для распознавания фотографий из датасета ракурса "сверху" использовалась предобученная модель ResNet18, дообученная на собранных изображениях. Для датасета ракурса "сбоку" - предобученная модель ResNet50, так же дообученная на собранных изображениях.

Для обучения и валидации моделей использовалась k-блочная кросс-валидация (количество блоков - 5). За обучение отвечают функции train_up (для датасета "сверху") и train_slide (для датасета "сбоку"). 

За тестирование отвечает функция test. В ней модели одновременно определяют два изображения гриба одного класса, выдавая в результате своей работы Softmax. Результаты х работы складываются, вес результатов для каждой модели умножается на 0.5 (веса 0.5 и 0.5 подобраны экспериментально, при таких весах модель выдает наибольшую точность распознавания на тестовых данных). Accuracy считается как процент изображений, для которых модели верно определили класс. 

https://colab.research.google.com/drive/17hVY_S-VDr5o7NLqTh46Jmhv1wSuucwe
