import os
from tqdm import tqdm
import tensorflow as tf

from tools.data_provider import DataProvider
from tools.preprocessors import ImageReader
from tools.transformers import ImageResizer, LabelIndexer, LabelPadding
from tools.ctc_loss import CTCloss
from tools.metrics import CWERMetric

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import define_model
import tensorflow as tf
#gpus = tf.config.list_physical_devices('GPU')
#gpu = gpus[0]

#tf.config.experimental.set_memory_growth(gpu, True)

data_path = "C:/Users/HrHoe/Desktop/mnt/ramdisk/max/90kDICT32px"
val_annotation_path = data_path + "/annotation_val.txt"
train_annotation_path = data_path + "/annotation_train.txt"

# Read metadata file and parse it
def read_annotation_file(annotation_path):
    dataset, vocab, max_len = [], set(), 0
    with open(annotation_path, "r") as f:
        for line in tqdm(f.readlines()):
            line = line.split()
            image_path = data_path + line[0][1:]
            label = line[0].split("_")[1]
            dataset.append([image_path, label])
            vocab.update(list(label))
            max_len = max(max_len, len(label))
    return dataset, sorted(vocab), max_len

train_dataset, train_vocab, max_train_len = read_annotation_file(train_annotation_path)
val_dataset, val_vocab, max_val_len = read_annotation_file(val_annotation_path)

# Save vocab and maximum text length
vocab = "".join(train_vocab)
max_text_length = max(max_train_len, max_val_len)

batch_size = 512
width = 128
height = 32

# Create training data provider
train_data_provider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=batch_size,
    data_preprocessors=[ImageReader()],
    transformers=[
        ImageResizer(width, height),
        LabelIndexer(vocab),
        LabelPadding(max_word_length=max_text_length, padding_value=len(vocab))
        ],
)

# Create validation data provider
val_data_provider = DataProvider(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=batch_size,
    data_preprocessors=[ImageReader()],
    transformers=[
        ImageResizer(width, height),
        LabelIndexer(vocab),
        LabelPadding(max_word_length=max_text_length, padding_value=len(vocab))
        ],
)

model = define_model(
    input_dim = (height, width, 3),
    output_dim = len(vocab),
)

# Compile the model
learning_rate = 1e-4

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
    loss=CTCloss(), 
    metrics=[CWERMetric(len(vocab))],
    run_eagerly=False
)

# Define path to save the model
model_path = '../api/ai_ocr1'
os.makedirs(model_path, exist_ok=True)

# Define callbacks
earlystopper = EarlyStopping(monitor='val_CER', patience=10, verbose=1)
checkpoint = ModelCheckpoint(f"{model_path}/model.h5", monitor='val_CER', verbose=1, save_best_only=True, mode='min')
reduceLROnPlat = ReduceLROnPlateau(monitor='val_CER', factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode='auto')

train_epochs = 1
train_workers = 20

model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=train_epochs,
    callbacks=[earlystopper, checkpoint, reduceLROnPlat],
    workers=train_workers
)

model.save('../api/ocr_ai')
model.save_weights('../api/ocr_ai/model_weights.h5')