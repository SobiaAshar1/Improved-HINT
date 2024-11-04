import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from tabulate import tabulate

# Reuse the MPDModule and SCAL classes
class MPDModule(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(MPDModule, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')
        self.pixel_shuffle = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        return x

class SCAL(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SCAL, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=filters)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.feed_forward1 = tf.keras.layers.Conv2D(filters, kernel_size=1)
        self.feed_forward2 = tf.keras.layers.Conv2D(filters, kernel_size=1)
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        x1 = self.conv(x)
        x2 = self.feed_forward1(x1)
        
        # Add a channel dimension to x2 for MultiHeadAttention
        x1_reshaped = tf.reshape(x1, [tf.shape(x1)[0], -1, tf.shape(x1)[-1]])
        x2_reshaped = tf.reshape(x2, [tf.shape(x2)[0], -1, tf.shape(x2)[-1]])
        
        attention_output = self.attention(query=x1_reshaped, value=x2_reshaped, key=x2_reshaped)
        
        # Reshape back to original dimensions
        attention_output = tf.reshape(attention_output, tf.shape(x1))
        
        x3 = self.feed_forward2(attention_output)
        x = self.layer_norm2(x + x3)
        return x

# Custom CNN model
def build_custom_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')  # Changed to 6 classes
    ])
    return model

def build_hyper_custom_cnn_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
            activation='relu',
            input_shape=(128, 128, 3)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(
            filters=hp.Int('conv_2_filter', min_value=64, max_value=256, step=64),
            kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=hp.Int('dense_units', min_value=128, max_value=512, step=128),
            activation='relu'
        ),
        tf.keras.layers.Dense(6, activation='softmax')  # Changed to 6 classes
    ])
    
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class CustomCNNTrainer:
    def __init__(self, model, batch_size=32, img_height=128, img_width=128):
        self.model = model
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.train_dataset, self.val_dataset = self.load_datasets()
        self.history = {}

    def load_datasets(self):
        # Load and preprocess the dataset
        dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
        data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)
        
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )

        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )
        
        return train_dataset, val_dataset

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=10):
        history = self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.val_dataset)
        self.history['train'] = history.history

    def evaluate_model(self):
        val_loss, val_accuracy = self.model.evaluate(self.val_dataset)
        print(f'Validation Loss: {val_loss}')
        print(f'Validation Accuracy: {val_accuracy}')
    
    def plot_history(self):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train']['loss'], label='Train Loss')
        plt.plot(self.history['train']['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train']['accuracy'], label='Train Accuracy')
        plt.plot(self.history['train']['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def print_history_table(self):
        # Create a table for the history
        table = []
        headers = ['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy']
        for i in range(len(self.history['train']['loss'])):
            row = [
                i+1,
                self.history['train']['loss'][i],
                self.history['train']['accuracy'][i],
                self.history['train']['val_loss'][i],
                self.history['train']['val_accuracy'][i]
            ]
            table.append(row)
        print(tabulate(table, headers, tablefmt='pretty'))

    def plot_pie_chart(self):
        last_epoch = len(self.history['train']['loss']) - 1
        labels = ['Train Accuracy', 'Validation Accuracy']
        sizes = [self.history['train']['accuracy'][last_epoch], self.history['train']['val_accuracy'][last_epoch]]
        colors = ['lightcoral', 'lightskyblue']
        explode = (0.1, 0)  # explode 1st slice

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Accuracy Distribution at Last Epoch')
        plt.show()

    def plot_bar_chart(self):
        epochs = list(range(1, len(self.history['train']['accuracy']) + 1))
        train_accuracies = self.history['train']['accuracy']
        val_accuracies = self.history['train']['val_accuracy']

        bar_width = 0.35
        index = range(len(epochs))

        plt.figure(figsize=(14, 7))
        plt.bar(index, train_accuracies, bar_width, label='Train Accuracy', alpha=0.8, color='blue')
        plt.bar([i + bar_width for i in index], val_accuracies, bar_width, label='Validation Accuracy', alpha=0.8, color='orange')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy per Epoch')
        plt.xticks([i + bar_width / 2 for i in index], epochs)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def train_model_with_tuning(self, max_epochs=10):
        tuner = kt.RandomSearch(
            build_hyper_custom_cnn_model,
            objective='val_accuracy',
            max_trials=5,
            seed=42,
            directory='hyperparam_tuning',
            project_name='custom_cnn'
        )
        tuner.search(self.train_dataset, epochs=max_epochs, validation_data=self.val_dataset)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps.values}")
        
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(self.train_dataset, epochs=max_epochs, validation_data=self.val_dataset)
        self.history['train_tuned'] = history.history
        return model, history

# Example usage
if __name__ == "__main__":
    custom_cnn_model = build_custom_cnn_model()
    trainer = CustomCNNTrainer(model=custom_cnn_model, batch_size=32)
    trainer.compile_model()
    trainer.train_model(epochs=10)
    trainer.evaluate_model()
    trainer.plot_history()
    trainer.print_history_table()
    trainer.plot_pie_chart()
    trainer.plot_bar_chart()  # New method to plot bar chart

    print("\nHyperparameter Tuning for Custom CNN Model:")
    hyper_custom_model, history_hyper_custom = trainer.train_model_with_tuning(max_epochs=10)
    print("\nHyper-tuned Custom CNN Model Evaluation:")
    trainer.model = hyper_custom_model  # Update trainer with the best model
    trainer.evaluate_model()
    trainer.plot_history()
    trainer.print_history_table()
    trainer.plot_pie_chart()
    trainer.plot_bar_chart()  # New method to plot bar chart
