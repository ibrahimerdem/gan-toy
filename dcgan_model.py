import os
import threading
import time
import numpy as np
import tensorflow as tf


class Dcgan(threading.Thread):
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 16
    NOISE_DIM = 100
    N_EXAMPLE = 4

    def __init__(self, reporter):
        super().__init__()
        if not reporter:
            pass
        else:
            self.data_array = []
            self.train_data = []
            self.generator = self.make_generator_model()
            self.discriminator = self.make_discriminator_model()
            self.ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.go = tf.keras.optimizers.Adam(1e-4)
            self.do = tf.keras.optimizers.Adam(1e-4)
            self.checkpoint_dir = os.getcwd() + "./ImageWorkLogs/training_checkpoints"
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.go,
                                                  discriminator_optimizer=self.do,
                                                  generator=self.generator,
                                                  discriminator=self.discriminator)
            self.pause = 0
            self.reporter = reporter

    @staticmethod
    def resize(input_image, height=256, width=256):
        input_image = tf.image.resize(
            input_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image

    def random_crop(self, input_image):
        stacked_image = tf.stack([input_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[1, self.IMG_WIDTH, self.IMG_HEIGHT, 3])

        return cropped_image[0]

    @staticmethod
    def normalize(input_image):
        input_image = (input_image / 127.5) - 1

        return input_image

    @tf.function()
    def random_jitter(self, input_image):
        input_image = self.resize(input_image, self.IMG_WIDTH+30, self.IMG_HEIGHT+30)
        input_image = self.random_crop(input_image)

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)

        return input_image

    def loader_train(self, input_image):
        input_image = self.random_crop(input_image)
        input_image = self.normalize(input_image)

        return input_image

    def make_dataset(self, arr):
        sample_arr = []
        for ind, i in enumerate(arr):
            out = tf.io.read_file(i)
            out = tf.io.decode_png(out, channels=3)
            out = tf.image.resize(out, [self.IMG_WIDTH, self.IMG_HEIGHT])
            self.data_array.append(out.numpy())
            sample_arr.append(i)

        self.train_data = tf.data.Dataset.from_tensor_slices(np.array(self.data_array))
        self.train_data = self.train_data.map(self.loader_train,
                                              num_parallel_calls=tf.data.AUTOTUNE)
        self.train_data = self.train_data.shuffle(len(arr))
        self.train_data = self.train_data.batch(self.BATCH_SIZE)

        text = "Dataset Created"
        report_data = dict(text=text, sample_arr=sample_arr)
        self.reporter.put(report_data)

    @staticmethod
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((8, 8, 256)))

        model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                                  padding="same", use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2),
                                                  padding="same", use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                                  padding="same", use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2),
                                                  padding="same", use_bias=False,
                                                  activation="tanh"))
        model.summary()
        return model

    @staticmethod
    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same",
                                         input_shape=[128, 128, 3]))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.ce(tf.ones_like(real_output), real_output)
        fake_loss = self.ce(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.ce(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.go.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.do.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def restore_model(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        text = "Model restored"
        report_data = dict(text=text)
        self.reporter.put(report_data)
        print(text)
        seed = tf.random.normal([self.N_EXAMPLE, self.NOISE_DIM])
        predictions = self.generator(seed, training=False)
        img_list = []
        for i in range(predictions.shape[0]):
            img_list.append(np.uint8((predictions[i, :, :, :] * 127.5 + 127.5).numpy()))

        print(self.generator.losses)

    def run(self):
        for root, dirs, files in os.walk(os.getcwd() + "./ImageWorkLogs", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        text = "Modeling Started"
        report_data = dict(text=text)
        self.reporter.put(report_data)

        epochs = 20000
        for epoch in range(epochs):
            start = time.time()
            seed = tf.random.normal([self.N_EXAMPLE, self.NOISE_DIM])
            if self.pause:
                text = "Training interrupted in {}. epoch".format(epoch + 1)
                report_data = dict(text=text)
                self.reporter.put(report_data)
                print(text)
                break

            for image_batch in self.train_data:
                self.train_step(image_batch)

            predictions = self.generator(seed, training=False)
            img_list = []
            for i in range(predictions.shape[0]):
                img_list.append(np.uint8((predictions[i, :, :, :] * 127.5 + 127.5).numpy()))

            if (epoch + 1) % 50 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                text = "Checkpoint created"
                report_data = dict(text=text)
                self.reporter.put(report_data)
                print(text)

            text = "Time for epoch {} is {} sec".format(epoch + 1, time.time() - start)
            report_data = dict(text=text, image_list=img_list)
            self.reporter.put(report_data)
            print(text)

        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        text = "Checkpoint created"
        report_data = dict(text=text)
        self.reporter.put(report_data)
        print(text)
        seed = tf.random.normal([self.N_EXAMPLE, self.NOISE_DIM])
        predictions = self.generator(seed, training=False)
        img_list = []
        for i in range(predictions.shape[0]):
            tf.keras.utils.save_img(
                os.getcwd() + "\\ImageWorkLogs\\output"+str(i)+".png", predictions[i, :, :, :] * 127.5 + 127.5)
            img_list.append(np.uint8((predictions[i, :, :, :] * 127.5 + 127.5).numpy()))
        text = "Output saved"
        report_data = dict(text=text, image_list=img_list)
        self.reporter.put(report_data)
        print(text)
        text = "Training ended with {} epochs".format(epoch+1)
        report_data = dict(text=text)
        self.reporter.put(report_data)
        print(text)




