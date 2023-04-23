import time
from libraries import *
from loss_fns import dice_loss_fn, wce_loss_fn, calc_loss_weights
from model import SSA_Net
from dataset_loader import DataGenerator

class Trainer():
    
    def __init__(self, train_data, val_data, test_data, n_classes:int=1):
        self.n_classes = n_classes
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
    def __optimizer(self):
        # Default optimizer
        initial_learning_rate = 0.0001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        return optimizer
    
    def __metrics(self):
        # Default metrics
        train_acc_metric = tf.keras.metrics.MeanAbsoluteError()
        val_acc_metric = tf.keras.metrics.MeanAbsoluteError()
        return [train_acc_metric, val_acc_metric]
        
    def __training_loop(self, train_data:tuple, val_data:tuple,  model:keras.Model, epochs:int, loss_fn, name, optimizer:tf.keras.optimizers,  metrics:list, batch_size:int=8, has_sa:bool=True):
        # Some usefull data to be returned
        episode_start_time = time.time()
        episode_losses = []
        episode_train_metrics = []
        episode_val_metrics = []
        
        # Build datasets from input data
        train_gen = DataGenerator(*train_data)
        val_gen = DataGenerator(*val_data)
        
        # Set batchsize and metrics
        batch_size = batch_size
        train_acc_metric, val_acc_metric = metrics
        if self.n_classes != 1:
            weights = calc_loss_weights(train_data[1], class_nums=self.n_classes)
        else:
            weights = 0
        
        # Define optimizer
        optimizer = optimizer
        
        # Functions for each step of train and validation
        if self.n_classes == 1:
            @tf.function
            def train_step(x, y, weights=0):
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss_value = loss_fn(y, logits, sa=has_sa)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_acc_metric.update_state(y, logits[0])
                return loss_value
        else:
            @tf.function
            def train_step(x, y, weights):
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss_value = loss_fn(y, logits, weights, sa=has_sa)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_acc_metric.update_state(y, logits[0])
                return loss_value
        @tf.function
        def test_step(x, y):
            val_logits = model(x, training=False)
            val_acc_metric.update_state(y, val_logits[0])
        
        # Training loop
        best_train_acc = 1
        best_val_acc = 1

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            epoch_start_time = time.time()
            losses = []

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
                loss_value = train_step(x_batch_train, y_batch_train, weights)
                losses.append(loss_value)
                # Log every 5 batches.
                if step % 5 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * batch_size))

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            episode_train_metrics.append(train_acc)

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in val_gen:
                test_step(x_batch_val, y_batch_val)
                
            if epoch % 10 == 0:
                    model.save_weights(CWD + '/Weights/{}_Epoch{}.h5'.format(name, str(epoch)))
                    print('Model weights are saved. Name: {}'.format('Epoch' + str(epoch)))

            val_acc = val_acc_metric.result()
            episode_val_metrics.append(val_acc)
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Loss over epoch: %.4f" % (float(np.mean(losses)),))
            episode_losses.append(losses)
            print("Time taken: %.2fs" % (time.time() - epoch_start_time))
            
            # Save best model weights
            if (best_train_acc > train_acc):
                best_train_acc = train_acc
                model.save_weights(f'{CWD}/Weights/{name}_best_train.h5')
            if (best_val_acc > val_acc):
                best_val_acc = val_acc
                model.save_weights(f'{CWD}/Weights/{name}_best_val.h5')

            train_gen.on_epoch_end()
        return [episode_start_time, episode_losses, episode_train_metrics, episode_val_metrics, model]
            
    def train_simple_loop(self, epochs:int, batch_size:int, has_sc:bool=True, has_sa:bool=True):
        # Select loss function
        if self.n_classes == 1:
            loss_fn = dice_loss_fn
        else:
            loss_fn = wce_loss_fn
            
        # Build model
        model = SSA_Net(self.n_classes, has_sc, has_sa)
            
        episode_start_time, episode_losses, episode_train_metrics, episode_val_metrics, model = self.__training_loop(self.train_data, self.val_data, model, epochs, loss_fn, 'simple', self.__optimizer(),  self.__metrics(), batch_size, has_sa)
        
        print(f"---------------------------------------")
        print(f"Best training acc: {np.amin(episode_train_metrics):.4f}")
        print(f"Best validation acc: {np.amin(episode_val_metrics):.4f}")
        print(f"Min loss: {np.amin(episode_losses):.4f}")
        print(f"Mean loss: {np.mean(episode_losses):.4f}")
        print(f"Time taken : {(time.time() - episode_start_time):.2f}s")
        print(f"---------------------------------------")
        
    def train_ss_fs(self, episodes, epochs, unlabeld_data, random_sample, label_cutoff, batch_size:int, has_sc:bool=True, has_sa:bool=True):
        # Select loss function
        if self.n_classes == 1:
            loss_fn = dice_loss_fn
        else:
            loss_fn = wce_loss_fn
            
        train_data_t = copy.deepcopy(self.train_data)
        loaded_unlabeled = []
        for filename in unlabeld_data:
            loaded_unlabeled.append(np.expand_dims(np.load(CWD + '/' + filename), axis=0))

        loaded_unlabeled = np.concatenate(loaded_unlabeled, axis=0)
        print(loaded_unlabeled.shape)
        
        for episode in range(episodes):
            
            print("\nStart of episode %d" % (episode,))
            episode_start_time = time.time()
            
            # Build model
            model = SSA_Net(self.n_classes, has_sc, has_sa)
            
            # prepare data for this episode
            print(f"Number of initial train data: {len(self.train_data[0])}")
            print(f"Number of current train data: {len(train_data_t[1])}")
            # random.Random(1).shuffle(train_data_t[0])
            # random.Random(1).shuffle(train_data_t[1])
            x_train = train_data_t[0]
            y_train = train_data_t[1]
            
            # start training loop
            _, episode_losses, episode_train_metrics, episode_val_metrics, model = self.__training_loop((x_train, y_train, train_data_t[2]), self.val_data, model, epochs, loss_fn, episode, self.__optimizer(),  self.__metrics(), batch_size, has_sa)

            # generate pseudolabels
            print('Generating pseudolabels...')
            pseudolabels = model.predict(loaded_unlabeled, verbose=0)[0]
            
            # trust module
            pseudolabels[pseudolabels > label_cutoff] = 1
            pseudolabels[pseudolabels <= label_cutoff] = 0
            
            # check if new data is added to training data
            if random_sample * episode < len(unlabeld_data):
                message = 'New data with pseudo-labels added to training data.\n'
            else:
                message = 'No new data is added only new pseudo-labels.\n'

            train_data_t = [[], {}, self.train_data[2]]
            train_data_t[0].extend(self.train_data[0])
            train_data_t[0].extend(unlabeld_data[:(random_sample * (episode + 1))])
            train_data_t[1].update(self.train_data[1])
            for index, label in enumerate(pseudolabels[:(random_sample * (episode + 1))]):
                # create file name for label
                file_name = unlabeld_data[0].split('.')[0][:-1] + 'y.npy'
                np.save(CWD + '/' + file_name, label)
                train_data_t[1][unlabeld_data[index]] = file_name
                
            # select random pseudo-labels and add them to train data
            # del train_data_t
            # train_data_t = []
            # train_data_t.append(np.concatenate((self.train_data[0], loaded_unlabeled[:(random_sample * (episode + 1))]), axis=0))
            # train_data_t.append(np.concatenate((self.train_data[1], pseudolabels[:(random_sample * (episode + 1))]), axis=0))
            print(message)
            
            print(f"---------------------------------------")
            print(f"Best training acc in episode {episode}: {np.amin(episode_train_metrics):.4f}")
            print(f"Best validation acc in episode {episode}: {np.amin(episode_val_metrics):.4f}")
            print(f"Min loss in episode {episode}: {np.amin(episode_losses):.4f}")
            print(f"Mean loss in episode {episode}: {np.mean(episode_losses):.4f}")
            print(f"Time taken for episode {episode}: {(time.time() - episode_start_time):.2f}s")
            print(f"---------------------------------------")
        
        
        
        
    def train_sa_sc(self, epochs, batch_size, sc:bool=True, sa_start:int=0):
        # Some usefull data to be printed
        episode_start_time = time.time()
        episode_losses = []
        episode_train_metrics = []
        episode_val_metrics = []
        
        # Select loss function
        if self.n_classes == 1:
            loss_fn = dice_loss_fn
        else:
            loss_fn = wce_loss_fn
            
        model = SSA_Net(self.n_classes, sc, has_sa=(sa_start == 0))
        optimizer = self.__optimizer()
        train_acc_metric, val_acc_metric = self.__metrics()
        if self.n_classes != 1:
            weights = calc_loss_weights(self.train_data[1], class_nums=self.n_classes)
        else:
            weights = 0
        
        # Build datasets from input data
        train_gen = DataGenerator(*self.train_data)
        val_gen = DataGenerator(*self.val_data)
            
        # Functions for each step of train and validation
        if self.n_classes == 1:
            @tf.function
            def train_step(x, y, weights=0):
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss_value = loss_fn(y, logits, sa=(sa_start == 0))
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_acc_metric.update_state(y, logits[0])
                return loss_value
        else:
            @tf.function
            def train_step(x, y, weights):
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss_value = loss_fn(y, logits, weights, sa=(sa_start == 0))
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_acc_metric.update_state(y, logits[0])
                return loss_value
        @tf.function
        def test_step(x, y):
            val_logits = model(x, training=False)
            val_acc_metric.update_state(y, val_logits[0])
            
        best_train_acc = 1
        best_val_acc = 1
        changed_model = (sa_start == 0)

        for epoch in range(epochs):
            # A attention in selected epoch
            if not changed_model and epoch == sa_start and epoch != 0:
                print('changed model')
                new_model = SSA_Net(class_nums=self.n_classes, has_sc=sc, has_sa=True)
                
                new_model.build(input_shape = (None, 512, 512, 2))
                new_model.set_weights(model.get_weights())
                model = new_model
                if self.n_classes == 1:
                    @tf.function
                    def train_step(x, y, weights=0):
                        with tf.GradientTape() as tape:
                            logits = model(x, training=True)
                            loss_value = loss_fn(y, logits, sa=True)
                        grads = tape.gradient(loss_value, model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, model.trainable_weights))
                        train_acc_metric.update_state(y, logits[0])
                        return loss_value
                else:
                    @tf.function
                    def train_step(x, y, weights):
                        with tf.GradientTape() as tape:
                            logits = model(x, training=True)
                            loss_value = loss_fn(y, logits, weights, sa=True)
                        grads = tape.gradient(loss_value, model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, model.trainable_weights))
                        train_acc_metric.update_state(y, logits[0])
                        return loss_value
                changed_model = True
                
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            losses = []

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
                loss_value = train_step(x_batch_train, y_batch_train, weights)
                losses.append(loss_value)
                # Log every 5 batches.
                if step % 5 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * batch_size))
                train_gen.on_epoch_end()

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            episode_train_metrics.append(train_acc)

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in val_gen:
                test_step(x_batch_val, y_batch_val)
                
            if epoch % 5 == 0:
                    model.save_weights('{CWD}/Weights/sa_sc_Epoch{}.h5'.format(str(epoch)))
                    print('Model weights are saved. Name: {}'.format('Epoch' + str(epoch)))

            val_acc = val_acc_metric.result()
            episode_val_metrics.append(val_acc)
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Loss over epoch: %.4f" % (float(np.mean(losses)),))
            episode_losses.append(losses)
            print("Time taken: %.2fs" % (time.time() - start_time))
            if (best_train_acc > train_acc):
                best_train_acc = train_acc
                model.save_weights('{CWD}/Weights/sa_sc_best_train.h5')
            if (best_val_acc > val_acc):
                best_val_acc = val_acc
                model.save_weights('{CWD}/Weights/sa_sc_best_val.h5')

            train_gen.on_epoch_end()
                
        print(f"---------------------------------------")
        print(f"Best training acc: {np.amin(episode_train_metrics):.4f}")
        print(f"Best validation acc: {np.amin(episode_val_metrics):.4f}")
        print(f"Min loss: {np.amin(episode_losses):.4f}")
        print(f"Mean loss: {np.mean(episode_losses):.4f}")
        print(f"Time taken : {(time.time() - episode_start_time):.2f}s")
        print(f"---------------------------------------")