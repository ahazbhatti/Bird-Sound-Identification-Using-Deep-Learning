import efficientnet.tfkeras as efn
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from vit_keras import vit
import numpy as np
import efficientnet.tfkeras as efn
import tensorflow_extra as tfe
from tensorflow.keras.applications import EfficientNetB1
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras.layers import Input,GlobalAveragePooling2D
from tensorflow.keras.layers import MultiHeadAttention,  Dropout,Add
from tensorflow.keras.layers import Input, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16

class VGG16_RCNN(object):
""" VGG16_RCNN class represents all the methods needed to run the model"""
    def __init__(self):
""" Whenver an object is created for this method, the message is displayed"""
        self.message = 'VGG16_RCNN'
    def initaiteModel(self):
        print(self.message)
    
    # Will download and load pretrained imagenet weights.
    def build_model(self, CFG, utility_obj,num_heads=4,
                        dropout_rate=0.1,
                        mlp_activation=tf.nn.gelu,
                        model_name=None, 
                        num_classes=264, 
                        compile_model=True, 
                        seq_len=12):
         """
        Builds and returns a model based on the specified configuration.
        Attributes:
        CFG: Object that represents configuration settings in config file
        Utility_obj: Represents utilities object
        num_classes: This paramter sets the number of classes in the classification task
        compile_model=True: Parameter to determine whether the model should be compiled or not
        """
        
        print(CFG)
            
        hidden_dim=1280
        mlp_dim=1280
        inp = tf.keras.layers.Input(shape=(None,))
        # Spectrogram
        x = tfe.layers.MelSpectrogram(n_mels=CFG.img_size[0],
                                        n_fft=CFG.nfft,
                                        hop_length=CFG.hop_length, 
                                        sr=CFG.sample_rate,
                                        ref=1.0,
                                        out_channels=3)(inp)
        # Normalize
        x = tfe.layers.ZScoreMinMax()(x)
        # TimeFreqMask
        x = tfe.layers.TimeFreqMask(freq_mask_prob=0.5,
                                      num_freq_masks=1,
                                      freq_mask_param=10,
                                      time_mask_prob=0.5,
                                      num_time_masks=2,
                                      time_mask_param=25,
                                      time_last=False,)(x)
        # Load backbone model\
        base = VGG16(input_shape=(None, None, 3),
                 include_top=False,
                 weights='imagenet')

        # Add RPN layer
        out = base(x)

        # Region of Interest (RoI) pooling layer
        out = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), activation='relu')(out)
        out = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(out)
        out = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(out)
        out = tf.keras.layers.GlobalMaxPooling2D()(out)

        # Feed the RoI feature maps into a fully connected layer and perform classification
        out = tf.keras.layers.Dense(1024, activation='relu')(out)
        out = tf.keras.layers.Dense(1024, activation='relu')(out)
        out = tf.keras.layers.Dense(num_classes, activation='softmax')(out)

        # Create the model
        model = tf.keras.models.Model(inputs=inp, outputs=out)

        if compile_model:
            # Set the optimizer
            opt = utility_obj.get_optimizer()
            # Set the loss function
            loss = utility_obj.get_loss()
            # Set the evaluation metrics
            metrics = utility_obj.get_metrics()
            # Compile the model with the specified optimizer, loss function, and metrics
            model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return model
    
    def pretrain_vgg(self,model, CFG, utility_obj, df_pre, fold,strategy,debug=True):
        """
        This method represents pretraining steps required for the model.
        model: Existing neural network model used for pretraining
        df_pre: Dataframe object that contains pretraining data
        fold: number of folds
        strategy: represents which environment is the code running in, such as GPUs, or multiple machines etc.
        debug=True: Whether the model should be in debug mode or not
        """
        
        # Configurations
        num_classes = utility_obj.num_classes2
        df = df_pre.copy()
        fold = fold

        # Compute batch size and number of samples to drop
        infer_bs = (CFG.batch_size*CFG.infer_bs)
        drop_remainder = CFG.drop_remainder

        # Split dataset with cv filter
        if CFG.cv_filter:
            df = utility_obj.filter_data(df, thr=5)
            train_df = df.query("fold!=@fold | ~cv").reset_index(drop=True)
            valid_df = df.query("fold==@fold & cv").reset_index(drop=True)
        else:
            train_df = df.query("fold!=@fold").reset_index(drop=True)
            valid_df = df.query("fold==@fold").reset_index(drop=True)

        # Upsample train data
        train_df = utility_obj.upsample_data(train_df, thr=50)
        train_df = utility_obj.downsample_data(train_df, thr=500)

        # Get file paths and labels
        train_paths = train_df.filepath.values; train_labels = train_df.target.values
        valid_paths = valid_df.filepath.values; valid_labels = valid_df.target.values

        # Shuffle the file paths and labels
        index = np.arange(len(train_paths))
        np.random.shuffle(index)
        train_paths  = train_paths[index]
        train_labels = train_labels[index]

        # For debugging
        if debug:
            min_samples = CFG.batch_size*CFG.replicas*2
            train_paths = train_paths[:min_samples]; train_labels = train_labels[:min_samples]
            valid_paths = valid_paths[:min_samples]; valid_labels = valid_labels[:min_samples]

        # Ogg or Mp3
        train_ftype = list(map(lambda x: '.ogg' in x, train_paths))
        valid_ftype = list(map(lambda x: '.ogg' in x, valid_paths))

        # Compute the number of training and validation samples
        num_train = len(train_paths); num_valid = len(valid_paths)

        # Build the training and validation datasets
        cache=False
        train_ds = utility_obj.build_dataset(train_paths, train_ftype, train_labels, 
                                 batch_size=CFG.batch_size*CFG.replicas, cache=cache, shuffle=True,
                                drop_remainder=drop_remainder, num_classes=num_classes)
        valid_ds = utility_obj.build_dataset(valid_paths, valid_ftype, valid_labels,
                                 batch_size=CFG.batch_size*CFG.replicas, cache=True, shuffle=False,
                                 augment=False, repeat=False, drop_remainder=drop_remainder,
                                 take_first=True, num_classes=num_classes)

        # Print information about the fold and training
        print('#'*25); print('#### Pre-Training')
        print('#### Image Size: (%i, %i) | Model: %s | Batch Size: %i | Scheduler: %s'%
              (*CFG.img_size, CFG.model_name, CFG.batch_size*CFG.replicas, CFG.scheduler))
        print('#### Num Train: {:,} | Num Valid: {:,}'.format(len(train_paths), len(valid_paths)))

        # Clear the session and build the model
        K.clear_session()
        with strategy.scope():
            model = self.build_model(CFG, utility_obj, model_name=CFG.model_name, num_classes=num_classes)

        print('#'*25) 

        # Checkpoint Callback
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            'birdclef_VGG16_RCNN_pretrained.h5', monitor='val_auc', verbose=1, save_best_only=True,
            save_weights_only=False, mode='max', save_freq='epoch')
        # LR Scheduler Callback
        lr_cb = utility_obj.get_lr_callback(CFG.batch_size*CFG.replicas)
        callbacks = [ckpt_cb, lr_cb]

        print(train_ds)
        # Training
        history = model.fit(
            train_ds, 
            epochs=1 if debug else CFG.epochs, 
            callbacks=callbacks, 
            steps_per_epoch=len(train_paths)/CFG.batch_size//CFG.replicas,
            validation_data=valid_ds, 
            verbose=CFG.verbose,
        )

        # Show training plot
        if CFG.training_plot:
            utility_obj.plot_history(history)
            
    def vgg_rcnn_inference(self,CFG,utility_obj,test_df,test_paths,test_ftype,finetuned_model_path):
        """
        This method is called when the testing/inference of the model needs to be performed.
        test_df: Represents test data dataframe
        test_paths: Variable that represents location of the test file
        test_ftype: Variable that represents the file type or format of the test data files.
        finetuned_model_path: Variable that represents file path of a finetuned model
        """
        # Compute batch size and number of samples to drop
        num_classes = utility_obj.num_classes
        infer_bs = (CFG.batch_size*CFG.infer_bs)
        drop_remainder = CFG.drop_remainder
        num_test = len(test_paths)
         # Print information about the fold and training
        print('#'*25); print('#### Testing')
        print('#### Image Size: (%i, %i) | Model: %s'%
                      (*CFG.img_size, 'VGG16_RCNN'))
        print('#### Num Test: {:,}'.format(len(test_paths)))

        oof_pred = []; oof_true = []
        print('# Loading finetuned model')
        model = tf.keras.models.load_model(finetuned_model_path)

        # Predict on the testing data
        print('# Testing the model')
        test_ds = utility_obj.build_dataset(test_paths, test_ftype, labels=None, augment=CFG.tta>1, repeat=True, cache=False, 
                                         shuffle=False, batch_size=infer_bs*CFG.replicas, 
                                         drop_remainder=drop_remainder, take_first=True, num_classes=num_classes)
        ct_test = len(test_paths); STEPS = CFG.tta * ct_test/infer_bs/CFG.replicas
        pred = model.predict(test_ds,steps=STEPS,verbose=CFG.verbose)[:CFG.tta*ct_test,] 
        pred = np.mean(pred.reshape((CFG.tta,ct_test,-1)),axis=0)
        oof_pred.append(pred)               
        # Get ids and targets
        oof_true.append(test_df[CFG.target_col].values[:ct_test])
        # Save valid data prediction
        y_true = oof_true[-1].reshape(-1).astype('float32')
        y_pred = oof_pred[-1].argmax(axis=-1)
        test_df.loc[:num_test - 1, 'pred'] = y_pred
        test_df.loc[:num_test - 1, 'miss'] = y_true != y_pred
        test_df.loc[:num_test - 1, utility_obj.class_names] = oof_pred[-1].tolist()

        save_df = test_df #.query("miss==True")
        # Map the predicted and target labels to their corresponding names
        save_df.loc[:, 'pred_name'] = save_df.pred.map(utility_obj.label2name)
        save_df.loc[:, 'target_name'] = save_df.target.map(utility_obj.label2name)
        # Trim the dataframe for debugging purposes
        if CFG.debug:
            save_df = save_df.iloc[:CFG.replicas*CFG.batch_size*CFG.infer_bs]
        noimg_cols = [*CFG.tab_cols, 'target', 'pred', 'target_name','pred_name','miss']

        # Retain only the necessary columns
        save_df = save_df.loc[:, noimg_cols]
        save_df.to_csv('Test_Predictions.csv', index=False)
        print("Saved all test predictions to Test_Predictions.csv")        
        missed_df = test_df.query("miss==True")
        missed_df.to_csv('Missed_Predictions.csv', index=False)
        print("Saved all misclassified predictions to Missed_Predictions.csv")  

        cal_accuracy = ((len(save_df.index)- len(missed_df.index))/len(save_df.index))*100
        print("Test Accuracy :",cal_accuracy)
        
    def vgg_rcnn_finetune_test(self,df_23,utility_obj,strategy,CFG, model_path,inference=True,epochs=10,batch_size=32,test_samples_size=100,debug=True):
        """
        The vgg_rcnn_finetune_test method appears to be responsible for performing the finetuning and 
        testing of a pretrained VGG16-RCNN model using the provided data, configurations, and utility objects. 
        The specifics of the finetuning and testing process, including the loss function, optimizer, callbacks, 
        and evaluation metrics, would likely be implemented within the method using the provided parameters and additional code.
        """

    
        CFG.epochs = epochs
        num_classes = utility_obj.num_classes
        df = df_23.copy()
        CFG.selected_folds=[0]
        CFG.debug = debug

        
        test_fold = 4
        
        if(test_samples_size<64):
            print('Test Sample Size cannot be less than 64')
            return

        for fold in range(CFG.num_fold):
            # Check if the fold is selected
            if fold not in CFG.selected_folds:
                continue

            # Compute batch size and number of samples to drop
            infer_bs = (CFG.batch_size*CFG.infer_bs)
            drop_remainder = CFG.drop_remainder

            # Split dataset with cv filter
  
            if CFG.cv_filter:
                df = utility_obj.filter_data(df, thr=5)
                train_df = df.query("fold!=@fold | fold!=@test_fold | ~cv").reset_index(drop=True)
                test_df = df.query("fold==@test_fold | ~cv").reset_index(drop=True)
                valid_df = df.query("fold==@fold & cv").reset_index(drop=True)
            else:
                train_df = df.query("fold!=@fold | fold!=@test_fold").reset_index(drop=True)
                test_df = df.query("fold==@test_fold").reset_index(drop=True)
                valid_df = df.query("fold==@fold").reset_index(drop=True)



            # Upsample train data
            train_df = utility_obj.upsample_data(train_df, thr=50)

          
            # Get file paths and labels
            train_paths = train_df.filepath.values; train_labels = train_df.target.values
            valid_paths = valid_df.filepath.values; valid_labels = valid_df.target.values
            test_paths = test_df.filepath.values; test_labels = test_df.target.values

            # Shuffle the file paths and labels
            index = np.arange(len(train_paths))
            np.random.shuffle(index)
            train_paths  = train_paths[index]
            train_labels = train_labels[index]
            # For debugging
            if debug==True:
                min_samples = CFG.batch_size*CFG.replicas*2
                train_paths = train_paths[:min_samples]; train_labels = train_labels[:min_samples]
                valid_paths = valid_paths[:min_samples]; valid_labels = valid_labels[:min_samples]
                test_paths = test_paths[:min_samples]; test_labels = test_labels[:min_samples]

            # Ogg or Mp3
            train_ftype = list(map(lambda x: '.ogg' in x, train_paths))
            valid_ftype = list(map(lambda x: '.ogg' in x, valid_paths))
            test_ftype = list(map(lambda x: '.ogg' in x, test_paths))

            # Compute the number of training and validation samples
            num_train = len(train_paths); num_valid = len(valid_paths); num_test = len(test_paths)


            # Build the training and validation datasets
            cache=True
            train_ds = utility_obj.build_dataset(train_paths, train_ftype, train_labels, 
                                     batch_size=CFG.batch_size*CFG.replicas, cache=cache, shuffle=True,
                                    drop_remainder=drop_remainder, num_classes=num_classes)
            valid_ds = utility_obj.build_dataset(valid_paths, valid_ftype, valid_labels,
                                     batch_size=CFG.batch_size*CFG.replicas, cache=cache, shuffle=False,
                                     augment=False, repeat=False, drop_remainder=drop_remainder,
                                     take_first=True, num_classes=num_classes)
            test_ds = utility_obj.build_dataset(test_paths, test_ftype, test_labels,
                                     batch_size=CFG.batch_size*CFG.replicas, cache=cache, shuffle=False,
                                     augment=False, repeat=False, drop_remainder=drop_remainder,
                                     take_first=True, num_classes=num_classes)


            # Clear the session and build the model
            K.clear_session()
            if inference==True:
                self.vgg_rcnn_inference(CFG,utility_obj,test_df,test_paths,test_ftype,model_path)
            else:
                 # Print information about the fold and training
                print('#'*25); print('#### Training')
                print('#### Fold: %i | Image Size: (%i, %i) | Model: %s | Batch Size: %i | Scheduler: %s'%
                      (fold+1, *CFG.img_size, 'VGG-RCNN', CFG.batch_size*CFG.replicas, CFG.scheduler))
                print('#### Num Train: {:,} | Num Valid: {:,} | Num Test: {:,}'.format(len(train_paths), len(valid_paths), len(test_paths)))

                oof_pred = []; oof_true = []; oof_val = []; oof_ids = []; oof_folds = [] 
                with strategy.scope():
                    model = self.build_model(CFG,utility_obj, model_name=CFG.model_name, num_classes=num_classes,num_heads=8,dropout_rate=0.2)
                # Load birdclef pretrained weights
                model.load_weights(model_path, by_name=True, skip_mismatch=True)

                print('#'*25) 

                # Callbacks
                ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
                    'birdclef_enet_vit_finetuned.h5', monitor='val_auc', verbose=0, save_best_only=True,
                    save_weights_only=False, mode='max', save_freq='epoch')
                # LR Scheduler Callback
                lr_cb = utility_obj.get_lr_callback(CFG.batch_size*CFG.replicas)
                callbacks = [ckpt_cb, lr_cb]

                # Training
                history = model.fit(
                    train_ds, 
                    epochs=2 if debug else CFG.epochs, 
                    callbacks=callbacks, 
                    steps_per_epoch=len(train_paths)/CFG.batch_size//CFG.replicas,
                    validation_data=valid_ds, 
                    verbose=CFG.verbose,
                )

                # Load best checkpoint
                print('# Loading best model')
                model.load_weights('birdclef_enet_vit_finetuned.h5')

                # Predict on the validation data for oof result
                print('# Infering OOF')
                valid_ds = utility_obj.build_dataset(valid_paths, valid_ftype, labels=None, augment=CFG.tta>1, repeat=True, cache=False, 
                                         shuffle=False, batch_size=infer_bs*CFG.replicas, 
                                         drop_remainder=drop_remainder, take_first=True, num_classes=num_classes)
                ct_valid = len(valid_paths); STEPS = CFG.tta * ct_valid/infer_bs/CFG.replicas
                pred = model.predict(valid_ds,steps=STEPS,verbose=CFG.verbose)[:CFG.tta*ct_valid,] 
                pred = np.mean(pred.reshape((CFG.tta,ct_valid,-1)),axis=0)
                oof_pred.append(pred)               

                # Get ids and targets
                oof_true.append(valid_df[CFG.target_col].values[:ct_valid])
                oof_folds.append(np.ones_like(oof_true[-1],dtype='int8')*fold )
                oof_ids.append(valid_paths)

                # Save valid data prediction
                y_true = oof_true[-1].reshape(-1).astype('float32')
                y_pred = oof_pred[-1].argmax(axis=-1)
                valid_df.loc[:num_valid - 1, 'pred'] = y_pred
                valid_df.loc[:num_valid - 1, 'miss'] = y_true != y_pred
                valid_df.loc[:num_valid - 1, utility_obj.class_names] = oof_pred[-1].tolist()

                # Log the metrics
                scores = {}
                #cmAP = utility_obj.padded_cmap(tf.keras.utils.to_categorical(y_true), oof_pred[-1])
                save_df = valid_df #.query("miss==True")
                # Map the predicted and target labels to their corresponding names
                save_df.loc[:, 'pred_name'] = save_df.pred.map(utility_obj.label2name)
                save_df.loc[:, 'target_name'] = save_df.target.map(utility_obj.label2name)
                # Trim the dataframe for debugging purposes
                if CFG.debug:
                    save_df = save_df.iloc[:CFG.replicas*CFG.batch_size*CFG.infer_bs]
                noimg_cols = [*CFG.tab_cols, 'target', 'pred', 'target_name','pred_name','miss']

                # Retain only the necessary columns
                save_df = save_df.loc[:, noimg_cols]
                missed_df = valid_df.query("miss==True")
                

                cmAP = ((len(save_df.index)- len(missed_df.index))/len(save_df.index))*100
                print("###### Test Accuracy :",cmAP)
                print("###### Misclassified predictions: ",missed_df)  
                
                best_epoch = np.argmax(history.history['val_'+CFG.monitor], axis=-1) + 1
                best_score = history.history['val_'+CFG.monitor][best_epoch - 1]
                scores.update({'auc': best_score,
                               'epoch': best_epoch,
                               'cmAP': cmAP,})
                oof_val.append(best_score)
                print('\n>>> FOLD %i OOF AUC = %.3f | Validation Accuracy = %.3f' % (fold+1, oof_val[-1], cmAP))
                
                # Show training plot
                if CFG.training_plot:
                    utility_obj.plot_history(history)
                    
     #change in klagglre               
    def vgg_main(self,CFG,utility_obj,mode_of_training,load_model_path,test_sample_size,epochs=10,dropout_rate=0.2, mlp_activation=tf.nn.gelu, num_heads=8,debug=True):
              """
        Method that serves as an entry point for training, finetuning and inference using the vgg16 model.
        """
        
        # Initialize GPU/TPU/TPU-VM
        strategy, CFG.device, tpu = utility_obj.get_device()
        CFG.replicas = strategy.num_replicas_in_sync


        #prepare the pre-training and training datasets
        #We need to use birdclef_23 along with 21/22 to make sure that there are no duplicates the the pre-training data
        df_23_original = utility_obj.get_birdclef2023_df()
        df_pre_original = utility_obj.get_pretraining_df(df_23_original)
        df_pre = utility_obj.stratifydata(df_pre_original,25)
        df_23 = utility_obj.stratifydata(df_23_original,5)

        #Call the methods based on what action needs to be preformed
        if(mode_of_training == "pretraining"):

            #Build the model
            model = self.build_model(CFG,utility_obj, model_name=CFG.model_name,num_heads=num_heads,dropout_rate=dropout_rate)
            model.summary()

            #Check the built model for output
            audios = tf.random.uniform((1, CFG.audio_len))
            with strategy.scope():
                out = model(audios, training=False)
            print(out.shape)

            fold = 0 #Only fold 0 will be used for validation and all others for pre-training
            self.pretrain_vgg(model,CFG,utility_obj,df_pre,fold,strategy,debug=debug)
        elif(mode_of_training == "finetuning"): 
            #myModel.finetune_enet_vit(df_23,utility_obj,strategy,CFG,inference)
            inference=False
            self.vgg_rcnn_finetune_test(df_23,utility_obj,strategy,CFG,load_model_path,inference=inference,epochs=epochs,test_samples_size=test_sample_size,debug=debug)
                                                   
        elif(mode_of_training == "inference"):
            inference=True
            self.vgg_rcnn_finetune_test(df_23,utility_obj,strategy,CFG,load_model_path,inference=inference,epochs=epochs,test_samples_size=test_sample_size,debug=debug)
        else:
            print("Mode of Training not supplied")
            
VGG_RCNN = VGG16_RCNN()

if __name__ == "__main__":
    VGG16_RCNN.initaiteModel()