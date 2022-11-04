checkpoint = ModelCheckpoint(r'D:\IBM Project\gesture.h5',
                            monitor='val_loss',save_best_only=True,verbose=3)

earlystop = EarlyStopping(monitor = 'val_loss', patience=7, verbose= 3, restore_best_weights=True)

learning_rate = ReduceLROnPlateau(monitor= 'val_loss', patience=7, verbose= 3, )

callbacks=[checkpoint,earlystop,learning_rate]

model.fit_generator(train_gen,
                    epochs=25,
                    steps_per_epoch=18000//32,
                    validation_data=test_gen,
                    callbacks=callbacks,
                    verbose = 1,validation_steps=3600//32)
