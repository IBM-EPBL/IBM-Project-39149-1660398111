checkpoint = ModelCheckpoint(r'D:\IBM Project\gesture.h5',
                            monitor='val_loss',save_best_only=True,verbose=3)

earlystop = EarlyStopping(monitor = 'val_loss', patience=7, verbose= 3, restore_best_weights=True)

learning_rate = ReduceLROnPlateau(monitor= 'val_loss', patience=3, verbose= 3, )

callbacks=[checkpoint,earlystop,learning_rate]

model.fit(train_data,
          validation_data=test_data,
          epochs=25,callbacks=callbacks,validation_steps=len(test_data)//8)
