model.add(Dense(units=512 , activation='relu'))
model.add(Dense(units=6, activation='softmax'))
model.summary()
