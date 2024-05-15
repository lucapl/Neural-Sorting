from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Metric, Precision, Recall, AUC,Accuracy

from src.tensors.metrics import F1Score

def train(model,X_train,y_train,loss,val_data=None,batch=10,epochs=50,patience=3,metrics=None):
    early_stopping = EarlyStopping(monitor='loss' if val_data == None else "val_loss", patience=patience, restore_best_weights=True)
    if metrics == None: metrics = ["accuracy",AUC(name="auc"),F1Score()]

    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=metrics)

    history = model.fit(X_train,y_train,
    batch_size=batch,
    epochs=epochs,
    callbacks=[early_stopping],
    validation_data=val_data)

    return history