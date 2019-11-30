import tensorflow as tf
from    tensorflow.keras import datasets,layers,optimizers,metrics

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.convert_to_tensor(x_train,dtype=tf.float32) /255.
    y_train = tf.convert_to_tensor(y_train,dtype=tf.int32)

    print(x_train.shape,y_train.shape)
    print(x_train[:5],y_train[:5])

    train_datasets = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(128)

    model = tf.keras.Sequential(
        [
            layers.Dense(512,activation="relu"),
            layers.Dense(128,activation="relu"),
            layers.Dense(10),
        ]
    )

    optimizers = optimizers.SGD(learning_rate=0.001)
    acc_meter = metrics.Accuracy()
    for epoch in range(30):
        for step ,(x,y) in enumerate(train_datasets):
            with tf.GradientTape() as tape:
                x = tf.reshape(x,(-1,28*28))
                y_one_hot = tf.one_hot(y, depth=10)
                out = model(x)
                loss = tf.reduce_sum(tf.square(out - y_one_hot)) / x.shape[0]

            grads = tape.gradient(loss,model.trainable_variables)

            optimizers.apply_gradients(zip(grads,model.trainable_variables))

            acc_meter.update_state(tf.argmax(out, axis=1),y)
            if step % 100 == 0 :
                print(epoch,step,"loss: ",float(loss),"acc: ",acc_meter.result().numpy())
                acc_meter.reset_states()
