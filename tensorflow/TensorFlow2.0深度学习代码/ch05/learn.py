import tensorflow as tf
from matplotlib import pyplot as plt

def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32) / 255.
    x = tf.reshape(x,[-1,28*28])
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
traindb = tf.data.Dataset.from_tensor_slices((x_train,y_train))
traindb = traindb.shuffle(10000)
traindb = traindb.batch(128)
traindb = traindb.map(preprocess)
traindb = traindb.repeat(20)

testdb = tf.data.Dataset.from_tensor_slices((x_test,y_test))
testdb = testdb.shuffle(10000).batch(128).map(preprocess)

x,y = next(iter(testdb))
print(x.shape,y.shape)



def train():
    lr = 1e-2
    acc ,losses = [],[]

    w1,b1 = tf.Variable(tf.random.normal([784,256],stddev=0.1)),tf.Variable(tf.zeros([256]))
    w2,b2 = tf.Variable(tf.random.normal([256,128],stddev=0.1)),tf.Variable(tf.zeros([128]))
    w3,b3 = tf.Variable(tf.random.normal([128,10],stddev=0.1)),tf.Variable(tf.zeros([10]))

    for step,(x,y) in enumerate(traindb):
        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)

            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            out = h2 @ w3 + b3
            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
        for p,g in zip([w1,b1,w2,b2,w3,b3],grads):
            p.assign_sub(g * lr)

        if step % 80 == 0 :
            print(step ,"loss:",float(loss))
            losses.append(loss)

        if step % 80 == 0 :
            total , total_correct = 0.,0
            for x,y in testdb:
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)

                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)

                out = h2 @ w3 + b3
                prep = tf.argmax(out,axis=1)
                y = tf.argmax(y,axis=1)
                correct = tf.equal(prep,y)
                total_correct += tf.reduce_sum(tf.cast(correct,dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct/total)
            acc.append(total_correct/total)

    plt.figure()
    x = [i*80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    # plt.savefig('train.svg')
    plt.show()

    plt.figure()
    plt.plot(x, acc, color='C1', marker='s', label='测试')
    plt.ylabel('准确率')
    plt.xlabel('Step')
    plt.legend()
    # plt.savefig('test.svg')
    plt.show()

if __name__ == '__main__':
    train()