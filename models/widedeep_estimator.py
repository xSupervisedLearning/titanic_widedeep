import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os


class WideDeep:
    def __init__(self, train_df, eval_df, params):
        self.train_df = train_df
        self.eval_df = eval_df
        self.params = params
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn, params=self.params,
                                                model_dir=self.params["check_point_path"])

    def input_fn(self,df, num_epochs, shuffle, batch_size, predict=False):
        if not predict:
            labels = tf.constant(df[params['label_column']].apply(int).values)
        else:
            labels = tf.constant(np.zeros(df.shape[0]))

        features = {}
        for c in df.columns:
            if c != params['label_column']:
                features[c] = tf.constant(df[c].values)

        data_set = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            data_set = data_set.shuffle(100)
        data_set = data_set.repeat(num_epochs)
        data_set = data_set.batch(batch_size)
        return data_set

    def model_fn(self, features, labels, mode, params):
        wide = tf.feature_column.input_layer(features, params['wide_features'])
        wide = tf.layers.dense(wide, units=params["n_classes"])

        deep = tf.feature_column.input_layer(features, params['deep_features'])
        for units in params['hidden_units']:
            deep = tf.layers.dense(deep, units=units, activation=tf.nn.relu)
        deep = tf.layers.dense(deep, units=params['n_classes'])

        logits = wide + deep

        if mode == tf.estimator.ModeKeys.PREDICT:
            prob = tf.nn.softmax(logits)
            predicted_indices = tf.argmax(prob, 1)
            predictions = {
                'class': tf.gather(params['classes'], predicted_indices),
                'prob': prob
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(prob)
            }
            return tf.estimator.EstimatorSpec(mode,
                                              predictions=predictions,
                                              export_outputs=export_outputs)

        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        tf.summary.scalar("loss", loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.params['lr'])
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            prob = tf.nn.softmax(logits)
            predicted_indices = tf.argmax(prob, 1)
            labels_one_hot = tf.one_hot(
                labels,
                depth=2,
            )
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(labels, predicted_indices),
                'auroc': tf.metrics.auc(labels_one_hot, prob)
            }
            # Provide an estimator spec for `ModeKeys.EVAL` modes.
            return tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              eval_metric_ops=eval_metric_ops)

    def train_input_fn(self):
        return self.input_fn(self.train_df, self.params['num_epochs_evals'],
                             self.params['shuffle'], self.params['batch_size'], False)

    def eval_input_fn(self):
        return self.input_fn(self.train_df, 1, self.params['shuffle'], self.params['batch_size'], False)

    def train_with_early_stopping(self, metric_key="accuracy"):
        eval_results = []
        no_growth = 0
        for n in range(self.params['num_epochs'] // self.params['num_epochs_evals']):
            self.estimator.train(input_fn=self.train_input_fn)
            results = self.estimator.evaluate(input_fn=self.eval_input_fn)

            print("EVAL", " ".join(['%s: %-10s\t' % (key, results[key]) for key in sorted(results)]))

            if len(eval_results) < 5:
                eval_results.append(results)
                no_growth = 0
            elif results[metric_key] > max(eval_results, key=lambda x: x[metric_key])[metric_key]:
                eval_results.pop(0)
                eval_results.append(results)
                no_growth = 0
            else:
                no_growth += 1

            if no_growth > 3:
                break
        return max(eval_results, key=lambda x: x[metric_key])

    def predict(self, bst_steps, test_df):
        ckp_path = self.params['check_point_path'] + "model.ckpt-%d" % bst_steps
        res = self.estimator.predict(lambda: self.input_fn(test_df, 1, False, params['batch_size'], True),
                                     checkpoint_path=ckp_path)
        return res


def dataprocess():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    df = pd.concat([train, test], axis=0)
    df.drop(["Name", "PassengerId", "Ticket"], axis=1, inplace=True)

    df.Cabin.fillna("", inplace=True)
    df.Embarked.fillna("", inplace=True)
    df.Fare.fillna(0, inplace=True)
    df.Age.fillna(0, inplace=True)
    return df


def get_wide_deep_columns(df):
    numeric_columns = ["Age", "Fare", "Parch", "SibSp"]
    category_columns = ["Cabin", "Embarked", "Sex", "Pclass"]
    features = {}
    for c in numeric_columns:
        features[c] = tf.feature_column.numeric_column(c)
    for c in category_columns:
        features[c] = tf.feature_column.categorical_column_with_vocabulary_list(c, df[c].unique())

    age_buckets = tf.feature_column.bucketized_column(features["Age"], [6, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    fare_span = (df.Fare.max() - df.Fare.min()) // 10 + 1
    fare_buckets = tf.feature_column.bucketized_column(features["Fare"], [i * fare_span for i in range(10)])

    # embedding
    cabin_emb = tf.feature_column.embedding_column(features["Cabin"], dimension=8)
    embarked_emb = tf.feature_column.embedding_column(features['Embarked'], dimension=8)
    age_emb = tf.feature_column.embedding_column(age_buckets, dimension=8)
    fare_emb = tf.feature_column.embedding_column(fare_buckets, dimension=8)

    # one-hot
    sex_one_hot = tf.feature_column.indicator_column(features['Sex'])
    pclass_one_hot = tf.feature_column.indicator_column(features['Pclass'])
    cabin_one_hot = tf.feature_column.indicator_column(features['Cabin'])
    embarked_one_hot = tf.feature_column.indicator_column(features['Embarked'])

    wide_columns = [features[c] for c in numeric_columns] + [sex_one_hot, pclass_one_hot, cabin_one_hot,
                                                             embarked_one_hot]
    deep_columns = [features[c] for c in numeric_columns] + [sex_one_hot, pclass_one_hot, cabin_emb, embarked_emb,
                                                             age_emb, fare_emb]

    return wide_columns, deep_columns


if __name__ == "__main__":
    df = dataprocess()
    wide_columns, deep_columns = get_wide_deep_columns(df)

    params = {
        "label_column": "Survived",
        "num_epochs": 30,
        "num_epochs_evals": 1,
        "hidden_units": [100, 20],
        "wide_features": wide_columns,
        "deep_features": deep_columns,
        "n_classes": 2,
        "lr": 0.1,
        "batch_size": 32,
        "shuffle": True,
        "classes": [0, 1],
        "check_point_path": "../model_output/checkpoint/"
    }
    if os.path.exists(params['check_point_path']):
        for root, dirs, files in os.walk(params['check_point_path'], topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    train_df, valid_df = train_test_split(df[df.Survived.notna()], test_size=0.2)

    model = WideDeep(train_df, valid_df, params)
    max_step = model.train_with_early_stopping()

    test_df = df[df.Survived.isna()]

    print(max_step)

