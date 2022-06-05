import snsynth
import pandas as pd

def get_generator(generator, epsilon, training_samples=10000):

    # update score

    update_score(epsilon,accuracy=0)

    data = pd.read_csv("./HCAIM_Challenge/data/res_train.csv",index_col = 0)
    data = data.sample(training_samples)

    # Ensure that generator is one of the accepted methods
    if generator not in ['MWEM','PATE-CTGAN','DP-CTGAN']:

        print(f"{generator} not in list of accepted methods. Please use one of MWEM, PATE-CTGAN or DP-CTGAN")
        return

    # Make sure epsilon is an acceptable size
    if (epsilon < 0.5) or (epsilon > 10):

        print("Please enter an epsilon value between 0.5 and 10")
        return

    if generator == "MWEM":


        nf = data.to_numpy().astype(int)

        synth = snsynth.MWEMSynthesizer(epsilon=epsilon, split_factor=nf.shape[1]) 

        synth.fit(nf)

        return synth

    elif generator == "PATE-CTGAN":

        from snsynth.pytorch.nn import PATECTGAN
        from snsynth.pytorch import PytorchDPSynthesizer

        synth = PytorchDPSynthesizer(epsilon, PATECTGAN(regularization='dragan'), None)

        synth.fit(data, categorical_columns=data.columns.to_list())

        return synth

    elif generator == "DP-CTGAN":

        from snsynth.pytorch.nn import DPCTGAN
        from snsynth.pytorch import PytorchDPSynthesizer

        synth = PytorchDPSynthesizer(epsilon, DPCTGAN(), None)

        synth.fit(data, categorical_columns=data.columns.to_list())

        return synth







def evaluate_model(model):

    from sklearn.metrics import classification_report, accuracy_score

    test_data = pd.read_csv('./HCAIM_Challenge/data/res_test.csv',index_col=0)

    x = test_data.drop('HeartDisease',axis=1)

    y = test_data['HeartDisease']

    y_pred = model.predict(x)

    clr = classification_report(y, y_pred)
    acc = accuracy_score(y,y_pred)

    update_score(epsilon=0,accuracy = acc)

    return clr


def update_score(epsilon, accuracy):

    import json

    path = './HCAIM_Challenge/score_status.json'

    with open(path,'r+') as f:
        data = json.load(f)

        data['total epsilon'] += epsilon

        if accuracy > data['best accuracy']:

            data['best accuracy'] = accuracy

        f.seek(0)        # <--- should reset file position to the beginning.
        json.dump(data, f, indent=4)
        f.truncate()     # remove remaining part








def check_score():

    print('temp')

