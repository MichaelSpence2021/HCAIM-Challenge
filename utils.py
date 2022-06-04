import snsynth
import pandas as pd

def get_generator(generator, epsilon):

    data = pd.read_csv("./HCAIM_Challenge/data/res_train.csv")
    data = data.sample(10000)

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

    from sklearn.metrics import classification_report

    test_data = pd.read_csv('./HCAIM_Challenge/data/test.csv')

    x = test_data.drop('HeartDisease',axis=1)

    y = test_data['HeartDisease']

    y_pred = model.predict(x)

    clr = classification_report(y, y_pred)

    return clr








def check_score():

    print('temp')

