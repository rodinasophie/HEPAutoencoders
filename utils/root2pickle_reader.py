import pandas as pd
import uproot
import numpy as np
from sklearn.model_selection import train_test_split


ROOTFilePath = "data/DAOD_TRIG6.16825104._000230.pool.root.1"
ROOTFilePath = "data/DAOD_TRIG6.16825104._000230.pool.root.1"

"""
This class unpacks the 4-dimensional and 27-dimaensional data from the ROOT file to a pickled pandas dataframe
"""


class ROOT2PickleReader:
    def __init__(self):
        self.processed_folder = "processed_data/"
        self.processed_folder_aod = "processed_data/aod/"

    def root2pickle4D(
        self, root_file, pickle_file_prefix=None, auto_filename=True, save=False
    ):
        # Unused if auto_filename = True
        if not auto_filename:
            train_filename = pickle_file_prefix + "_train.pkl"
            test_filename = pickle_file_prefix + "_test.pkl"

        # Fraction of data to be saved
        data_frac = 0.2

        tree = uproot.open(root_file)["CollectionTree"]

        # Specifies the dataset. The available 'columns' can be read with ttree.keys()
        branchnames = [
            "HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.m",
            "HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.pt",
            "HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.phi",
            "HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.eta",
        ]

        df_dict = {}
        for pp, branchname in enumerate(branchnames):
            print("Reading: " + branchname)
            if "EnergyPerSampling" in branchname:
                pass
            else:
                variable = branchname.split(".")[1]
                df_dict[variable] = []
                jaggedX = tree.array(branchname)
                for ii, arr in enumerate(jaggedX):
                    for kk, val in enumerate(arr):
                        df_dict[variable].append(val)

        print("100%")
        print("Creating DataFrame...")
        df = pd.DataFrame(data=df_dict)
        print("Head of data:")
        print(df.head())

        train, test = train_test_split(df, test_size=0.2, random_state=41)

        if auto_filename:
            train_filename = (
                "all_jets_train_4D_" + str(int(data_frac * 100)) + "_percent.pkl"
            )
            test_filename = (
                "all_jets_test_4D_" + str(int(data_frac * 100)) + "_percent.pkl"
            )

        partial_train_percent = train.sample(
            frac=data_frac, random_state=42
        ).reset_index(
            drop=True
        )  # Pick out a fraction of the data
        partial_test_percent = test.sample(frac=data_frac, random_state=42).reset_index(
            drop=True
        )

        print("Train data shape: " + str(train.shape))
        print("Test data shape: " + str(test.shape))

        # Save train and test sets
        if save:
            print("Saving " + self.processed_folder + train_filename)
            train.to_pickle(self.processed_folder + train_filename)
            print("Saving " + self.processed_folder + test_filename)
            test.to_pickle(self.processed_folder + test_filename)
        return train, test

    def root2pickle27D(
        self, root_file, pickle_file_prefix=None, auto_filename=True, save=False
    ):
        # Unused if auto_filename = True
        if not auto_filename:
            train_filename = pickle_file_prefix + "_train.pkl"
            test_filename = pickle_file_prefix + "_test.pkl"

        # Fraction of data to be saved
        data_frac = 0.05

        tree = uproot.open(root_file)["CollectionTree"]

        n_jets = sum(
            tree.array(
                "HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.pt"
            ).counts
        )

        # Number of events to be processed
        maxEvents = int(n_jets * data_frac)

        # Specifies the dataset. The available 'columns' can be read with ttree.keys()
        prefix = "HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn"
        branchnames = [
            # 4-momentum
            prefix + ".pt",
            prefix + ".eta",
            prefix + ".phi",
            prefix + ".m",
            # Energy deposition in each calorimeter layer
            # prefix + '.EnergyPerSampling',
            # Area of jet,used for pile-up suppression (4-vector)
            prefix + ".ActiveArea",
            prefix + ".ActiveArea4vec_eta",
            prefix + ".ActiveArea4vec_m",
            prefix + ".ActiveArea4vec_phi",
            prefix + ".ActiveArea4vec_pt",
            # prefix + '.JetGhostArea',
            # Variables related to quality of jet
            prefix + ".AverageLArQF",
            # prefix + '.BchCorrCell',
            prefix + ".NegativeE",
            prefix + ".HECQuality",
            prefix + ".LArQuality",
            # Shape and position, most energetic cluster
            prefix + ".Width",
            prefix + ".WidthPhi",
            prefix + ".CentroidR",
            prefix + ".DetectorEta",
            prefix + ".LeadingClusterCenterLambda",
            prefix + ".LeadingClusterPt",
            prefix + ".LeadingClusterSecondLambda",
            prefix + ".LeadingClusterSecondR",
            prefix + ".N90Constituents",
            # Energy released in each calorimeter
            prefix + ".EMFrac",
            prefix + ".HECFrac",
            # Variables related to the time of arrival of a jet
            prefix + ".Timing",
            prefix + ".OotFracClusters10",
            prefix + ".OotFracClusters5",
        ]

        df_dict = {}
        for pp, branchname in enumerate(branchnames):
            print("Reading: " + branchname)
            if "EnergyPerSampling" in branchname:
                pass
            else:
                variable = branchname.split(".")[1]
                df_dict[variable] = []
                jaggedX = tree.array(branchname)[:maxEvents]
                for ii, arr in enumerate(jaggedX):
                    for kk, val in enumerate(arr):
                        df_dict[variable].append(val)

        print("Creating DataFrame...")
        df = pd.DataFrame(data=df_dict)
        print("Head of data:")
        print(df.head())

        train, test = train_test_split(df, test_size=0.2, random_state=41)

        if auto_filename:
            train_filename = (
                "all_jets_train_27D_" + str(int(data_frac * 100)) + "_percent.pkl"
            )
            test_filename = (
                "all_jets_test_27D_" + str(int(data_frac * 100)) + "_percent.pkl"
            )

        partial_train_percent = train.sample(
            frac=data_frac, random_state=42
        ).reset_index(
            drop=True
        )  # Pick out a fraction of the data
        partial_test_percent = test.sample(frac=data_frac, random_state=42).reset_index(
            drop=True
        )

        print("Train data shape: " + str(train.shape))
        print("Test data shape: " + str(test.shape))

        # Save train and test sets
        if save:
            print("Saving " + self.processed_folder_aod + train_filename)
            train.to_pickle(self.processed_folder_aod + train_filename)
            print("Saving " + self.processed_folder_aod + test_filename)
            test.to_pickle(self.processed_folder_aod + test_filename)
        return train, test
