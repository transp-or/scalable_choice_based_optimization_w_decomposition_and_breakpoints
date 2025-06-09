import copy
import random
import numpy as np
from scipy.stats import truncnorm


def discrete_choice_model(data_dict, N, J_PSP=1, J_PUP=1, more_classes=False):
    # Define the type of discrete choice model
    data_dict['DCM'] = 'MixedLogit'
    data_dict['N'] = N

    ##########################################################
    # DISCRETE CHOICE MODEL PARAMETERS
    ##########################################################

    # Alternative Specific Coefficients
    data_dict['ASC_FSP'] = 0.0  # Free Street Parking
    data_dict['ASC_PSP'] = 32.0  # Paid Street Parking
    data_dict['ASC_PUP'] = 34.0  # Paid Underground Parking

    # Beta coefficients
    data_dict['Beta_TD'] = -0.612
    data_dict['Beta_Origin'] = -5.762
    data_dict['Beta_Age_Veh'] = 4.037
    data_dict['Beta_FEE_RES_PSP'] = -11.440
    data_dict['Beta_FEE_INC_PSP'] = -10.995  # reduce mean of PSP price param by this if low income == 1
    data_dict['Beta_FEE_INC_PUP'] = -13.729  # reduce mean of PUP price param by this if low income == 1
    if more_classes:
        data_dict['Beta_FEE_INC_PSP_POOR'] = -14.995  # reduce mean of price param by this if low income == 3 (poor)
        data_dict['Beta_FEE_INC_PUP_POOR'] = -18.351  # reduce mean of price param by this if low income == 3 (poor)
        data_dict['Beta_FEE_INC_PSP_RICH'] = 4.152  # increase mean of price param by this if low income == 2 (rich)
        data_dict['Beta_FEE_INC_PUP_RICH'] = 7.221  # increase mean of price param by this if low income == 2 (rich)
    data_dict['Beta_FEE_RES_PUP'] = -10.668

    gener = "old"

    if gener == "old":
        # Access time coefficient (random parameter with normal distribution)
        vecAT = np.random.normal(-0.788, 1.064, size=(data_dict['N'], data_dict['R']))
        # Fee coefficient (random parameter with normal distribution)
        vec = np.random.normal(-32.328, 14.168, size=(data_dict['N'], data_dict['R']))

        # Iterate through each element and redraw if it's positive
        for i in range(vec.shape[0]):
            for j in range(vec.shape[1]):
                while vec[i, j] > 0:
                    vec[i, j] = np.random.normal(-32.328, 14.168)

        data_dict['Beta_AT'] = vecAT
        data_dict['Beta_FEE'] = vec
    elif gener == "true":
        # Define the mean vector and covariance matrix
        mean = [-0.788, -32.328]
        cov = [[1.064 ** 2, -12.8], [-12.8, 14.168 ** 2]]
        # Draw samples from the multivariate normal distribution
        samples = np.random.multivariate_normal(mean, cov, size=(data_dict['N'], data_dict['R']))
        # # Split the samples into vecAT and vec
        vecAT = samples[:, :, 0]
        vec = samples[:, :, 1]
        data_dict['Beta_AT'] = vecAT
        data_dict['Beta_FEE'] = vec
    else:
        # Define the mean vector and covariance matrix
        mean = [-0.788, -32.328]
        cov = [[1.064 ** 2, -12.8], [-12.8, 14.168 ** 2]]
        # Draw samples from the multivariate normal distribution
        # samples = np.random.multivariate_normal(mean, cov, size=(data_dict['N'], data_dict['R']))
        # # Split the samples into vecAT and vec
        # vecAT = samples[:, :, 0]
        # vec = samples[:, :, 1]

        # Function to generate truncated normal samples
        def truncated_normal(mean, sd, low, upp, size):
            a, b = (low - mean) / sd, (upp - mean) / sd
            return truncnorm(a, b, loc=mean, scale=sd).rvs(size=size)

        # Draw samples from the multivariate normal distribution
        samples = np.random.multivariate_normal(mean, cov, size=(data_dict['N'], data_dict['R']))

        # Split the samples into vecAT and vec
        vecAT = samples[:, :, 0]

        # Generate truncated normal samples for vec
        vec_mean = mean[1]
        vec_sd = np.sqrt(cov[1][1])
        vec = truncated_normal(vec_mean, vec_sd, -np.inf, 0, size=(data_dict['N'], data_dict['R']))

        data_dict['Beta_AT'] = vecAT
        data_dict['Beta_FEE'] = vec

    # vec[vec > 0] *= -1  # we manually make sure that every beta is < 0.

    # # Iterate through each element and redraw if it's positive
    # for i in range(vec.shape[0]):
    #     for j in range(vec.shape[1]):
    #         while vec[i, j] > 0:
    #             vec[i, j] = np.random.normal(-32.328, 14.168)
    #
    # for i in range(vec.shape[0]):
    #     for j in range(vec.shape[1]):
    #         while (vec[i, j] + data_dict['Beta_FEE_INC_PSP'] * data_dict['Low_inc'][i]
    #                + data_dict['Beta_FEE_RES_PSP'] * data_dict['Res'][i] > 0) or \
    #                 (vec[i, j] + data_dict['Beta_FEE_INC_PUP'] * data_dict['Low_inc'][i]
    #                  + data_dict['Beta_FEE_RES_PUP'] * data_dict['Res'][i] > 0):
    #             vec[i, j] = np.random.normal(-32.328, 14.168)

    # data_dict['Beta_AT'] = vecAT
    # data_dict['Beta_FEE'] = vec

    ### Alternatives' features
    # Access times to parking (AT)
    data_dict['AT_FSP'] = 10
    data_dict['AT_PSP'] = 10
    data_dict['AT_PUP'] = 5

    # Access time to final destination from the parking space (TD)
    data_dict['TD_FSP'] = 10
    data_dict['TD_PSP'] = 10
    data_dict['TD_PUP'] = 10

    # Number of PSP alternatives in the choice set
    data_dict['J_PSP'] = J_PSP
    # Number of PUP alternatives in the choice set
    data_dict['J_PUP'] = J_PUP
    # Number of opt-out alternatives in the choice set
    data_dict['J_opt_out'] = 1
    # Size of the universal choice set
    data_dict['J_tot'] = data_dict['J_PSP'] + data_dict['J_PUP'] + data_dict['J_opt_out']


def discrete_choice_model_logit(data_dict, N, J_PSP=1, J_PUP=1):
    # Define the type of discrete choice model
    data_dict['DCM'] = 'Logit'
    data_dict['N'] = N

    ##########################################################
    # DISCRETE CHOICE MODEL PARAMETERS
    ##########################################################

    # Alternative Specific Coefficients
    data_dict['ASC_FSP'] = 0.0  # Free Street Parking
    data_dict['ASC_PSP'] = 32.0  # Paid Street Parking
    data_dict['ASC_PUP'] = 34.0  # Paid Underground Parking

    # Beta coefficients
    data_dict['Beta_TD'] = -0.612
    data_dict['Beta_Origin'] = -5.762
    data_dict['Beta_Age_Veh'] = 4.037
    data_dict['Beta_FEE_INC_PSP'] = -10.995
    data_dict['Beta_FEE_RES_PSP'] = -11.440
    data_dict['Beta_FEE_INC_PUP'] = -13.729
    data_dict['Beta_FEE_RES_PUP'] = -10.668

    # Access time coefficient (random parameter with normal distribution)
    data_dict['Beta_AT'] = np.random.normal(-0.788, 1.064, size=(data_dict['N'], data_dict['R']))
    # Fee coefficient (random parameter with normal distribution)
    vec = np.random.normal(-32.328, 14.168, size=(data_dict['N'], data_dict['R']))
    vec[vec > 0] *= -1  # we manually make sure that every beta is < 0.

    # Iterate through each element and redraw if it's positive
    for i in range(vec.shape[0]):
        for j in range(vec.shape[1]):
            while vec[i, j] > 0:
                vec[i, j] = np.random.normal(-32.328, 14.168)

    data_dict['Beta_FEE'] = vec

    ### Alternatives' features
    # Access times to parking (AT)
    data_dict['AT_FSP'] = 10
    data_dict['AT_PSP'] = 10
    data_dict['AT_PUP'] = 5

    # Access time to final destination from the parking space (TD)
    data_dict['TD_FSP'] = 10
    data_dict['TD_PSP'] = 10
    data_dict['TD_PUP'] = 10

    # Number of PSP alternatives in the choice set
    data_dict['J_PSP'] = J_PSP
    # Number of PUP alternatives in the choice set
    data_dict['J_PUP'] = J_PUP
    # Number of opt-out alternatives in the choice set
    data_dict['J_opt_out'] = 1
    # Size of the universal choice set
    data_dict['J_tot'] = data_dict['J_PSP'] + data_dict['J_PUP'] + data_dict['J_opt_out']


def demand(N, data_dict, more_classes=False):
    if N <= 197:
        if N == 10:  # customers are ordered. Thus we can simply insert Vis classes as such:
            # Number of customers
            data_dict['Pop'] = 10

            ##########################################################
            # Customer socio-economic characteristics:
            # origin, age of vehicle, income, resident
            ##########################################################
            data_dict['Origin'] = np.array(
                [0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
            )
            data_dict['Age_veh'] = np.array(
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
            )

            if more_classes:
                data_dict['Low_inc'] = np.array(
                    [1, 1, 1, 1, 1, 0, 1, 3, 1, 1]
                )
            else:
                data_dict['Low_inc'] = np.array(
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
                )
            data_dict['Res'] = np.array(
                [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]
            )
        else:
            # Number of customers
            data_dict['Pop'] = 197

            ##########################################################
            # Customer socio-economic characteristics:
            # origin, age of vehicle, income, resident
            ##########################################################
            data_dict['Origin'] = np.array(
                [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
                 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
                 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
                 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
                 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0,
                 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
            )
            data_dict['Age_veh'] = np.array(
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,
                 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0,
                 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
            )

            if more_classes:
                data_dict['Low_inc'] = np.array(
                    [1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 1, 0, 1, 0, 1, 3, 3, 1, 1, 2, 2, 3,
                     3, 3, 2, 3, 1, 3, 3, 3, 3, 1, 0, 1, 2, 1, 2, 3, 0, 3, 1, 3, 0, 3,
                     3, 2, 0, 1, 1, 0, 3, 0, 3, 1, 1, 3, 3, 3, 1, 1, 3, 1, 1, 2, 1, 0,
                     3, 3, 1, 1, 0, 3, 1, 3, 0, 2, 3, 0, 1, 1, 3, 3, 3, 3, 3, 3, 2, 0,
                     3, 2, 1, 3, 3, 2, 3, 1, 0, 1, 3, 0, 0, 1, 1, 1, 3, 0, 1, 2, 1, 1,
                     2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 3, 1, 0, 3, 1, 2, 2, 2, 2, 1, 2, 3,
                     2, 3, 1, 3, 3, 1, 3, 3, 1, 1, 3, 1, 2, 3, 0, 0, 3, 2, 2, 3, 1, 1,
                     1, 1, 3, 3, 1, 0, 1, 1, 0, 3, 3, 1, 0, 1, 0, 1, 0, 3, 3, 1, 1, 0,
                     0, 1, 2, 2, 0, 0, 3, 3, 2, 3, 1, 1, 3, 3, 0, 1, 2, 3, 1, 1, 1]
                )
            else:
                data_dict['Low_inc'] = np.array(
                    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                     1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1,
                     1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
                     1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                     1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                     0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1,
                     0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
                     1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
                     0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]

                    # more equally distributed
                    # [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                    #  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                    #  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                    #  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                    #  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                    #  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                    #  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                    #  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                    #  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
                )
            data_dict['Res'] = np.array(
                [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0,
                 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
                 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
                 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1,
                 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1,
                 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0]
            )
    else:
        # for each attribute: count how many elements (distribution)
        # ask chat gpt how to build a list with these elements and these distributions

        # Number of customers
        data_dict['Pop'] = N

        ##########################################################
        # Customer socio-economic characteristics:
        # origin, age of vehicle, income, resident
        ##########################################################
        proportion_0_origin = 0.5532994923857868
        # Calculate the number of 0s and 1s in the new list
        num_0s_new = int(proportion_0_origin * N)
        num_1s_new = N - num_0s_new  # Ensuring the total length is n

        # Generate the new list and shuffle it
        new_list = [0] * num_0s_new + [1] * num_1s_new
        random.shuffle(new_list)
        data_dict['Origin'] = np.array(new_list)

        proportion_0_AgeVeh = 0.6243654822335025
        # Calculate the number of 0s and 1s in the new list
        num_0s_new = int(proportion_0_AgeVeh * N)
        num_1s_new = N - num_0s_new  # Ensuring the total length is n

        # Generate the new list and shuffle it
        new_list = [0] * num_0s_new + [1] * num_1s_new
        random.shuffle(new_list)
        data_dict['Age_veh'] = np.array(new_list)

        if more_classes:
            counts = {1: 72, 0: 32, 3: 66, 2: 27}
            total_count = sum(counts.values())
            proportions = {key: value / total_count for key, value in counts.items()}
            expected_counts = {key: round(proportion * N) for key, proportion in proportions.items()}

            # Directly adjust the counts if the total doesn't match N
            diff = N - sum(expected_counts.values())
            if diff > 0:
                for key in sorted(proportions, key=proportions.get, reverse=True):
                    while diff > 0 and expected_counts[key] < counts[key] * N / total_count:
                        expected_counts[key] += 1
                        diff -= 1
            elif diff < 0:
                for key in sorted(proportions, key=proportions.get):
                    while diff < 0 and expected_counts[key] > 0:
                        expected_counts[key] -= 1
                        diff += 1

            new_list = []
            for number, count in expected_counts.items():
                new_list.extend([number] * count)
            random.shuffle(new_list)
            data_dict['Low_inc'] = np.array(new_list)
        else:
            proportion_0_Low_inc = 0.29949238578680204
            # Calculate the number of 0s and 1s in the new list
            num_0s_new = int(proportion_0_Low_inc * N)
            num_1s_new = N - num_0s_new  # Ensuring the total length is n

            # Generate the new list and shuffle it
            new_list = [0] * num_0s_new + [1] * num_1s_new
            random.shuffle(new_list)
            data_dict['Low_inc'] = np.array(new_list)

        proportion_0_Res = 0.467005076142132
        # Calculate the number of 0s and 1s in the new list
        num_0s_new = int(proportion_0_Res * N)
        num_1s_new = N - num_0s_new  # Ensuring the total length is n

        # Generate the new list and shuffle it
        new_list = [0] * num_0s_new + [1] * num_1s_new
        random.shuffle(new_list)
        data_dict['Res'] = np.array(new_list)


def getData(N, R, J_PSP=1, J_PUP=1, more_classes=False):
    '''Construct a dictionary 'dict' containing all the input data'''

    # Initialize the output dictionary
    data_dict = {}

    # Name of the instance
    data_dict['Instance'] = 'Parking_MixedLogit'

    # Number of draws
    data_dict['R'] = R

    # Set random seed
    np.random.seed(10)

    # 1) Read discrete choice model parameters
    # 2) Read supply data
    # 3) Read demand data
    # 4) Generate groups of customers
    demand(N, data_dict, more_classes)
    discrete_choice_model(data_dict, N, J_PSP, J_PUP, more_classes)


    # Random term (Gumbel distributed 0,1)
    data_dict['xi'] = np.random.gumbel(size=(data_dict['J_tot'], data_dict['N'], data_dict['R']))

    ##########################################################
    # Deepcopy of the initial data (for restarts)
    ##########################################################
    data_dict['initial_data'] = copy.deepcopy(data_dict)

    return data_dict


def getData_logit(N, R, J_PSP=1, J_PUP=1):
    '''Construct a dictionary 'dict' containing all the input data'''

    # Initialize the output dictionary
    data_dict = {}

    # Name of the instance
    data_dict['Instance'] = 'Logit'

    # Number of draws
    data_dict['R'] = R

    # Set random seed
    np.random.seed(10)

    # 1) Read discrete choice model parameters
    # 2) Read supply data
    # 3) Read demand data
    # 4) Generate groups of customers
    discrete_choice_model_logit(data_dict, N, J_PSP, J_PUP)
    demand(data_dict)

    # Random term (Gumbel distributed 0,1)
    # data_dict['xi'] = np.random.gumbel(size=(data_dict['J_tot'], data_dict['N'], data_dict['R']))
    data_dict['xi'] = np.zeros((data_dict['J_tot'], data_dict['N'], data_dict['R']))
    ##########################################################
    # Deepcopy of the initial data (for restarts)
    ##########################################################
    data_dict['initial_data'] = copy.deepcopy(data_dict)

    return data_dict


def preprocessUtilities(data_dict, more_classes, minutes):
    ##########################################################
    # Exogenous utilities and endogenous parameters
    ##########################################################
    exo_utility = np.empty([data_dict['J_tot'], data_dict['N'], data_dict['R']])
    # h_utility = np.empty([data_dict['J_tot'], data_dict['N']])

    J_PSP = data_dict['J_PSP']
    low_PSP = 1
    up_PSP = 2
    J_PUP = data_dict['J_PUP']
    low_PUP = 0.5
    up_PUP = 1
    for n in range(data_dict['N']):
        for r in range(data_dict['R']):
            # Opt-Out
            exo_utility[0, n, r] = (data_dict['Beta_AT'][n, r] * data_dict['AT_FSP'] +
                                    data_dict['Beta_TD'] * data_dict['TD_FSP'] +
                                    data_dict['Beta_Origin'] * data_dict['Origin'][n] +
                                    data_dict["xi"][0, n, r])
            # PSP
            if J_PSP == 1:
                exo_utility[1, n, r] = (data_dict['ASC_PSP'] +
                                        data_dict['Beta_AT'][n, r] * data_dict['AT_PSP'] +
                                        data_dict['Beta_TD'] * data_dict['TD_PSP'] + data_dict["xi"][1, n, r])
                # print(f"PSP {1} gets Travel distance = ", data_dict['TD_PSP'])
            else:
                for i in range(J_PSP):
                    # if we have more than one PSP alternative we scale the
                    # TD: access time to destination from the parking space
                    # within bounds that are [low, up], divided into as many PSPs as we want, i.e.
                    # TD_i = low * TD + ((J_PSP - 1 - i) / (J_PSP - 1)) * (up - low) * TD
                    # where i is indexed from 0 to J_PSP - 1
                    # TDi = low_PSP * data_dict['TD_PSP'] \
                    #       + (J_PSP - 1 - i) / (J_PSP - 1) * (up_PSP - low_PSP) * data_dict['TD_PSP']
                    # what if we just want to have a small penalty for new PSPs?
                    # ADi = data_dict['AT_PSP'] - 2 * i
                    if minutes > 0:
                        TDi = data_dict['TD_PSP'] + minutes * i
                    else:
                        TDi = data_dict['TD_PSP'] + 5 * i
                    exo_utility[i + 1, n, r] = (data_dict['ASC_PSP'] +
                                                data_dict['Beta_AT'][n, r] * data_dict['AT_PSP'] +
                                                data_dict['Beta_TD'] * TDi +
                                                data_dict["xi"][i + 1, n, r])
                    # print(f"PSP {i} gets Travel distance = ", TDi)
            # PUP
            if J_PUP == 1:
                exo_utility[J_PSP + 1, n, r] = (data_dict['ASC_PUP'] +
                                                data_dict['Beta_AT'][n, r] * data_dict['AT_PUP'] +
                                                data_dict['Beta_TD'] * data_dict['TD_PUP'] +
                                                data_dict['Beta_Age_Veh'] * data_dict['Age_veh'][n] +
                                                data_dict["xi"][J_PSP + 1, n, r])
                # print(f"PUP {1} gets Travel distance = ", data_dict['TD_PUP'])
            else:
                for i in range(J_PUP):
                    # if we have more than one PUP alternative we scale the
                    # TD: access time to destination from the parking space
                    # within bounds that are [low, up], divided into as many PUPs as we want, i.e.
                    # TD_i = low * TD + ((J_PUP - 1 - i) / (J_PUP - 1)) * (up - low) * TD
                    # where i is indexed from 0 to J_PUP - 1
                    # TDi = low_PUP * data_dict['TD_PUP'] \
                    #       + (J_PUP - 1 - i) / (J_PUP - 1) * (up_PUP - low_PUP) * data_dict[
                    #     'TD_PUP']
                    # ADi = data_dict['AT_PUP'] - 2 * i
                    if minutes > 0:
                        TDi = data_dict['TD_PUP'] + minutes * i
                    else:
                        TDi = data_dict['TD_PUP'] + 5 * i
                    exo_utility[i + J_PSP + 1, n, r] = (data_dict['ASC_PUP'] +
                                                        data_dict['Beta_AT'][n, r] * data_dict['AT_PUP'] +
                                                        data_dict['Beta_TD'] * TDi +
                                                        data_dict["xi"][i + J_PSP + 1, n, r])
                    # print(f"PUP {i} gets Travel distance = ", TDi)
    data_dict['exo_utility'] = exo_utility
    # Beta coefficient for endogenous variables
    beta_FEE_PSP = np.empty([data_dict['N'], data_dict['R']])
    beta_FEE_PUP = np.empty([data_dict['N'], data_dict['R']])

    if more_classes:
        for n in range(data_dict['N']):
            for r in range(data_dict['R']):
                beta_FEE_PSP[n, r] = (data_dict['Beta_FEE'][n, r] +
                                      data_dict['Beta_FEE_INC_PSP'] * (data_dict['Low_inc'][n] == 1) +
                                      data_dict['Beta_FEE_INC_PSP_POOR'] * (data_dict['Low_inc'][n] == 3) +
                                      data_dict['Beta_FEE_INC_PSP_RICH'] * (data_dict['Low_inc'][n] == 2) +
                                      data_dict['Beta_FEE_RES_PSP'] * data_dict['Res'][n])
                beta_FEE_PUP[n, r] = (data_dict['Beta_FEE'][n, r] +
                                      data_dict['Beta_FEE_INC_PUP'] * (data_dict['Low_inc'][n] == 1) +
                                      data_dict['Beta_FEE_INC_PUP_POOR'] * (data_dict['Low_inc'][n] == 3) +
                                      data_dict['Beta_FEE_INC_PUP_RICH'] * (data_dict['Low_inc'][n] == 2) +
                                      data_dict['Beta_FEE_RES_PUP'] * data_dict['Res'][n])
    else:
        for n in range(data_dict['N']):
            for r in range(data_dict['R']):
                beta_FEE_PSP[n, r] = (data_dict['Beta_FEE'][n, r] +
                                      data_dict['Beta_FEE_INC_PSP'] * data_dict['Low_inc'][n] +
                                      data_dict['Beta_FEE_RES_PSP'] * data_dict['Res'][n])
                beta_FEE_PUP[n, r] = (data_dict['Beta_FEE'][n, r] +
                                      data_dict['Beta_FEE_INC_PUP'] * data_dict['Low_inc'][n] +
                                      data_dict['Beta_FEE_RES_PUP'] * data_dict['Res'][n])

    opt_out_matrix = np.zeros([1, data_dict['N'], data_dict['R']])
    repeat_beta_FEE_PSP = np.repeat(beta_FEE_PSP[np.newaxis, ...], J_PSP, axis=0)
    repeat_beta_FEE_PUP = np.repeat(beta_FEE_PUP[np.newaxis, ...], J_PUP, axis=0)

    # Concatenate the arrays along the first dimension (axis 0)
    endo_coef_array = np.concatenate((opt_out_matrix, repeat_beta_FEE_PSP, repeat_beta_FEE_PUP), axis=0)
    data_dict['endo_coef'] = endo_coef_array

    # print(f"lets say all prices are fixed to 0.5, then the utilities for n, r = 0 are:")
    # for i in range(J_PSP + J_PUP + 1):
    #     print(f"util_{i} = ", exo_utility[i, 0, 0] + 0.5 * endo_coef_array[i, 0, 0])


def get_input_data_parking(N, R, J_PSP=1, J_PUP=1, more_classes=False, minutes=0):
    # Read instance
    data = getData(N, R, J_PSP, J_PUP, more_classes)
    # Precompute exogenous part of the utility and beta_cost parameters

    # Given data_dict['J_tot'], data_dict['N'], data_dict['R'] and some beta stuff compute the actual exo and endo terms
    preprocessUtilities(data, more_classes, minutes)
    return data


def get_input_data_parking_logit(N, R, J_PSP=1, J_PUP=1):
    # Read instance
    data = getData_logit(N, R, J_PSP, J_PUP)
    # Precompute exogenous part of the utility and beta_cost parameters

    # Given data_dict['J_tot'], data_dict['N'], data_dict['R'] and some beta stuff compute the actual exo and endo terms
    preprocessUtilities(data)
    return data


if __name__ == '__main__':
    N = 3
    R = 3
    # Read instance
    data = getData(N, R, J_PSP=2, J_PUP=2)
    # Precompute exogenous part of the utility and beta_cost parameters
    preprocessUtilities(data)
