import pandas as pnds
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # bed_type = ['Real Bed', 'Couch', 'Pull-out Sofa', 'Airbed', 'Futon']
    # bedTypeDictionary = ['Real Bed', 'Couch', 'Pull-orut Sofa', 'Airbed', 'Futon']
    # rawData = pnds.read_csv("listings.csv", usecols=["id"], skipraws=[i for i in range(10, 2250)])
    # rawData = pnds.read_csv("listings.csv", usecols=columns)
    # print(rawData['host_is_superhost'])
    # rawData['room_type'] = pnds.Categorical(rawData['room_type'])
    # print(rawData.room_type.unique())
    # dataWithoutNanAndPercentage = dataWithoutNan.replace('%', '', regex=True)
    # dataWithoutNanAndPercentageAndDolar = dataWithoutNanAndPercentage.replace('$', '', regex=True)

    # TO CO BERIEME DO UVAHY
    amenitiesToConsider = ["TV", "Internet", "Wifi", "Kitchen", "Air conditioning", "Paid parking off premises",
                           "Free parking on premises",
                           "Smoke detector", "heating", "washer", "dryer", "oven", "First aid kit", "Fire extinguisher"]

    columns = ["latitude", "longitude",
               "host_response_time", "host_response_rate", "host_is_superhost",
               "room_type", "accommodates", "bathrooms", "bedrooms", "beds",
               "bed_type", "amenities",
               "price", "review_scores_rating"]
    host_is_superhost = ["t", "f"]

    room_type = {'Hotel room': 1,
                 'Entire home/apt': 2,
                 'Private room': 3,
                 'Shared room': 4}

    host_response_time = {'within an hour': 1, 'within a few hours': 2, 'within a day': 3, 'a few days or more': 4}

    rawData = pnds.read_csv("listings.csv", usecols=columns, skiprows=[i for i in range(2250, 2250)])

    rawData = rawData.replace(np.nan, 0, regex=True)
    rawData = rawData.replace('%', '', regex=True)
    rawData = rawData.replace('[\$)]', '', regex=True)
    rawData = rawData.replace('within an hour', 1, regex=True)
    rawData = rawData.replace('within a few hours', 2, regex=True)
    rawData = rawData.replace('within a day', 3, regex=True)
    rawData = rawData.replace('a few days or more', 4, regex=True)
    rawData = rawData.replace('t', 1, regex=False)
    rawData = rawData.replace('f', 0, regex=False)
    rawData = rawData.replace(',', '', regex=True)

    room_type = pnds.get_dummies(rawData['room_type'], prefix='room_type')
    bed_type = pnds.get_dummies(rawData['bed_type'], prefix='bed_type')

    amenities = rawData['amenities'].replace('{', '', regex=True)
    amenities = amenities.replace('}', '', regex=True)
    amenities = amenities.replace('"', '', regex=True)

    data = []
    for index, row in amenities.iteritems():
        valueOfHost = 0
        for amenity in amenitiesToConsider:
            if amenity in row and (amenity != 'Paid parking off premises'):
                valueOfHost += 1
            elif amenity in row and (amenity == 'Paid parking off premises'):
                valueOfHost -= 1

        valueOfHost = (valueOfHost / len(amenitiesToConsider)) * 100
        data.append(valueOfHost)

    convertAmenities = pnds.DataFrame(data, columns=['amenities_%'])

    rawData = rawData.drop(columns=['room_type', 'bed_type', 'amenities'])

    rawData = pnds.concat([rawData, room_type], axis=1)
    rawData = pnds.concat([rawData, bed_type], axis=1)
    rawData = pnds.concat([rawData, convertAmenities], axis=1)

    rawData.to_csv('rawData.csv', header=True)
    room_type.to_csv('room_type.csv', header=True)

    # len pre nazov :D
    parsedData = rawData
    print(parsedData)
    print('\n---------NORMALIZACIA VSETKYCH DAT----------')
    scaler = MinMaxScaler()
    scaler.fit(parsedData)
    scaledAllData = scaler.transform(parsedData)

    print(scaledAllData[2248])

    # df_x = scaledAllData.iloc[:, 1:]
    # df_y = scaledAllData.iloc[:, 0]
    # x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

    # print(x_train)
    # print(x_test)
    splitArrayData = np.split(scaledAllData, [225, 2025])
    validacneData = splitArrayData[0]
    trenovacieData = splitArrayData[1]
    testovacieData = splitArrayData[2]

    print('\n---------validacneData----------')
    print(validacneData)
    print(len(splitArrayData[0]))
    print('\n---------trenovacieData----------')
    print(trenovacieData)
    print(len(splitArrayData[1]))
    print('\n---------testovacieData----------')
    print(testovacieData)
    print(len(splitArrayData[2]))

    print('\n---------SKUUUUSKA ? :D ----------')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # clf.fit(trenovacieData, validacneData)

    # for x in range(len(rawData.index)):
    #
    #     # superhost
    #     if rawData.at[x, "host_is_superhost"] == "t":
    #         rawData.at[x, "host_is_superhost"] = 1
    #     elif rawData.at[x, "host_is_superhost"] == "f":
    #         rawData.at[x, "host_is_superhost"] = 0
    #     else:
    #         rawData.at[x, "host_is_superhost"] = -1 #ten isty problem bude ako pri host_response_time
    #         print(x)
    #
    #     # room type
    #     if rawData.at[x, "room_type"] in room_type:
    #         rawData.at[x, "room_type"] = room_type[rawData.at[x, "room_type"]]
    #     else:
    #         rawData.at[x, "room_type"] = -1 #ten isty problem bude ako pri host_response_time
    #
    #     # host_response_time
    #     if rawData.at[x, "host_response_time"] in host_response_time:
    #         rawData.at[x, "host_response_time"] = host_response_time[rawData.at[x, "host_response_time"]]
    #         print("DOBRE")
    #     else:
    #         rawData.at[x, "host_response_time"] = 69 #tu je problem, prepise to tu, ale ptm to nejako zmizne
    #         print(x)
    #         print(rawData.at[x, "host_response_time"])
    #         print("---ZLEE---")
    #
    #     print(rawData.at[x, "host_response_time"])

    # print(rawData.room_type.unique())
    # print(rawData.host_is_superhost.at[208, "host_is_superhost"])
