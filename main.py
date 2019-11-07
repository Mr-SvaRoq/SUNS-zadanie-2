import pandas as pnds
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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
               "room_type", "bathrooms", "bedrooms", "beds",
               "bed_type", "amenities", "review_scores_rating"]

    columnsOutput = ["price", "guests_included", "extra_people", "accommodates"
                     ]
    host_is_superhost = ["t", "f"]

    room_type = {'Hotel room': 1,
                 'Entire home/apt': 2,
                 'Private room': 3,
                 'Shared room': 4}

    host_response_time = {'within an hour': 1, 'within a few hours': 2, 'within a day': 3, 'a few days or more': 4}

    rawData = pnds.read_csv("listings.csv", usecols=columns, skiprows=[i for i in range(2250, 2250)])
    rawDataOutput = pnds.read_csv("listings.csv", usecols=columnsOutput, skiprows=[i for i in range(2250, 2250)])

    rawData = rawData.replace(np.nan, 0, regex=True)
    rawData = rawData.replace('%', '', regex=True)
    rawData = rawData.replace('[\$)]', '', regex=True)
    rawData = rawData.replace('within an hour', 1, regex=True)
    rawData = rawData.replace('within a few hours', 2, regex=True)
    rawData = rawData.replace('within a day', 3, regex=True)
    rawData = rawData.replace('a few days or more', 4, regex=True)
    rawData = rawData.replace('t', 1, regex=False)
    rawData = rawData.replace('f', 0, regex=False)
    # rawData = rawData.replace({'t': 1, 'f': 0}, inplace=True)
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

    # rawData.hist(figsize=(20, 20));

    # for index, row in rawDataOutput.iterrows():
    #     print(row['price'])
    # print('*************************************')

    rawData.to_csv('rawData.csv', header=True)

    rawDataOutput = rawDataOutput.replace('[\$,)]', '', regex=True)
    rawDataOutput['price'] = rawDataOutput['price'].astype(float)

    rawDataOutputPriceForPerson = []
    for index, row in rawDataOutput.iterrows():
        rawDataOutputPriceForPerson.append(row['price'] / row['accommodates'])

    # average = np.mean(rawDataOutputPriceForPerson) - 4.005
    toCompare = 24.5

    rawDataOutputClass = []
    oneCounter = 0
    zeroCounter = 0
    for row in rawDataOutputPriceForPerson:
        if row > toCompare:
            rawDataOutputClass.append(1)
            oneCounter += 1
        else:
            rawDataOutputClass.append(0)
            zeroCounter += 1

    print(rawDataOutputPriceForPerson)
    print('**********************************22***')
    print(toCompare)
    print('*************************************')
    print(rawDataOutputClass)
    print('**************************233***********')
    print(oneCounter)
    print(zeroCounter)
    print('**************************233***********')

    rawDataOutput.to_csv('rawDataOutput.csv', header=True)

    room_type.to_csv('room_type.csv', header=True)

    # len pre nazov :D
    parsedData = rawData
    print(parsedData)
    print('\n---------NORMALIZACIA VSETKYCH DAT----------')
    scaler = MinMaxScaler()
    scaler.fit(parsedData)
    scaledAllData = scaler.transform(parsedData)

    scaler.fit(rawDataOutput)
    scaledAllDataOutput = scaler.transform(rawDataOutput)

    print(scaledAllData[2248])

    # df_x = scaledAllData.iloc[:, 1:]
    # df_y = scaledAllDataOutput.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(scaledAllData, rawDataOutputClass, test_size=0.2,
                                                        random_state=4)

    print('\n---------x_train----------')
    print(x_train)
    print('\n---------x_test----------')
    print(x_test)
    print('\n---------y_train----------')
    print(y_train)
    print('\n---------y_test----------')
    print(y_test)

    print('\n---------MLPClassifier ----------')
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(150, 100), random_state=4, verbose=5, max_iter=500, tol=0.0017)
    # learning_rate
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred) * 100

    print(accuracy, ' %')

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, center=True)
    plt.show()
 # price/accom   +  (daco/daco/2)
