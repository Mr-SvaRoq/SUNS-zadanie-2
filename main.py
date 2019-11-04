import pandas as pnds
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    amenities = ["TV", "Cable TV", "Internet", "Wifi", "Kitchen", "Air conditioning", "Paid parking off premises", "Free parking on premises",
     "Smoke detector", "heating", "washer", "dryer", "oven", "First aid kit", "Fire extinguisher"]

    columns = ["latitude", "longitude",
               "host_response_time", "host_response_rate", "host_is_superhost",
               "room_type", "accommodates", "bathrooms", "bedrooms", "beds",
               "bed_type", "amenities",
               "price", "review_scores_rating"]
    host_is_superhost = ["t", "f"]

    room_type = {'Hotel room': 1,
                 'Entire home/apt': 2,
                 'Private room':3,
                 'Shared room':4}

    host_response_time = {'within an hour': 1, 'within a few hours': 2, 'within a day': 3, 'a few days or more': 4}


    bed_type = ['Real Bed', 'Couch', 'Pull-out Sofa', 'Airbed', 'Futon']
    bedTypeDictionary = ['Real Bed', 'Couch', 'Pull-out Sofa', 'Airbed', 'Futon']
    # rowData = pnds.read_csv("listings.csv", usecols=["id"], skiprows=[i for i in range(10, 2250)])
    rowData = pnds.read_csv("listings.csv", usecols=columns, skiprows=[i for i in range(25, 2250)])
    # rowData = pnds.read_csv("listings.csv", usecols=columns)
    # print(rowData.amenities.unique())

    # rowData.at[0, "host_response_time"] = "KOKOKOIT"
    lala = len(rowData.index)

    for x in range(len(rowData.index)):

        # superhost
        if rowData.at[x, "host_is_superhost"] == "t":
            rowData.at[x, "host_is_superhost"] = 1
        elif rowData.at[x, "host_is_superhost"] == "f":
            rowData.at[x, "host_is_superhost"] = 0
        else:
            rowData.at[x, "host_is_superhost"] = -1 #ten isty problem bude ako pri host_response_time
            print(x)

        # room type
        if rowData.at[x, "room_type"] in room_type:
            rowData.at[x, "room_type"] = room_type[rowData.at[x, "room_type"]]
        else:
            rowData.at[x, "room_type"] = -1 #ten isty problem bude ako pri host_response_time

        # host_response_time
        if rowData.at[x, "host_response_time"] in host_response_time:
            rowData.at[x, "host_response_time"] = host_response_time[rowData.at[x, "host_response_time"]]
            print("DOBRE")
        else:
            rowData.at[x, "host_response_time"] = 69 #tu je problem, prepise to tu, ale ptm to nejako zmizne
            print(x)
            print(rowData.at[x, "host_response_time"])
            print("---ZLEE---")

        print(rowData.at[x, "host_response_time"])


    print(lala)
    print(rowData.room_type.unique())
    # print(rowData.host_is_superhost.at[208, "host_is_superhost"])
