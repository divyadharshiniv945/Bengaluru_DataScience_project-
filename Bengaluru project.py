import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# ////DATA CLEANING/////
df1=pd.read_csv("Bengaluru_House_Data.csv")
print(df1)
print("=========================================")
# (13320, 9) df1 shape
print("df1.shpe",df1.shape)
# # size-16,society-5502,bath-73,balcony-609  are having null values
print(df1.isnull().sum())
# print("==========================================")
seperate_areatype=df1.groupby('area_type')['area_type'].agg('count')
print(seperate_areatype)
# # LETS ASSUME THESE COLUMNS ARE NOT IMPORTANT
df2=df1.drop(['area_type', 'society','balcony','availability'],axis='columns')
print(df2.head())
# # size-16,total_sqft-0,bath-73
print(df2.isnull().sum())
print("---------------------------")
df3=df2.dropna()
print(df3.isnull().sum())
# # POSSIBLE VALUES IN SIZE
print(df3['size'].unique())
# (13246, 5) df3 shape
print(df3.shape)
# # BY CREATING A NEW COLUMN "bhk" DERIVED FROM "SIZE" COLUMN
df3['bhk']=df3['size'].apply(lambda x:int(x.split()[0]))
print(df3.head())
print(df3['bhk'].dtype) 
print(df3['bhk'].unique())
print(df3[df3['bhk']>20])
print(df3['total_sqft'].unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
print(df3[~df3['total_sqft'].apply(is_float)].head(10))
def convert_sqft_to_num(x):
    tokens=x.split('-')
 
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
print(convert_sqft_to_num('3090 - 5002'))
print(convert_sqft_to_num('34.46Sq. Meter'))
df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
print(df4.head(3))
print(df4.loc[410])

# # //// FEATURE ENGININEERING////

df5=df4.copy()
# # /// CREATING PRICE FOR PER SQ FEET
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
print(df5.head())
print(df5.isnull().sum())

# # //// REMOVING UN WANTED SPACES FROM START AND END
df5['location']=df5['location'].apply(lambda x:x.strip( ))
print(df5)
# # //////LOCATION GROUPBY COUNT//////
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)                           
print(location_stats.head(1293))
print('=================================')
less_than_ten=(location_stats[location_stats<=10])
print(less_than_ten)

def replace_location(z):
 
    if z in less_than_ten:
        return 'other'
    else:
        return z

df5['location'] = df5['location'].apply(replace_location)
print(df5.head(20))
o=(df5.location.unique())
print(o)
print(df5.shape)
# # //////OUTLIER DETECTION AND REMOVAL//////'
# #EG---> HOME=600 SQFEET 
# # TOTAL BEDROOM=9
# # SO HOME/TOTAL BEDROOM MEANS 600/9=66.66 
# # 66 IS NOT IDEAL TYPICAL BEDROOM SIZE  SO WE HAVE TO REMOVE IT
# # LIKE WISE 
# # --------> this is anomalies
print(df5[df5['total_sqft']/df5['bhk']<300])
print(df5.shape)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
df6=df5[~(df5['total_sqft']/df5['bhk']<300)]
print(df6.head())
print(df6.shape)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

print(df6['price_per_sqft'].describe())
# # 1% cost like
print(df6['price_per_sqft'].quantile(0.1))
a=df6.groupby('location')


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for place,data in df.groupby('location'):
        _mean=np.mean(data.price_per_sqft)
        standard_diviation=np.std(data.price_per_sqft)
# #         # 2140<=10476 limit for price_per_sqft
        reduced_df=data[
            (data['price_per_sqft']>(_mean-standard_diviation))&
            (data['price_per_sqft']<=(_mean+standard_diviation))
        ]
        df_out = pd.concat([df_out, reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
print(df7.head(100))
# # [2301 rows x 7 columns]
print(df7.shape)
print(df7.location.unique())

# IN SOME CASES LIKE  THE SIZE OF  "3 BHK =121O SQFT =81(PRICE)" BUT IN   "2 BHK=1268=127(PRICE) BECOZ OF THE LOCATION" 
# WHAT IF BOTH FROM SAME LOCATION SO IT WILL LEAD CONFUSION

def plot_scatter_chart(df,location):
    bhk2=df[(df['location']==location)&(df["bhk"]==2)]
    bhk3=df[(df['location']==location)&(df["bhk"]==3)]
    print(bhk2)
    plt.figure(figsize=(15,10))

    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 bhk',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 bhk',s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
    plt.show()
  
print(plot_scatter_chart(df6,"Kothannur"))

def remove_bhk_outliers(df):
       exclude_indices=np.array([])
       for location,location_df in df.groupby('location'):
           bhk_stats={}
           for bhk,bhk_df in location_df.groupby('bhk'):
               bhk_stats[bhk]={
                   'mean':np.mean(bhk_df.price_per_sqft),
                    'std':np.std(bhk_df.price_per_sqft),
                    'count':bhk_df.shape[0]
                                   }
           for bhk,bhk_df in location_df.groupby('bhk'):
               stats = bhk_stats.get(bhk-1)
               if stats and stats['count']>5:
                  exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
       return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
# print(df7.shape)
