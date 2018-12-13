import json
import csv
from copy import deepcopy
import codecs
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
import itertools
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from operator import itemgetter, attrgetter

def get_forecast(dff, itemname, Results, endDate, show_startDate):
    try:

        df = deepcopy(dff)
        results = deepcopy(Results)
        # Get forecast 500 steps ahead in future
        forecastDate = 7
        pred_uc = results.get_forecast(steps=forecastDate) #or step = pd.to_datetime('2018-11-20')
        
        # alpha parameter. alpha=0.05 implies 95% CI
        pred_ci = pred_uc.conf_int(alpha=0.05)
        # print(df)
        # print(pred_ci)

        ax = df[show_startDate:]['數量'].plot(label='observed', color='b', figsize = (10, 7)) # figsize=(20, 15)
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast', style = "r--", alpha=.7)
        # ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)

        ax.set_title('SARIMA Forecasting for next ' + str(forecastDate) + ' days')
        ax.set_xlabel('Date')
        ax.set_ylabel('Customer Flow')
        ax.legend(frameon=False, loc='lower center', ncol=2)
        plt.savefig('佐麥咖啡//byItem//SARIMAX_' + itemname + '_' + endDate + '-next' + str(forecastDate) + 'days_getforecast.png', bbox_inches='tight', pad_inches=0.5, figsize = (10, 7))
        plt.show()
        # plt.close()
    except Exception as ex:
        print(ex.with_traceback())

def SARIMAX(dff, itemname, test_predictdate, plot_startdate, select_startdate, T):
    try:
        df = deepcopy(dff)
        df = pd.DataFrame(np.array(pd.to_numeric(df['數量'], downcast='signed')), index=df.index, columns=['數量'])        # print(df)        # print(type(df))

        p = d = q = range(0, 2) # Define the p, d and q parameters to take any value between 0 and 2
        pdq = list(itertools.product(p, d, q)) # Generate all different combinations of p, q and q triplets

        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
        # print(pdq)
        # print(seasonal_pdq)
        # print('Examples of parameter combinations for Seasonal ARIMA...')
        # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
        # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
        # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
        # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

        # warnings.filterwarnings("ignore") # specify to ignore warning messages
        #
        # for param in pdq:
        #     for param_seasonal in seasonal_pdq:
        #         try:
        #             mod = sm.tsa.statespace.SARIMAX(df2,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,
        #                                             enforce_invertibility=False)
        #             results = mod.fit()
        #             # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        #         except:
        #             continue

        p1 = 3 #趨勢自動回歸
        d1 = 1 #趨勢差異順序
        q1 = 4 #趨勢均線訂單
        P = 4 #季節性自回歸
        D = 1 #季節性差異
        Q = 4 #季節性移動平均
        M = 7 #單季節期間 ex. month=12, seasonal=4, day=365.25
        order = [p1, d1, q1]
        seasonal_order =[P, D, Q, M]

        #Training Data
        mod = sm.tsa.statespace.SARIMAX(df, order=order, seasonal_order=seasonal_order,enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit()
        # results.plot_diagnostics(figsize=(8, 6))

        pred = results.get_prediction(start=pd.to_datetime(test_predictdate), dynamic=False)
        pred_ci = pred.conf_int(alpha=0.05)

        plt.style.use('fivethirtyeight')
        ax = df[plot_startdate:].plot(label="Original", color='k', figsize=(10, 7))
        pred.predicted_mean.plot(ax=ax, label='Test Forecast', color = 'y', alpha=.7)

        # ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Customer Flow')
        ax.legend(frameon=False, loc='lower center', ncol=2)
        # plt.show()

        #save image
        orderlist = "".join(str(x) for x in order)
        seasonal_orderlist = "".join(str(x) for x in seasonal_order)

        if (T == True):
            ax.set_title('SARIMAX MODEL (original)')
            plt.savefig('佐麥咖啡//byItem//SARIMAX(original) ' + itemname + '_' + orderlist + '_Season' + seasonal_orderlist + '_' + select_startdate + '.png', bbox_inches='tight', pad_inches=0.5, figsize = (10, 7))
        else:
            ax.set_title('SARIMAX MODEL (adjusted)')
            plt.savefig('佐麥咖啡//byItem//SARIMAX(adjusted) ' + itemname + '_' + orderlist + '_Season' + seasonal_orderlist + '_' + select_startdate + '.png', bbox_inches='tight', pad_inches=0.5, figsize = (10, 7))
        # plt.show()
        plt.clf()
        measureError(df, pred, test_predictdate)
        return results
    except Exception as ex:
        print(ex.with_traceback())

def MSE(difference): #傳入 difference[] = forecast - truth 
    squares = []
    for j in range(len(difference)):
        number = difference[j] ** 2
        squares.append(number)
    mse = sum(range(len(squares)))/len(squares)
    return mse

def measureError(df, pred, predictdate):
    try:
        df_forecasted = pd.DataFrame(pred.predicted_mean)
        df_forecasted['數量'] = df_forecasted.values
        # df_forecasted['日期'] = df_forecasted.index
        df_forecasted = df_forecasted.drop([0], axis='columns')

        df_truth = df['數量']
        # df_truth['日期'] = df_truth.index
        df_truth  = df[predictdate:]

        n = []
        for i in df_truth.index:
            for j in df_forecasted.index:
                if (i == j):
                    diff = df_forecasted["數量"][j] - df_truth["數量"][i]
                    n.append(diff)

        df3 = pd.DataFrame(n, index=df_forecasted.index, columns=['誤差'])
        # print(df3)

        # Bias, MAE, MSE, RMSE (List = [])
        forecast_error = n
        bias = sum(forecast_error)*1.0/len(n)
        print('The Bias of our forecasts is {}'.format(bias))

        mae = mean_absolute_error(df_forecasted['數量'], df_truth['數量'])
        print('The MAE of our forecasts is {}'.format(mae))

        # mse = mean_squared_error(df_forecasted['來客數'], df_truth['來客數'])/len(df_forecasted)
        mse = MSE(n)
        print('The MSE of our forecasts is {}'.format(mse))

        rmse = sqrt(mse)
        print('The RMSE of our forecasts is {}'.format(rmse))

        record(bias, mae, mse, rmse)
    except Exception as ex:
        print(ex.with_traceback())

def ListItem(df):
    try:
        DF = deepcopy(df)
        menu = DF.drop_duplicates('品項')
        menu = menu.reset_index(drop = True)
        menu = menu.drop(['日期','數量'], axis = 'columns') # dataframe type
        # menuList = menu['品項'].tolist()    # List type
        print(menu) 
        return menu

    except Exception as ex:
        print(ex.with_traceback())

def SelectItem(df, itemlist):
    try:
        DF = deepcopy(df)
        num = input('Select Item number: ')
        ItemName = itemlist['品項'][int(num)]
        print('Selected: ', ItemName)
        return ItemName
    except Exception as ex:
        print(ex.with_traceback())

def ItemSum(df, item):
    try:
        DF = deepcopy(df)
        DF = DF.loc[DF['品項'] == item] # data with selected-item
        DF = DF.groupby('日期')['數量'].sum().to_frame() # add quantity by date
        return DF
    except Exception as ex:
        print(ex.with_traceback())

def ItemData(filename):
    try:
        df = pd.read_csv(filename, delimiter=',', usecols=['日期', '品項', '數量'])
        df = df[['日期', '品項', '數量']]
        df['日期'] = pd.to_datetime(df["日期"])
        df = df.sort_values(by='日期')
        df.index = df['日期']
        df.index = df.index.normalize() # remove time
        df['日期'] = df.index
        return df
    except Exception as ex:
        print(ex.with_traceback())

def SelectRange(data, select_Start, select_End):
    try:
        df = data.loc[select_Start:select_End]
        return df
    except Exception as ex:
        print(ex.with_traceback())

def FillMissingDate(df, startDate, endDate):
    try:
        data = deepcopy(df)
        alldates = pd.date_range(start = startDate, end= endDate)
        alldatesData = alldates.to_frame(index = True)
        alldatesData['日期'] = alldatesData[0]
        alldatesData = alldatesData.drop([0], axis = 'columns')

        # merge data
        mergeresult = pd.concat([alldatesData, data], axis = 'columns', join_axes=[alldatesData.index]) #以左方資料index為主
        mergeresult = mergeresult.fillna(0)

        day = []
        for i in mergeresult.index:
            daynumber = dow(i)
            day.append(daynumber)
        mergeresult['星期'] = day
        mergeresult['日期'] = mergeresult.index
        return mergeresult

    except Exception as ex:
        print(ex.with_traceback())

def record(bias, mae, mse, rmse):
    try:
        f = open('佐麥咖啡//byItem//誤差評估.txt', 'a')
        f.write('\n(1) The Bias of our forecasts is {}'.format(bias))
        f.write('\n(2) The MAE of our forecasts is {}'.format(mae))
        f.write('\n(3) The MSE of our forecasts is {}'.format(mse))
        f.write('\n(4) The RMSE of our forecasts is {}'.format(rmse))
        f.write('\n-------------------\n')
        f.close()
    except Exception as ex:
       print(ex.with_traceback())

def jsontocsv(filename):
    try:
        f = open('佐麥咖啡//byItem//' + filename + '.json')
        data = json.load(f)
        # print(json.dumps(data, ensure_ascii = False, indent = 4))

        date=[]
        name=[]
        quantity=[]

        for i in range(len(data)):
            # multiple items
            for j in range(len(data[i].get("Items"))):
                date.append(data[i].get("date"))
                name.append(data[i].get("Items")[j].get("name"))
                quantity.append(data[i].get("Items")[j].get("quantity"))

        # date, name, quantity
        all_context=[]
        with open('佐麥咖啡//byItem//' + filename + '_ItemData.csv', 'w', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['日期'] + ['品項'] + ['數量'])
            for i in range(len(date)):
                context=[]
                context=[date[i], name[i], quantity[i]]
                writer.writerow(context)
                all_context.append(context)
        csvfile.close()
        f.close()
        # return all_context
        # print(all_context)
    except Exception as ex:
        print(ex.with_traceback())

def save_OriginalPlot(df, Pic_name):
    try:
        ts = df['數量']
        plt.plot(ts)
        plt.savefig('佐麥咖啡//byItem//' + Pic_name + '.png')
        plt.close()
    except Exception as ex:
        print(ex.with_traceback())

def writeCSV(file, filename):
    try:
        file.to_csv('佐麥咖啡//byItem//' + filename + '.csv', index = False, encoding='utf-8')

    except Exception as ex:
       print(ex.with_traceback())

def dow(date):
    days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dayNumber=date.weekday()
    return days[dayNumber]

def mergefile(f1, f2):
    try:
        merge_df = f1.append(f2)
        return merge_df
    except Exception as ex:
        print(ex.with_traceback())

def remove_outlier(df):
    try:
        DF = deepcopy(df)
        days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        for i in days:
            M = DF.loc[(DF['星期'] == i) & (DF['outlier'] == False)]['數量'].mean()
            DF.loc[(DF['星期'] == i) & (DF['outlier'] == True), '數量'] = M
        DF = DF.drop(['z-score', 'outlier'], axis='columns')
        return DF

    except Exception as ex:
        print(ex.with_traceback())

def StatisticInfo(df, itemname, startDate, endDate, predictDate, T):
    try:
        DFF = deepcopy(df)
        f = open('佐麥咖啡//byItem//誤差評估.txt', 'a')

        # Box plot
        plt.style.use('fivethirtyeight')
        p = DFF.boxplot(whis=2, figsize = (10, 7)) # lower bound = Q1 - (whis*Q1) | upper bound = Q3 + (whis*Q3) | confidence internval = 95%
        p.set_title('Testing Outlier', fontsize=5)
        p.set_ylabel('customer flow', fontsize=5)
        # p.set_xlabel('DataSet')
        plt.setp(p, xticklabels=['DataSet'])
        
        # add z-score & determine outlier
        z = np.abs(stats.zscore(DFF['數量']))
        DFF['z-score'] = z
        out=[]
        for i in DFF['z-score']:
            if (i < 2): #正負2倍標準差以外 95%
                out.append(False)
            else:
                out.append(True)
        DFF['outlier'] = out
        describes = DFF.describe()

        if (T == True):
            f.write('--------------------------------------------\n')
            f.write('----------------  ' + itemname + '  --------------\n')
            f.write('Selecting date from ' + startDate + ' to ' + endDate + '\n')
            f.write('--------------------------------------------\n')
            plt.savefig('佐麥咖啡//byItem//BoxplotwithOutlier' + itemname + '_' + startDate + ' to ' + endDate + '.png', bbox_inches='tight', pad_inches=0.5, figsize = (10, 7))
            f.write('Original data: \n' + str(describes) + '\n')
        else:
            plt.savefig('佐麥咖啡//byItem//BoxplotwithoutOutlier' + itemname + '_' + startDate + ' to ' + endDate + '.png', bbox_inches='tight', pad_inches=0.5, figsize = (10, 7))
            # f.write('\nnumber of outlier: ' + str([df['outlier'] == True].index))
            f.write('\nAdjusted data: \n' + str(describes) + '\n')
        
        # plt.show()
        f.close()
        return DFF
    except Exception as ex:
        print(ex.with_traceback())

def main():
    real_startDate = '2017-06-27'
    real_endDate = '2018-03-31'
    # jsontocsv('佐麥咖啡20180401_20181105')
    # jsontocsv('佐麥咖啡20180401_20181105')

    df1 = ItemData('佐麥咖啡//byItem//佐麥咖啡20170627_20180331_ItemData.csv')
    df2 = ItemData('佐麥咖啡//byItem//empty.csv')
    mergeData = mergefile(df1,df2)

    ItemList = ListItem(mergeData)
    Item = SelectItem(mergeData, ItemList)
    df_Sum = ItemSum(mergeData, Item)
    FillData = FillMissingDate(df_Sum, real_startDate, real_endDate)
    writeCSV(FillData, '佐麥咖啡' + real_startDate + '_' + real_endDate + '_' + Item + '__allData')
    save_OriginalPlot(FillData, Item + '_' + real_startDate + '_' + real_endDate)

    select_startDate = '2017-11-10'  
    select_endDate = '2018-03-10'
    select_predictDate = '2018-02-24'
    show_startdate = '2018-01-10'
    Selected_Data = SelectRange(FillData, select_startDate, select_endDate)
    print(Selected_Data)
    stats_df = StatisticInfo(Selected_Data, Item, select_startDate, select_endDate, select_predictDate, T = True)
    writeCSV(stats_df, '佐麥咖啡' + real_startDate + '_' + real_endDate + '_' + Item + '_statsData')

    SARIMAX(Selected_Data, Item, select_predictDate, show_startdate, select_startDate, T = True) # with some outlier

    # remove outlier with 95% confidence interval
    adjusted_df = remove_outlier(stats_df)
    StatisticInfo(adjusted_df, Item, select_startDate, select_endDate, select_predictDate, T = False)

    result = SARIMAX(adjusted_df, Item, select_predictDate, show_startdate, select_startDate, T = False) #(訓練的資料（range-selected）, 預測起始日期(test), 圖表起始日期, select_startDate)
    get_forecast(adjusted_df, Item, result, select_endDate, show_startdate)

if __name__ == "__main__":
    main()
