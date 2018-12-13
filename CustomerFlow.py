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

def get_forecast(df, results, endDate, show_startDate):
    try:
        # Get forecast 500 steps ahead in future
        forecastDate = 7
        pred_uc = results.get_forecast(steps=forecastDate) #or step = pd.to_datetime('2018-11-20')

        # alpha parameter. alpha=0.05 implies 95% CI
        pred_ci = pred_uc.conf_int(alpha=0.05)

        # plotDate = '2018-09-15'
        ax = df[show_startDate:]['來客數'].plot(label='observed') # figsize=(20, 15)
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
        ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)

        ax.set_title('SARIMA Forecasting for next ' + str(forecastDate) + ' days')
        ax.set_xlabel('Date')
        ax.set_ylabel('Customer Flow')

        plt.legend()
        plt.savefig('八方雲集//SARIMAX_' + endDate + '-next' + str(forecastDate) + 'days_getforecast.png', bbox_inches='tight', pad_inches=0.5)
        plt.show()

    except Exception as ex:
        print(ex.with_traceback())

def SARIMAX(df, test_predictdate, plot_startdate, select_startdate, T):
    try:
        df = pd.DataFrame(np.array(pd.to_numeric(df['來客數'], downcast='signed')), index=df.index, columns=['來客數'])        # print(df)        # print(type(df))

        plt.style.use('fivethirtyeight')
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
        results.plot_diagnostics(figsize=(15, 12))

        pred = results.get_prediction(start=pd.to_datetime(test_predictdate), dynamic=False)
        pred_ci = pred.conf_int(alpha=0.05)

        ax = df[plot_startdate:].plot(label='Observed', color='b') # figsize=(20, 15)
        pred.predicted_mean.plot(ax=ax, label='Test_Forecast', color = 'orange', alpha=.7)

        ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Customer Flow')
        plt.legend(frameon=False, loc='lower center', ncol=2)

        #save image
        orderlist = "".join(str(x) for x in order)
        seasonal_orderlist = "".join(str(x) for x in seasonal_order)

        if (T == True):
            ax.set_title('SARIMAX MODEL (original)')
            plt.savefig('八方雲集//SARIMAX_Order' + orderlist + '_Season' + seasonal_orderlist + '_' + select_startdate + '.png', bbox_inches='tight', pad_inches=0.5)
            plt.show()
        else:
            ax.set_title('SARIMAX MODEL (adjusted)')
            plt.savefig('八方雲集//SARIMAX_Order(adjusted)' + orderlist + '_Season' + seasonal_orderlist + '_' + select_startdate + '.png', bbox_inches='tight', pad_inches=0.5)
            plt.show()
        # plt.show()

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
        df_forecasted['來客數'] = df_forecasted.values
        # df_forecasted['日期'] = df_forecasted.index
        df_forecasted = df_forecasted.drop([0], axis='columns')

        df_truth = df['來客數']
        # df_truth['日期'] = df_truth.index
        df_truth  = df[predictdate:]

        n = []
        for i in df_truth.index:
            for j in df_forecasted.index:
                if (i == j):
                    diff = df_forecasted["來客數"][j] - df_truth["來客數"][i]
                    n.append(diff)

        df3 = pd.DataFrame(n, index=df_forecasted.index, columns=['誤差'])
        # print(df3)

        # Bias, MAE, MSE, RMSE (List = [])
        forecast_error = n
        bias = sum(forecast_error)*1.0/len(n)
        print('The Bias of our forecasts is {}'.format(bias))

        mae = mean_absolute_error(df_forecasted['來客數'], df_truth['來客數'])
        print('The MAE of our forecasts is {}'.format(mae))

        # mse = mean_squared_error(df_forecasted['來客數'], df_truth['來客數'])/len(df_forecasted)
        mse = MSE(n)
        print('The MSE of our forecasts is {}'.format(mse))

        rmse = sqrt(mse)
        print('The RMSE of our forecasts is {}'.format(rmse))

        record(bias, mae, mse, rmse)
    except Exception as ex:
        print(ex.with_traceback())

def CustomerFlowData(filename):
    try:
        df = pd.read_csv(filename, delimiter=',', usecols=['日期', '時間', '品項'], encoding='utf-8')
        df = df[['日期', '品項']]
        count = df['日期'].value_counts()
        count = count.to_frame()
        count["來客數"] = count["日期"]
        count["日期"] = count.index
        df1 = pd.DataFrame(data=count["日期"], columns=['日期'])
        df2 = pd.DataFrame(data=count["來客數"], columns=['來客數'])
        df = pd.merge(df1, df2, left_index=True, right_index=True)
        df['日期'] = pd.to_datetime(df["日期"])
        df = df.sort_values(by='日期')
        df.index = df['日期']
        return df
    except Exception as ex:
        print(ex.with_traceback())

def dow(date):
    days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dayNumber=date.weekday()
    return days[dayNumber]

def SelectRange(data, select_Start, select_End):
    try:
        df = data.loc[select_Start:select_End]
        return df
    except Exception as ex:
        print(ex.with_traceback())

def FillMissingDate(data, startDate, endDate):
    try:
        # print('start date: ' + startDate, '\nend date: ' + endDate + '\n')
        data.index = pd.to_datetime(data['日期'])
        data = data.drop(['日期'], axis='columns') #axis = 1 or 'columns'

        alldates = pd.date_range(start = startDate, end= endDate)
        alldatesData = alldates.to_frame(index = True)
        alldatesData['日期'] = alldatesData[0]
        alldatesData = alldatesData.drop([0], axis = 'columns')

        # merge data
        mergeresult = pd.concat([alldatesData, data], axis = 'columns', join_axes=[alldatesData.index]) #以左方資料index為主
        mergeresult = mergeresult.fillna(0)
        # print(mergeresult)

        day = []
        for i in mergeresult['日期']:
            daynumber = dow(i)
            day.append(daynumber)
        mergeresult['星期'] = day
        # print(mergeresult)
        # print('-------------')
        return mergeresult

    except Exception as ex:
        print(ex.with_traceback())

def save_OriginalPlot(df, Pic_name):
    try:
        ts = df['來客數']
        plt.plot(ts)
        plt.savefig('八方雲集//' + Pic_name + '.png')
    except Exception as ex:
        print(ex.with_traceback())

def jsontocsv(filename):
    try:
        f = open(filename+'.json')
        data = json.load(f)

        f2 = codecs.open('八方雲集//' + filename + '_data.csv', 'w', encoding="utf-8")
        writer = csv.writer(f2)
        writer.writerow(['日期'] + ['時間'] + ['品項'])

        for item in data:
            date = item['date']
            if ([item['date']]):
                date = pd.to_datetime(date)
                ItemList = [i["name"] for i in item['Items']]
                writer.writerow([date.date()] + [date.time()] + ItemList)
    except Exception as ex:
        print(ex.with_traceback())

def writeCSV(file, filename):
    try:
        file.to_csv('八方雲集//' + filename + '.csv', index = False, encoding='utf-8')

    except Exception as ex:
       print(ex.with_traceback())

def record(bias, mae, mse, rmse):
    try:
        f = open('八方雲集//誤差評估.txt', 'a')
        f.write('\n(1) The Bias of our forecasts is {}'.format(bias))
        f.write('\n(2) The MAE of our forecasts is {}'.format(mae))
        f.write('\n(3) The MSE of our forecasts is {}'.format(mse))
        f.write('\n(4) The RMSE of our forecasts is {}'.format(rmse))
        f.write('\n-------------------\n')
        f.close()
    except Exception as ex:
       print(ex.with_traceback())

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
            M = DF.loc[(DF['星期'] == i) & (DF['outlier'] == False)]['來客數'].mean()
            DF.loc[(DF['星期'] == i) & (DF['outlier'] == True), '來客數'] = M
        # print(DF)
        DF = DF.drop(['z-score', 'outlier'], axis='columns')
        # print(DF)
        return DF

    except Exception as ex:
        print(ex.with_traceback())

def StatisticInfo(df, startDate, endDate, predictDate, T):
    try:
        DFF = deepcopy(df)
        f = open('八方雲集//誤差評估.txt', 'a')
        # Box plot
        p = DFF.boxplot(whis=2) # lower bound = Q1 - (whis*Q1) | upper bound = Q3 + (whis*Q3) | confidence internval = 95%
        p.set_title('Testing Outlier',fontsize=15)
        p.set_ylabel('customer flow')
        # p.set_xlabel('DataSet')
        plt.setp(p, xticklabels=['DataSet'])
        
        # add z-score & determine outlier
        z = np.abs(stats.zscore(DFF['來客數']))
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
            f.write('Selecting date from ' + startDate + ' to ' + endDate + '\n')
            f.write('--------------------------------------------\n')
            plt.savefig('八方雲集//BoxplotwithOutlier' + startDate + ' to ' + endDate + '.png', bbox_inches='tight', pad_inches=0.5)
            f.write('Original data: \n' + str(describes) + '\n')
        else:
            plt.savefig('八方雲集//BoxplotwithoutOutlier' + startDate + ' to ' + endDate + '.png', bbox_inches='tight', pad_inches=0.5 )
            # f.write('\nnumber of outlier: ' + str([df['outlier'] == True].index))
            f.write('\nAdjusted data: \n' + str(describes) + '\n')
        
        # plt.show()
        f.close()
        # print(DFF)
        return DFF
    except Exception as ex:
        print(ex.with_traceback())

def main():
    real_startDate = '2017-06-14'
    real_endDate = '2018-11-05'
    # jsontocsv('八方雲集//八方雲集20170614_20180331')
    # jsontocsv('八方雲集//八方雲集20180401_20181105')

    df1 = CustomerFlowData('八方雲集//八方雲集20170614_20180331.csv')
    df2 = CustomerFlowData('八方雲集//八方雲集20180401_20181105.csv')
    mergeData = mergefile(df1,df2)

    FillData = FillMissingDate(mergeData, real_startDate, real_endDate)
    writeCSV(FillData, '八方雲集' + real_startDate + '_' + real_endDate + '_allData')
    save_OriginalPlot(FillData, real_startDate + '_' + real_endDate)

    select_startDate = '2018-7-10'
    select_endDate = '2018-10-10'
    select_predictDate = '2018-09-26'
    show_startdate = '2018-07-20'
    Selected_Data = SelectRange(FillData, select_startDate, select_endDate)
    

    stats_df = StatisticInfo(Selected_Data, select_startDate, select_endDate, select_predictDate, T = True)
    writeCSV(stats_df, '八方雲集' + real_startDate + '_' + real_endDate + '_statsData')
    SARIMAX(Selected_Data, select_predictDate, show_startdate, select_startDate, T = True) # with some outlier
    print(Selected_Data)
    print('--------------------')
    print(stats_df)
    print('--------------------')

    # remove outlier with 95% confidence interval
    adjusted_df = remove_outlier(stats_df)
    print(adjusted_df)

    stats_df2 = StatisticInfo(adjusted_df, select_startDate, select_endDate, select_predictDate, T = False)
    writeCSV(stats_df2, '八方雲集' + real_startDate + '_' + real_endDate + '_statsData(withoutourlier)')
    # print(adjusted_df)
    result = SARIMAX(adjusted_df, select_predictDate, show_startdate, select_startDate, T = False) #(訓練的資料（range-selected）, 預測起始日期(test), 圖表起始日期, select_startDate)
    get_forecast(adjusted_df, result, select_endDate, show_startdate)

if __name__ == "__main__":
    main()

