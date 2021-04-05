# Homework2
This homework applies a simple trading strategy using data from yahoo finance. Also write a Dash App which allows user to iteractively choose some features of the model, see model strategy's back-test performance and decide whether to trade the order suggested by the model.
## Strategy
Determines whether the IVV's yield of tomorrow would be greater than 0 at the end of each day.

If it is greater than 0:

|ID|Action Type|Trading Symbol| Amount|Order Type|Price|
|--|---|---|---|---|---|
|1|BUY|IVV|100|MKT|N/A|

If it is less than or equal to 0:

|ID|Action Type|Trading Symbol| Amount|Order Type|Price|
|--|---|---|---|---|---|
|1|SELL|IVV|100|LMT|Closing price of previous day|
## Model
####Input
1. Yield of IVV today
2. Open price of IVV today
3. Close price of IVV today
4. Moving average yield of IVV for last x days (x cloud be determined by users)
5. Standard deviation of IVV for last x days (x cloud be determined by users)
6. Yield of 5 years US Treasury today
7. Yield of 10 years US Treasury today
8. Yield of 30 years US Treasury today
9. The change in oil price today

####Output
Whether the yield of IVV tomorrow would be bigger than 0.

1 for yes, 0 for no.

####Model selection
1. LogLinear Regression
2. K-NN
3. Decision Tree

## Dash
