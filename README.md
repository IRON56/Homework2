# Homework2
This homework applies a simple trading strategy using data from yahoo finance. Also write a Dash App which allows user to iteractively choose some features of the model, see model strategy's back-test performance and decide whether to trade the order suggested by the model.
## Strategy
Determines whether the IVV's yield of tomorrow would be greater than 0 at the end of each day.

**If it is greater than 0:**

|ID|Action Type|Trading Symbol| Amount|Order Type|Price|
|--|---|---|---|---|---|
|1|BUY|IVV|100|MKT|N/A|

**If it is less than or equal to 0:**

|ID|Action Type|Trading Symbol| Amount|Order Type|Price|
|--|---|---|---|---|---|
|1|SELL|IVV|100|LMT|Closing price of previous day|
## Model
#### Input
1. Yield of IVV today
2. Open price of IVV today
3. Close price of IVV today
4. Moving average yield of IVV for last x days (x cloud be determined by users)
5. Standard deviation of IVV for last x days (x cloud be determined by users)
6. Yield of 5 years US Treasury today
7. Yield of 10 years US Treasury today
8. Yield of 30 years US Treasury today
9. The change in oil price today

#### Output
Whether the yield of IVV tomorrow would be bigger than 0.

1 for yes, 0 for no.

#### Model selection
1. LogLinear Regression
2. K-NN
3. Decision Tree

## Back-Test
The back-test is down on the past 30 days.
#### Blotters
On the previous 30 days.
#### Statistics
Scatter plot of total asset yield in this time period.

## Dash
#### Step 1: Model prepration
1. Allow users to choose the window size of the model
2. Allow users to choose what model they wish to use.
#### Step 2: Back-Test
1. Generate blotter on previous 30 days.
2. Draw graph shows how the total yield of a user changed during that time period.
#### Step 3: Trade Suggestion
1. Using model to give user a trade suggestion on next day.
2. Once the user click Confirm button, the order will be sent to TWS.
## Dash Graph
![avatar](Dash1.pdf)
