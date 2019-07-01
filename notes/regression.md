# Regression
We can have function of y number of students by determined by number of classes. So total number students y = 10*x if every class has 10 students, but usually Regression is used to figure out reletionship of x and y for a non-deterministic relationship i.g y = number of miles and x size of engine in a car, size of an engine doesn not alone tell number of miles but larger size gives larger miles.

## we dont know all the factors therefore we hvave a error term e
Y = f(x) + e
Y is the observed value
e has mean value of 0, all variation in y due to factors other then x

if e is not there y value will wall on x entirely which can not give us a non-determinitic relationship

## How to determin f(x)
usually n number of a x_i value is choosen and drawn in a scatterplot and from there we can choose a reasonalble f(x)

## Simple linear regression model
the simple linear regresion model is used when the reletionship between teo data poitn seems to be linear, and we can draw a straight line between them

Y = b_0 + b_1x + e

## The logistic regresion model
if we are looking at function where the values can either be 0 or 1 e.i classes. We can no longer use the simple linear regression model as it only map linearly. but we can still make the probabilty of 1 or 0 be a linear function of b_0 + b_1x

a funtion that have been quite useful is the logit function, the inverse of the logit function is the activation function sigmoid these are used to make the grapgh non-linear:

P(x) = e^(b_0 + b_1x) / 1 + e^(b_0 + b_1x)

## estimating model parameters b_0 and b_1
the values b_0 and b_1 would never be known by an investigator only pairs of (x, y) values. he will used them to come up with reasonable b_0 and b_1 values.

to find values b that makes the line go throught the point as close to pissible we use the method least squres:

## least squres
se if the line that gives the smallest sum to the vertical distance betwwen the points and the line. the vertical diviation id the height og points - height of line. y_i -(b_0 + b_1*x_i)

the sum is then f(b_0, b_1) = sum(y_i - (b_0 + b_1*x_i))**2 becomes squeares as we have **2, to estimate b_0^ and b_1^ we need f(b_0, b_1) that is smaller then any other f(b_0, b_1) we can find that by taking the partiall derivative for both b_0 and b_1.

b_1^ = sum((x_i - m_mean)(y_i - y_mean))/sum(x_i - m_mean)
b_0^ = y_mean - b_1^*m_mean

before we estimate b-as we should draw a catterplot to see if an linrear estimation is plausiable.

## estimate varianse and stdv
we use the error sum of squres to figure out the variance SSE =  sum(y-y_i)**2 = sum(y-(b_0^ + b_1^*x_i))**2
varians^ = sum(y_i - y_i^)/ (n-2)

stdv^ = kvad(varians^)

SSE can be interpreted as a measure on how much of the variation in data is still un explained

## Total sum squares.
we can use the total sum of squares to se how much of the variation in y is explained by x
total sum = sum(y_i - y_mean) = SST

prosent of y not explained by x or by the regression line = variation not explained by y / total variation
SSE/SST
1 - SSE/SST is described by the regression linear is called the coefiisient of ditermination or r^2 measured from 0 to 1 where 1 explain a lot and 0 not so much.

if the r^2 is big then the researcher should concider a different model, a model with multiple regression or that can support non-linearity.

regression sum of squares sum(y^ - y_mean) (SSR) is the amount of total variation explained by the model thereby SST = SSE + SSR.

## regression analasys
being pulled back by the mean, e.i. if fether height is above average sons wil be to but not so much as father, same pricible for below average height, the phenomenon cad mislead result av both x and y variables are random, while we woek with x as choosen adn y as radnom variables.

## we can satndarixe the values

if we take for every x, x -x_mean/stdv(x)
and for every y, y- y_mean/stdv(y)

we get values of how much the deviate from the mean the correleation corefficient is the same ad the one with the real data, and also the new data has the centrois(x_mean, y_mean) at orige and the if ewe plot a regresiion line the slope will increase by the correlation coefficient (how x and y is related to each other), the slope it self is related to the original.

Can be used to figure out which varibales contribute most to the grapgh and least.             

## understand model
Residual = y - y^
SSE = sum(y -y^)**2 
SSR = sum(y-y_mean)**2
MSE = varianse = sum(y - y^)**2/(n-2) #how spread out is the data from the regression line

standard diviation = kvad(MSE)

(n - 2) bequase we are estimating the slope and intercept always to in simple linear regsression in lutiple can be somtehing else

r^2 = SSR/SST

## test and interval for the slope
1. how much of the y i explained by the model
    for this wee look at R^2
2. we need to see if we can predict future variables, this can only happen if the model in linear
    is the overall F-test or t-test(same test in simple linear regression) significant

    can we jecte the null hopothesis that the slope b1 of the line is zero, if it is zero the there is so slope and they might not be linearly related.
    every line is a estimate, so they have a confidence intevell around them, does that contain slope zero, then vi have a problem

Linear regresison contain many estimators
- betas, varianse, mean valye of y, centroid

our slope is an estimate, this has an standar error of the estimate, which is the satndard deviation of the error term e. s = sqrt(MSE)

## the confidence intervall for the slope

(poitn estimator of slope) b1 +- (t-vale (t-distribution with 4 degree of freedom))t_alpha/2 * s_b1(standard diviation of the slope) (margin of error)
confidence intervall 95% sure the mean is inside this erea

standatd deviation of the slope
s_b1 = s/sqrt(sum(x_i - x_mean)**2)

With confidende intervall we are 95% confident that the intervall contains the true slope of the regression line.

question does it contain 0? id the intervall does not, we can reject the null hypotasis that says its equal to zero and therefor no significatn linear relationship exist between the teo variables, with the counterpart its not equal to zero.

t Test for significance
t = b_1/s_b1 , compate t value vs t_critical vale if t > t_critical then there is significant
 
## confidence interval bands
the estimator of the mean valye y^* for any x* value. mean tip for a meal of 50 buck will have fifferent tip. (confidence interval)

how do we find a mean value for x* (i.g. 50) we use our regression fuction f(x*) = y^*

regression is not deterministic, thats the y^* for one sample, another sample might have different values.

we generate a confidence intervell for the mean y for x*.

predicrion valye (individual value)
E(y*), when x is x* is point esitmator.

sY^* = s*sqrt(1/n + (x* . x_mean)**2/sum(x_i - x_mean)**2)
 this can be done for every value so we then have a for the whole line, the band is not a straight line it flares out in the end, the reason for that is the x* value, the deviation when get furthere out x* -x_mean so the intervall gets larger at eh ends.

its at the smallest at the independent variables mean x. x* - x_mean = 0, since the varince is the least.

## prediciton intervel bands
individual predictions

regression, not deterministic only one sample, but can have estimate from intervalls. preditcion intervall will always be largert then the condifence intervall.

y^* +- t_alpha/s s*pred

s^2pred = s^2 + s_y^2

predicted value is measured by the intervall of every value of y*, while the confidence intervall only is a measure of the measn of y* for dirrent samples i.e E(y*)

the prediction intervall is part of two intervalls, the one for the mean values and the one for eveyr points. also bows out int the end ad the confidence interval
as confidence intervall the prediction is larger as more we go waya from the mean.they are at there most accurate at the mean.

## residual analysis
is the model we use appropiate, the residual can help us with us that. it goes into data science and machine learning.

residual is quantity remaining after other things have been subtracted. (money after paying bill), whats left over after our model have tryed to explain the graph. y-y^. 

the regression model
y = b_0 + b_1 + e
first two regression model, e is the error left thats the residuals. 

SSE = is the distance betwwen the real points and the predicted poitns for ach poinr
SST = distance between real value and mean
SSR = distance between mean value and the predicted value

asssumption:
E(e) = 0
vales of e is independent.

if we plot the residual how much error there is for the predicted values, the residual can have constant varianse, or it can be non-canstant varianse, where the varianse is large at the beginning then face out later on.

if the data is shaped wiredly like a half circle the probabaly the data is non-linear and the model is wrong.

what is the data is heteroscedasticity, non linear, we can tebuild the model with different variables of perform tranformation as taking the log. ot fir non -linear regression model, dont overfit.

## test for resdiuals
Breusch-Pagan test
White test
NCV test (Non-Constant Variance)


## Multiple Regression
can get better value if you use multiple variables to preict one variables

record three values for each isntance (x1, yi, x2)

want yo know y by x1 and x2
 multiple rgression is just a extension of simple linear Regression
there we had 1-1 relationship, here we have many-1 relationship

this does not mean the regression is better. in fact it can make thing worse (overfitting)

adding more variables can always explain more, but bring in more problems total, also when more variables come in, the chances of them bering related to each other increases as realted to the dependent variables called (multicollinearity), the ideal for all x is that they are correlated with y but not each other.

a lot of prep work to do, before running the multiple regresiion.
- correlations
- scatter plots
- simple regressions (for each variable)

if the variables are related to each otheer we dont know which of them are ecplaining the dependent variable. with 4 x-es and one y we get 10 relationships. for mutiple regression we need to figure out which varibale gets the cut and which dos not. as some contribute much other nothing.

y = b_o b_1x_x b_2*x_2 .... b_p*x_p + e
E(y) = b_o b_1x_x b_2*x_2 .... b_p*x_p

in mutiple regression each b in an estimate of the change increase of y, when x_1 increases and  when x_2 is held constant

## data preparation
1. list independent variables and dependent varibales (pich which may think is suitable)
2. collect data on these variables
3. check the relationsjip between each independent variable and the dependent variable using scatterplots and correlations.
4. check the relationshsip between the indepentednt variables usin scatterplot and correlations
5. (optional) conduct simple linear regression for each IV/DV pairs
6. use the non-redundant independent varibale in the analysis to find the best fitting model
7. use the best fitting model to make predictions about the dependent variables

## 3rd and 4th point

draw scatterplot of each IV with DV and see if there is a strong linear realtionship. if one varibales does not show a linear relationshsip it might not be included in the linear Regression

We also need to check scatterplot for IV to IV, for checking multicollinearity. x_1 -> x_2, x_2 -> x_3 and x_3 -> x_1. if the variables shor linear relatiosnhsip to each other we might have a problem as there is multicolinearity, where we dont now which of them are cousing the linearity. if some variables has multicollinearity we wont use both in the mutilple Regression as they are redundant.

need to calculate the correleations (r) to get an objective measure.
p-value below 0.05 is staticlly significant. the correaltion gos from 0-1. shoudl be high for IV to DV. for IV to IV it should be low, as they should not be realted.

## 5th and 6ht point

firt do a simple linear regression for each IV with DV
and interprete results
- how result changes
    - corefficients
        - values, t-statitic, p-values
            values tells us hoe much increase there is for x += 1.
            p-value same as f-significance values because of only one IV
            t-value is n-2 degree on freedom and look at t-table to se value (n is number of observations) se the 95 prosent intelvall 0.05.
    - ANOVA table (analysis of variance)
        - f-value and p-value
            f-value should be high, expect them to be close for each 1 and two variables sets
            P-value tells the significance F and id model is signifficans 0.0001 is significant, below 0.05 is significant
    - R-squeard, R-squared(adjusted), R-squaerd(predicted)
        R-squard is the variantion in DV accounted from the IV
        Adjusted is R-squared accounted with number of IV's here one (always lower)
        predicted tells how well the model does on predicting data points
    - VIF (variance inflation factor)
        poitn outs variables thata are colinear, how much variance increases
        if not correlation the value is 1, value from 5-10 is high might be problematic. Value        above 10 indicates coeffitient are poorly esitmated due to multicollinearity 
    - Mallows C_p
        Look for the one thats low, and is equal to number of IV and 1 constant b_0.

mutilple R same as correleation
make a table for each x values with F,p-value,S,R^2,R² adj, R^ pred.
 - S standardavik is high for x-values with high p-values i.e not signifficans
 - if choosing bweeen two x-values can check witch is better by S, and R^2 ?

* regresion with IV who are linear related to each other
give good result on regression linear the overall model is segnificant, but coeffitient now so good value neither maybe significant beacuese we dont know wich is contribute as they are related. multicollinearity the coefiisient goed crazy

* also if r-adjusted is high but r^2-pred is low we know we have a sirous problem in our model

* also IV that are not linear related with DV, will have coefficient not making sence. migh give un logical answers.

*use the table, F,p-value, all R^2 and have coefiisient in backhead to pich best model
    Rule is want S smalles, R^2 adj and R^2 pred to be higgest and closesnt to each other, and is all is eaquals wants the simplest model easiest to understand.*

## MINITAB CAN MAKE THAT TABLE
 - a large dropoff from adj to pred indicated overfitting to many variables in the model.

## Dummy varaiables
*the inputs here not the outpust are categorical so the x values are either 0 or 1 and dependign on the sequance we cant fighure out which inputs are given, with 0 the beta value goes away

is used to store categorical information, as if the input value coulde be is it good yes or no. not numbers. can use dummy variables as 1 for good and 0 for not good. usually there are many categories so its good to set 0 as otherwise. for n categories we therefor have n-1 dummy variables.*

This can also be done for output variables, will at it later

only one input variable can be categorical to then its betavalue tells how much in contribute with 1 the distance or intercept is highter with null it not.

## two dummy variables
look at scatterploot showing difference between all values in a dummy variables, wee add variables with weight to get in all the differetn value for x. i.e region is a variable it can be nor, west, east, south. we being in 3 variables b*east + b*north + b*west. dont need south as when their all 0 then the value is countd for south.

dummy variables can leave out betas and ultimatly give us differetn equations for differetn values.

forntline systems in excel online last video mutiple regresion good tool !!!!!!!!!!!!!!!! not as much detals as MINITAB
[https://appsource.microsoft.com/en-us/product/office/WA104379190?src=office&corrid=bc3cb910-a2e3-4d5b-99eb-9ab6886cab30&omexanonuid=43a74e60-5c74-447e-b997-0510351ce388&referralurl=https%3a%2f%2fwww.youtube.com%2f]

is staticlly significant means the whole group or line containing rest of variables is moved up if the staticlly significant variable is counted for.
 watch out for causation, which factor result for the ther example large house attrached rich familyes or, rich familys build large house, and welcomes more rich familyes. this can not be answered by statitics more theoretic questions.

## Logistical Regression
tryis to figuore out a probabilty of DV from IV which can be cateforicAL OR NUMBERICAL. tryis to estimate if a probabilty occours or nor occours. effetc of serier os variable on a binary respons varibale. can classify by taking the probabilty of an obersvation being in a particula category.

why cant we use other function
SLR - one to one on quantative variables
MLR - Still predictis a quantative variable and need one as input?
non-linear regression is still in quantative variables, but the data is qurvilinear*

1. binary data does not have normal distribution, which is a condition fore most other type of Regression
2. probabilty are not often linear, as getting the flue, high as baby, low as young, high as old.

central poitns to understand logistic Regression
- probabilty
- odds and odd ratio

## probabilty

P = outcome of interest/ all outcomes

odds = p(somthing accouring)/p(somthing not accouring)
odds = p/1-p

odds 1 mean equla odds if happenign or not happening, over 1 mean higher chance of happening. and odds lower then 1 mean lower chance of happening then not happening.

odds ratio is the odds between two odds. which tells the odds of x bades on ods for x in y and odds for x in z. if over 1 odds for x is larger in x in y then x in z.

need to seperate odds and probabillity, as while the ods might be 4 times higher th eprobability might still be low.

## logit equation

we need to link our IV which can have any vaue to our output which is either 0 or 1 we use the bernoulli distribution which is from 0 to 1. this link is called the logit.

logit is ln(odds)
ln(p/1-p) is the logit(p) or ln(p) - ln(1-p)
log_e x = ln x
whrn the odds er even ln(1) = 0 the logit is 0

at the logit the probabillitys is on the x axes, but we want them on the y axez xan acheive that by taking the the inverse of the logit which is the sigmoid function

logit^-1(x) = sigmoid(x) = e^(x)/1+e^(x)
the sigmoid returns the probability of being 1
the s curve f the sigmoid function fits the scatterplot of binary data.

## the regression coefisient for logistic regression is calculated using the maximum likelihood function MLE

ln(p/1-p) = b0 + b1x

p/1-p = e^(b0 + b1x)
p = e^(b0 + b1x)(1-p)
p = e^(b0 + b1x)-p*e⁽b0+b1x⁾
p + p*e⁽b0+b1x⁾ = e^(b0 + b1x)
p (1 + e^b0+b1x) = e^(b0 + b1x)
p^ = e^(b0 + b1x) / 1 + e^(b0 + b1x)

## eatimaitng the probability
The deviation tablke as the NOVA table in linear Regression
show p-value but instead of usinf f-distrubution, the chi-shquard distrubution is used. since our dependent variable is categorical.

the confidence intervall if including 1, mean the varibale has not much inpact on the odds.

The fico score odds ratio shows the increase inn odds by one 1 point increase e.g if x = 217, then p = 0,7 odd id 0.7/1-07, we do the same for odds(218) odd ratio is then odds(218)/odds(217). and thtas true for eny 1 poitn change. its the same odds ratio. this is a characterestic logistical regresion, each intervall has its own value and if the intervall is same the sam eresult comes.

the regression line of increase of odds ratio, will have same beta for x as the original equations.

so we can use y=e^b1*delta to find the regression line for any intervall of points increased
so we can see our delta changed by 19 points how much the odss increase

we can use the equation from the logit to betas and set in a ratio to find what x need to be toobtain certein odds.

## ANCOVA (analysis of covariance)
 F = MSC/MSE

split SSE to SSE (reduced ) and Cov

## Nonlinear Regression

not all data can be fitt linearly.
goeal minimize error, and be good for new data.

if residual are not linears or shaped triagnled, if low high then low they are not lieanre model.

polynomial regression adds x, x^2, x^3.
X^2 is the quadratic model.
adding term might need to overfitting.
eck p-value x^2 its might much better fit
 


