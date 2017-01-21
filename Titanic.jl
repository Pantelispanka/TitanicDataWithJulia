using DataFrames
using Gadfly
using GLM

trainData = readtable("/Users/pantelispanka/Julia/Titanic/train.csv", header = true)
testData = readtable("/Users/pantelispanka/Julia/Titanic/test.csv", header = true)
showcols(testData)

showcols(trainData)

plot(trainData, x="Fare",y="Survived", Geom.histogram)


median(dropna(trainData[:Age]))

trainData[:Pclass] = float(trainData[:Pclass])
trainData[:Survived] = float(trainData[:Survived])
typeof(trainData[:Survived])
trainData[isna(trainData[:Age]), :Age] = 0

trainFull = DataFrame()
trainFull[:PassengerId] = float(trainData[:PassengerId])
trainFull[:Survived] = trainData[:Survived]
trainFull[:Pclass] = float(trainData[:Pclass])
trainFull[:Sex] = float(ifelse(trainData[:Sex] .== "male" ,1, 0))
trainFull[:Age] = float(isna(trainData[:Age]) .== 0)
trainFull[:Age] = trainData[:Age]

trainFull[:Age] = trainData[:Age]
trainFull[:Fare] = trainData[:Fare]
trainFull[:Parch] = float(trainData[:Parch])
trainFull[:SibSp] = float(trainData[:SibSp])
delete!(trainFull, :PassengerId)
trainFull
showcols(trainFull)

testFull = DataFrame()
push!(testFull, testData)
testFull = DataFrame(Pclass = Float64[], Sex = Float64[], Age = Float64[], Fare= Float64[], Parch = Float64[], SibSp = Float64[])

test = DataFrame()
test[:Pclass] = float(testData[:Pclass])
test[:Sex] = float(ifelse(testData[:Sex] .== "male" ,1, 0))
test[:Age] = testData[:Age]
test[:Age] = float(isna(testData[:Age]) .== 0)
test[:Fare] = float(isna(testData[:Fare]) .== 0)
test[:Parch] = float(testData[:Parch])
test[:SibSp] = float(testData[:SibSp] * 4)
test
showcols(test)

testFull1

trainFull[float(testData[:Pclass]), float(ifelse(testData[:Sex] .== "male" ,1, 0)), testData[:Age],float(testData[:Fare]), float(testData[:Parch]), float(testData[:SibSp])]

trainData[:Fare]
trainData[:Age]
trainFull[:Sex]
showcols(trainFull)
showcols(trainData)
trainData[:Embarked]
cov(dropna(trainData[:Fare]), trainData[:Survived])





trained = fit(GeneralizedLinearModel, Survived ~ Pclass + Sex + Age + Fare + Parch + SibSp, trainFull, Binomial(), LogitLink())
trainedwithoutNA = fit(GeneralizedLinearModel, Survived ~ Pclass + Sex + Age + Fare + Parch + SibSp, trainFull, Binomial(), LogitLink())

trained

trained1
trainFull[:Fare]

predict(trained)
a=[1.0, 1.0, 35.0, 53.1, 8.05, 0, 0]
a1 = DataFrame(Pclass = Float64[], Sex = Float64[],
 Age = Float64[], Fare = Float64[], Parch = Float64[], SibSp = Float64[])
a1


push!(a1, [3.0, 0.0, 4.0, 16.7, 1, 1])
a1
predict(trained, a1)


genderModel = readtable("/Users/pantelispanka/Julia/Titanic/gendermodel.csv", header = true)
genderModel
prediction = predict(trainedwithoutNA, test)

genderModel[ isna(genderModel[:prediction]), :prediction ] .== 0
genderModel[ genderModel[:prediction] .< 0.5 , :prediction ] = 0
genderModel[ genderModel[:prediction] .> 0.5 , :prediction ] = 1

showcols(genderModel)

sum(genderModel[:prediction] .== genderModel[:Survived]) / nrow(genderModel) * 100

nrow(genderModel)


331 / 418

genderModel
trained
trainedwithoutNA
writetable("/Users/pantelispanka/Julia/Titanic/prediction.csv", genderModel)
