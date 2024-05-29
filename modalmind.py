import numpy as np
import random as rm

# The statespace
states = ["Move","Think","Craft"] # Sleep is alternative compination

# Possible sequences of events
transitionName = [["MM","MC","MT"],["TM","TC","TT"],["CM","CT","CC"]]

# Probabilities matrix (transition matrix)
transitionMatrix = [[0.2,0.6,0.2],[0.1,0.6,0.3],[0.2,0.7,0.1]]
if sum(transitionMatrix[0])+sum(transitionMatrix[1])+sum(transitionMatrix[1]) != 3:
    print("Somewhere, something went wrong. Transition matrix, perhaps?")
else: print("All is gonna be okay, you should move on!! ;)")
# A function that implements the Markov model to forecast the state/mood.
def activity_forecast(days):
    # Choose the starting state
    activityToday = "Move"
    activityList = [activityToday]
    i = 0
    prob = 1
    while i != days:
        if activityToday == "Move":
            change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix[0])
            if change == "MM":
                prob = prob * 0.2

                #maybe probability *  float current time

                activityList.append("Move")
                pass
            elif change == "MT":
                prob = prob * 0.6
                activityToday = "Think"
                activityList.append("Think")
            else:  
                prob = prob * 0.2
                activityToday = "Craft"
                activityList.append("Craft")
        elif activityToday == "Craft":
            change = np.random.choice(transitionName[1],replace=True,p=transitionMatrix[1])
            if change == "CC":
                prob = prob * 0.5
                activityList.append("Craft")
                pass
            elif change == "CT":
                prob = prob * 0.2
                activityToday = "Think"
                activityList.append("Think")
            else:
                prob = prob * 0.3
                activityToday = "Move"
                activityList.append("Move")
        elif activityToday == "Think":
            change = np.random.choice(transitionName[2],replace=True,p=transitionMatrix[2])
            if change == "TT":
                prob = prob * 0.1
                activityList.append("Think")
                pass
            elif change == "TM":
                prob = prob * 0.2
                activityToday = "Move"
                activityList.append("Move")
            else:
                prob = prob * 0.7
                activityToday = "Craft"
                activityList.append("Craft")
        i += 1    
    return activityList

# To save every activityList
list_activity = []
count = 0

# `Range` starts from the first count up until but excluding the last count
for iterations in range(1,10000):
        list_activity.append(activity_forecast(2))

# Check out all the `activityList` we collected    
#print(list_activity)

# Iterate through the list to get a count of all activities ending in state:'Run'
for smaller_list in list_activity:
    if(smaller_list[2] == "Think"):
        count += 1




# Calculate the probability of starting from state:'Sleep' and ending at state:'Run'
percentage = (count/10000) * 100
print("The probability of starting at state:'Move' and ending at state:'Craft'= " + str(percentage) + "%")



#if some percentage nice change mode to another, maybe multimodal vision