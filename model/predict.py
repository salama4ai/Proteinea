# importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_to_output_file = "../data/example_output.txt"
path_to_input_file = "../data/example_input.txt"

month_dictionary = {"[JAN]": 1, "[FEB]": 2, "[MAR]": 3, "[APR]": 4, "[MAY]": 5, "[JUN]": 6,
                    "[JUL]": 7, "[AUG]": 8, "[SEP]": 9, "[OCT]": 10, "[NOV]": 11, "[DEC]": 12}
weekday_dict = {"[MON]": 0, "[TUE]": 1, "[WED]": 2, "[THU]": 3, "[FRI]": 4, "[SAT]": 5, "[SUN]": 6}
# list of all possibole (10) output lists, the output must be one of theses lists
groups = [[1, 8, 15, 22], 
         [2, 9, 16, 23], 
         [3, 10, 17, 24], 
         [4, 11, 18, 25], 
         [5, 12, 19, 26], 
         [6, 13, 20, 27], 
         [7, 14, 21, 28],
         [1, 8, 15, 22, 29], 
         [2, 9, 16, 23, 30], 
         [3, 10, 17, 24, 31]]
groups_dict = dict(enumerate(groups))
month_dict = {1:0, 2:3, 3:3, 4:6, 5:1, 6:4, 7:6, 8:2, 9:5, 10:0, 11:3, 12:5}
century_dict = {17:4, 18:2, 19:0, 20:6, 21:4, 22:2, 23:0}

dt = pd.read_csv(path_to_input_file, header=None)
# splitting every line to columns and modify some columns shape
df = pd.DataFrame(index=range(len(dt)))
df["weekday_name"] = [weekday_dict[row.split()[0]] for row in dt[0]]
df["month"] = [month_dictionary[row.split()[1]] for row in dt[0]]
df["leap_year_condition"] = [0 if row.split()[2]=="[False]" else 1 for row in dt[0]]
df["decade"] = [int(row.split()[3].strip("[]")) for row in dt[0]]
df["decade4"] = [1 if ((i*10)%4)!=0 else 0 for i in df["decade"]]       #decade*10%4
df["century_code"] = [century_dict[val] for val in (df.decade//10)]
df["month_code"] = [month_dict[val] for val in df.month]
####################################################################################################
          #####################         year prediction      ########################

class network(nn.Module):
    def __init__(self, in_features=3, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, out_features) 
        self.initialize_weights()
                
    def forward(self, inpt):
        out = F.leaky_relu(self.fc1(inpt))
        out = F.leaky_relu(self.fc2(out))
        out = (self.fc3(out))
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

model_year = network().to(device) 
# importing the trained model                                
model_year = torch.load("../data/year_model_saved")                

data_year = df[["leap_year_condition", "decade", "decade4"]]
data_year = torch.tensor(data_year.values.astype(np.float32)).to(device=device)
df["predicted_year_digit"] = model_year(data_year).argmax(dim=1).cpu()
df["predicted_year"] = [(dec+pred) for dec, pred in zip(df.decade*10, df.predicted_year_digit)]    
####################################################################################################
####################################################################################################
          #####################         day prediction      ########################

df["year_code"] = (df.predicted_year%100)
df["year_code"] = ((df["year_code"]//4)+df["year_code"])%7  
x_day = df[["weekday_name", "month", "month_code", "century_code", "year_code", "leap_year_condition"]]

class network(nn.Module):
    def __init__(self, in_features=5, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 24)
        self.fc2 = nn.Linear(24, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 14)
        self.fc5 = nn.Linear(14, out_features) 
        self.initialize_weights()
                
    def forward(self, inpt):
        out = F.relu(self.fc1(inpt))
        out = F.relu(self.fc2(out))
        out = F.leaky_relu(self.fc3(out))
        out = F.leaky_relu(self.fc4(out))
        out = ((self.fc5(out)))
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

model_day = network(in_features=6, out_features=10).to(device)
# import the trained model
model_day = torch.load("../data/day_model_saved")

d_test = torch.tensor(x_day.values.astype(np.float32))
df["predicted_days_group_index"] = model_day(d_test.to(device=device)).argmax(dim=1).cpu()
df["predicted_day"] = [np.random.choice(groups_dict[i]) for i in df["predicted_days_group_index"]]                

df["output"] = [str(day)+"-"+str(month)+"-"+str(year) for day, month, year in zip(df.predicted_day, df.month, df.predicted_year)]
dt[0] = [(f"{i} {str(j)}") for i, j in zip(dt[0], df["output"])]                

dt.to_csv(path_to_output_file, index=False, header=False)  
print(pd.read_csv(path_to_output_file).head())


##########################################################################################################################################
     ########################### Just for test use this code   ############################
import calendar
print(f"number of wrong leap condition = {([calendar.isleap(i) for i in df.predicted_year]!=df.leap_year_condition).sum()}")
print(f"number of wrong week-days = {([calendar.weekday(year, month, day) for day, month, year in zip(df.predicted_day, df.month, df.predicted_year)]!=df.weekday_name).sum()}")