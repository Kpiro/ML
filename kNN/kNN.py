from math import sqrt
import pandas as pd

class kNN:
    def __init__(self, df):
        self.mins = [None]*len(df.select_dtypes(include='number').columns)
        self.maxs = [None]*len(df.select_dtypes(include='number').columns)
        self.data = self.normalize_data(df)
        self.hiper_K = None

    def start_model(self, df):
        # Divide the data
        self.data = df
        size = len(self.data)
        sub_1_train = self.data[:int(size*0.14)]
        sub_1_val = self.data[int(size*0.14):int(size*0.19)]

        sub_2_train = self.data[int(size*0.19):int(size*0.33)]
        sub_2_val = self.data[int(size*0.33):int(size*0.38)]

        sub_3_train = self.data[int(size*0.38):int(size*0.52)]
        sub_3_val = self.data[int(size*0.52):int(size*0.57)]

        sub_4_train = self.data[int(size*0.57):int(size*0.71)]
        sub_4_val = self.data[int(size*0.71):int(size*0.76)]

        sub_5_train = self.data[int(size*0.76):int(size*0.90)]
        sub_5_val = self.data[int(size*0.90):int(size*0.95)]

        test = self.data[int(size*0.95):]

        # Adjust the hyperparameter k
        subs_k = self.get_k([sub_1_train, sub_2_train, sub_3_train, sub_4_train, sub_5_train],
                            [sub_1_val, sub_2_val, sub_3_val, sub_4_val, sub_5_val])

        # Return the best hyperparameter k
        self.hiper_K = self.tester([sub_1_train, sub_2_train, sub_3_train, sub_4_train, sub_5_train],
                            subs_k,
                            test)

    # Take the training sets and from the validation sets find the hyperparameter k
    def get_k(self, trains, vals):
        medium = len(trains[0])/10
        subs_k = [0]*len(vals)

        for i, (train, val) in enumerate(zip(trains, vals)):
            k = 1
            for v in val:    
                k = 1 if k > medium else k
                if v[1] != self.kNNClasiffier_(train, v[0], k):
                    k += 1
            subs_k[i] = k

        return subs_k

    # Find the best hyperparameter k
    def tester(self, trains, subs_k, test):
        ks = [0]*len(subs_k)
        for i, (train, k) in enumerate(zip(trains, subs_k)):
            count = 0
            for t in test:
                if t[1] == self.kNNClasiffier_(train, t[0], k):
                    count += 1
            ks[i] = count
    
        return subs_k[ks.index(max(ks))]

    def euclidean(self, a, b):
        suma = 0

        for i in range(len(a)):
            suma += (a[i] - b[i])**2

        return sqrt(suma)

    def manhattan(self, a, b):
        suma = 0

        for i in range(len(a)):
            suma += abs(a[i] - b[i])

        return suma

    def jaccard(self, a, b):
        intersection = 0
        for i in range(len(a)):
            intersection += 1 if a[i] == b[i] else 0

        return intersection / (len(a) + len(b) - intersection)

    def minkowski(self, a, b, p=3):
        suma = 0

        for i in range(len(a)):
            suma += (a[i] - b[i])**p

        return suma**(1/p)

    def kNNClasiffier_(self, S, x, k):
        distances = [(0,'')]*len(S)
        for i, item in enumerate(S):
            print('item ', item)
            distances[i] = (self.euclidean(item[0], x), item[1])

        distances = sorted(distances)

        groups = dict()
        i = 0
        while k>0:
            item = distances[i]
            if item[1] not in groups:
                groups[item[1]] = 0
            else:    
                groups[item[1]] += 1
            i += 1    
            k -= 1
        
        clasiffier = ''
        maxi = -1
        for item in groups:
            if groups[item] > maxi:
                maxi = groups[item]
                clasiffier = item
        return clasiffier

    def kNNClasiffier(self, x):
        x = [(item-self.mins[index])/(self.maxs[index]-self.mins[index]) for index, item in enumerate(x)]
        return self.kNNClasiffier_(S=self.data, x=x, k=self.hiper_K)
    
    def adjust(self, S, pred):
        zipx = zip(*S)
        return list(zip(zipx, pred))
    
    def normalize_data(self, df):
        for index, col in enumerate(df.select_dtypes(include='number').columns):
            minx = min(df[col].values)
            maxx = max(df[col].values)
            self.mins[index] = minx
            self.maxs[index] = maxx
            df[col] = [(x - minx)/(maxx-minx) for x in df[col].values]

        return df

    
# ------------------------------ MAIN ------------------------------
# Load data
df = pd.read_csv('houses.csv')

# Prepare data
model = kNN(df)
df = model.adjust(S=[df.amount_room, df.size_house], pred=df.place)
model.start_model(df)
x = [3, 674]
print(model.kNNClasiffier(x))

