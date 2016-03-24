import numpy as np
from numpy.linalg import inv
# hardcoded values
fileDir = '/home/sneha/Data/workspace/patterns_assignment/mnist_dataset/dataset_A/' 
rows = 28
cols = 28
numOfMoments = 8
numOfDigits = 10
numOfSamples = 10
mPQ = np.array([[0,0],[0,2],[1,1],[2,0],[3,0],[0,3],[1,2],[2,1]])      #format -> [p,q,moment value]
moments = np.zeros((numOfDigits,numOfSamples,numOfMoments))            #format -> [digit,sample,no of moments] 
rms = np.zeros((numOfMoments))
mean = np.zeros((numOfDigits,numOfMoments))
variance = np.zeros((numOfDigits,numOfMoments))

for d in xrange(numOfDigits):
    for s in range(numOfSamples):     
        f = open('/home/sneha/Data/workspace/patterns_assignment/mnist_dataset/dataset_A/' + `d` + '-' + `s+1`,'r')
        image = np.genfromtxt(f,delimiter=" ")
        inum = jnum = den = 0     
        for i in xrange(rows):
            for j in xrange(cols):
                inum +=  image[i][j] * (i+1)    # plus one because row and col index should start at 1 as per formula
                jnum += image[i][j] * (j+1)
                den +=  image[i][j]
        imean = inum/den
        jmean = jnum/den    
        for m in xrange(numOfMoments):
            for i in xrange(rows):
                for j in xrange(cols):     
                    moments[d][s][m] += (((i+1) - imean)**mPQ[m][0]) * (((j+1) - jmean)**mPQ[m][1]) * image[i][j]
                       
#finding root mean squared
for m in xrange(numOfMoments):
    sum = 0
    for d in xrange(numOfDigits):
        for s in xrange(numOfSamples):
            sum += moments[d][s][m]*moments[d][s][m]
    rms[m] = (sum/(numOfDigits*numOfSamples))**0.5

 
#normalization
for d in xrange(numOfDigits):
    for s in range(numOfSamples):
        for m in xrange(numOfMoments):
            moments[d][s][m] = moments[d][s][m]/rms[m]

#find mean
for d in xrange(numOfDigits):
    for m in xrange(numOfMoments):
        sum = 0
        for s in xrange(numOfSamples):
            sum += moments[d][s][m]
        mean[d][m] = sum/(numOfDigits*numOfSamples)
    
#find variance
for d in xrange(numOfDigits):
    for m in xrange(numOfMoments):
        sum = 0
        for s in xrange(numOfSamples):
            sum += (moments[d][s][m]-mean[d][m])**2
        variance[d][m] = sum/(numOfDigits*numOfSamples) 


#file to write output

f = open( fileDir + 'result.txt','w')

  
#f.write( values        
for d in xrange(numOfDigits):
    for s in range(numOfSamples):
        f.write( "%d-%2d   "%(d,s+1),)
        for m in xrange(numOfMoments):
             f.write( "%12.2f"%moments[d][s][m],)
        f.write( "\n")
    f.write( "\n")
    f.write( "mean(%d)"%d,)
    for m in xrange(numOfMoments):
        f.write( "%12.2f"%mean[d][m],)
    f.write( "\n")    
    f.write( "var (%d)"%d,)
    for m in xrange(numOfMoments):
        f.write( "%12.2f"%variance[d][m],)
    f.write( "\n")
    f.write( "\n")

f.write( "\n")
f.write( "RMS    ",)
for m in xrange(numOfMoments):
    f.write( "%12.2f"%rms[m],)
f.write( "\n")
f.write( "\n")

"""


"""

#setting precision to two decimal points, when printing matrix
#np.set_printoptions(precision=3,suppress=True)


cov_list = []                                          #empty list to store covariances of each class


for d in xrange(numOfDigits):                   
    temp = np.cov(moments[d],rowvar=0,bias=1)                   #find covariance of moments                              
    cov_list.append(temp)
    
#f.write(ing covariances for each class)
for d in xrange(numOfDigits):   
    f.write( "\n")
    f.write( "Cov-Dataset-A-" + `d`)
    f.write( "\n")
    f.write( "\n")
    for p in range(8):
        for q in range(8):
            f.write( "%10.3f "%cov_list[d][p][q], )                     
        f.write( "\n")
    f.write( "\n")

    
#finding inverse
#inverse = inv(np.matrix(cov))

inv_cov_list = []                                 #empty list to store corresponding inverses

for d in xrange(numOfDigits): 
    inv_cov_list.append(inv(cov_list[d]))


#printing inverses -- Multiply by 10^12
for d in xrange(numOfDigits): 
    f.write( "\n")
    f.write( "Inverse-Dataset-A-" + `d`)
    f.write( "\n")
    f.write( "\n")
    for p in range(8):
        for q in range(8):
            f.write( "%10.3f"%inv_cov_list[d][p][q], )                     
        f.write( "\n")               
    f.write( "\n"  )  




"""
Need to prove covariance * inverse = 1
"""



#average covariance matrix

avg_cov_matrix = np.zeros((8,8))

for d in xrange(numOfDigits):
    avg_cov_matrix = avg_cov_matrix + cov_list[d]   
avg_cov_matrix = avg_cov_matrix/numOfDigits


f.write( "\n")
f.write( "Average Covariance Matrix")
f.write( "\n")
f.write( "\n")
for p in range(8):
    for q in range(8):
        f.write("%10.3f "%avg_cov_matrix[p][q],)
    f.write( "\n")
f.write( "\n")

#Inverse average covariance matrix

inv_avg_cov_matrix = inv(avg_cov_matrix)

f.write( "\n")
f.write( "Inverse of Average Covariance Matrix")
f.write( "\n")
f.write( "\n")
for p in range(8):
    for q in range(8):
        f.write("%10.3f "%inv_avg_cov_matrix[p][q],)
    f.write( "\n")
f.write( "\n")



f.close()





