from __future__ import print_function
import sys
import numpy as np

from datetime import datetime
from numpy.random import rand
from numpy import matrix
from pyspark import SparkConf, SparkContext
from os.path import join, isfile


def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print("File %s does not exist." % ratingsFile)
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print("No ratings provided.")
        sys.exit(1)
    else:
        return ratings


def parseRating(line):
    """
    Parses a rating record in productLens format userId,productId,rating,timestamp .
    """
    fields = line.strip().split(",")
    return np.long(fields[3]), (int(fields[0]), int(fields[1]), float(fields[2]))


def parseProduct(line):
    """
    Parses a product record in productLens format productId,productTitle .
    """
    fields = line.strip().split(",")
    return int(fields[0]), fields[1]


def parseUser(line):
    """
    Parses a User record in productLens format productId,productTitle .
    """
    fields = line.strip().split(",")
    return int(fields[0]), fields[1]


def computeRMSE(R, us, vs):
    """
    compute Root mean square error value
    """
    diff = R - us * vs.T
    rmse_val = np.sqrt(np.sum(np.power(diff, 2)) / (U * V))
    return rmse_val


def updateUV(i, uv, r):
    """
    Calculate updated values of U,V
    """
    uu = uv.shape[0]
    ff = uv.shape[1]
    xtx = uv.T * uv
    xty = uv.T * r[i, :].T

    for j in range(ff):
        xtx[j, j] += lambdas * uu

    updated_val = np.linalg.solve(xtx, xty)
    return updated_val


def updateU(i, v, r):
    vv = v.shape[0]
    ff = v.shape[1]
    xtx = v.T * v
    xty = v.T * r[i, :].T

    for j in range(ff):
        xtx[j, j] += lambdas * vv

    updated_U = np.linalg.solve(xtx, xty)
    return updated_U


def updateV(i, u, r):
    uu = u.shape[0]
    ff = u.shape[1]
    xtx = u.T * u
    xty = u.T * r[i, :].T

    for j in range(ff):
        xtx[j, j] += lambdas * uu

    updated_V = np.linalg.solve(xtx, xty)
    return updated_V


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("USAGE: spark-submit AmazonALS.py AmznRatingsFileDir <outputfile> Iterations Partitions")
        sys.exit(1)

        # parameters are declared
    lambdas = 0.1
    np.random.seed(20)
    hdfs_src_dir = sys.argv[1]
    iterations = int(sys.argv[3]) if len(sys.argv) > 2 else 10
    partitions = int(sys.argv[4]) if len(sys.argv) > 3 else 4
    start_time = datetime.now()
    outputfile = sys.argv[2]

    # AppName, memory is set to SparkContext
    conf = SparkConf().setAppName("AmazonALS").set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    # ratings is an RDD of (timestamp, (userId, productId, rating))
    ratings = sc.textFile(join(hdfs_src_dir, "ratings.dat")).map(parseRating)

    # products is an RDD of (productId, productTitle)
    products = sc.textFile(join(hdfs_src_dir, "products.dat")).map(parseProduct)

    # users is an RDD of (userID, userName)
    users = sc.textFile(join(hdfs_src_dir, "users.dat")).map(parseUser)

    r_list = ratings.values().repartition(partitions).cache().collect()
    r_array = np.array(r_list)

    numRatings = ratings.count()
    U = ratings.values().map(lambda r: r[0]).max()
    V = ratings.values().map(lambda r: r[1]).max()
    F = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    Z = np.zeros((U, V))
    R = np.matrix(Z)

    for i in range(numRatings):
        r_local = r_array[i]
        R[(r_local[0] - 1), (r_local[1] - 1)] = r_local[2]

    us = matrix(rand(U, F))
    usb = sc.broadcast(us)

    vs = matrix(rand(V, F))
    vsb = sc.broadcast(vs)

    Rb = sc.broadcast(R)

    for i in range(iterations):
        us = sc.parallelize(range(U), partitions).map(lambda x: updateUV(x, vsb.value, Rb.value)).collect()
        us = matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)

        vs = sc.parallelize(range(V), partitions).map(lambda x: updateUV(x, usb.value, Rb.value.T)).collect()
        vs = matrix(np.array(vs)[:, :, 0])
        vsb = sc.broadcast(vs)

        rmse_val = computeRMSE(R, us, vs)

        print("Iteration %d:" % i)
        print("\nRMSE: %5.4f\n" % rmse_val)

    reco = np.dot(us, vs.T)

    end_time = datetime.now()
    total_time = start_time - end_time
    total_t = divmod(total_time.days * 86400 + total_time.seconds, 60)

    print("---------------------------------------------------------------------")
    print("User-Product Recommendation: Predicted User-Ratings Matrix is created")
    print("---------------------------------------------------------------------")
    print("Total Minutes and Seconds: " + str(total_t))

    l_prod = products.collect()
    # l_users = users.collect()
    # URatings = []
    # preURatings = []

    output = open("output_AmazonReco.dat", 'w')
    print("File writing started")

    for i in range(U):
        for j in range(V):
            pRating = reco[i, j]
            aRating = R[i, j]
            if aRating == 0 and pRating > 3:
                output.write(str(i) + "," + l_prod[j][1] + "," + str(pRating) + "\n")

    output.close()

    """
         # Calcuate Average rating of users before and after prediction for testing purpose

             # if(aRating!=0):
             # URatings.append(aRating)
         # preURatings.append(pRating)

    # avgURating = float(sum(URatings))/float(len(uRatings))
    # avgPRating = float(sum(preURatings))/float(len(preURatings))

    # print ("Avg User Rating: ", avgURating)
    # print ("Avg Predicted User Rating: ", avgPRating )
    """

    end_time = datetime.now()
    total_time = start_time - end_time
    total_t = divmod(total_time.days * 86400 + total_time.seconds, 60)

    print("-------------------------------------------------")
    print("User-Product Recommendation: File Write Completed")
    print("-------------------------------------------------")
    print("Total Minutes and Seconds: " + str(total_t))

    sc.stop()
