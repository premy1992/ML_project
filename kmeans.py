from random import shuffle
import numpy as np
import tensorflow as tf
from sklearn import metrics
class similarity:
    def __init__(self):
        return

    def euclid_distance(self,vector1, vector2):
        return tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(vector1, vector2), 2)))

    def minkowski_distance(self,vector1, vector2, order):
        return tf.pow(tf.reduce_sum(tf.pow(tf.subtract(vector1, vector2), order)), 1 / order)

    def manhattan_distance(self,vector1, vector2):
        return tf.reduce_sum(tf.abs(tf.subtract(vector1, vector2)))

    def cosine_similarity(self,vector1, vector2):
        numerator = tf.reduce_sum(tf.multiply(vector1, vector2))
        denominator = tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vector1))),
                                  tf.sqrt(tf.reduce_sum(tf.square(vector2))))
        return tf.divide(numerator, denominator)
    def cityblock_distance(self,vector1,vector2):
        return tf.reduce_sum(tf.abs(tf.subtract(vector1,vector2)))

    def mahalanobis_distance(self,vector1, vector2):
        # md = (x - y) * LA.inv(R) * (x - y).T
        return 0

    def jaccard_similarity(self,vector1, vector2):
        intersection_cardinality = tf.sets.set_intersection(vector1,vector2)
        union_cardinality = tf.sets.set_union(vector1,vector2)
        return tf.divide(intersection_cardinality,union_cardinality)

class KMeans:
    centroids_result=None
    assignments_result=None
    no_of_clusters=None
    epoch=50
    def __init__(self):
        return

    def set_epoch(self,value):
        self.epoch=value

    def get_centroids(self):
        return self.centroids_result

    def get_assignments(self):
        return self.assignments_result

    def set_centroids(self,value):
        self.no_of_clusters=len(value)
        self.centroids_result=value

    def set_assignments(self,value):
        self.assignments_result=value
    def homogeneity_score(self,labels):
        try:
            result=metrics.homogeneity_score(labels, self.assignments_result)
        except:
            result = metrics.homogeneity_score(labels.reshape(-1), self.assignments_result)
        return result

    def v_measure_score(self, labels):
        try:
            result=metrics.v_measure_score(labels, self.assignments_result)
        except:
            result = metrics.v_measure_score(labels.reshape(-1), self.assignments_result)
        return result

    def completeness_score(self, labels):
        try:
            result=metrics.completeness_score(labels, self.assignments_result)
        except:
            result = metrics.completeness_score(labels.reshape(-1), self.assignments_result)
        return result

    def adjusted_rand_score(self, labels):
        try:
            result=metrics.adjusted_rand_score(labels, self.assignments_result)
        except:
            result = metrics.adjusted_rand_score(labels.reshape(-1), self.assignments_result)
        return result

    def adjusted_mutual_info_score(self, labels):
        try:
            result=metrics.adjusted_mutual_info_score(labels, self.assignments_result)
        except:
            result = metrics.adjusted_mutual_info_score(labels.reshape(-1), self.assignments_result)
        return result

    def silhouette_score(self, labels):
        try:
            result=metrics.silhouette_score(labels, self.assignments_result)
        except:
            try:
                result = metrics.silhouette_score(labels.reshape(-1, 1), self.assignments_result)
            except:
                result = metrics.silhouette_score(labels.reshape(1, -1), self.assignments_result)
        return result
    def accuracy_score(self,labels):
        try:
            result=metrics.accuracy_score(labels, self.assignments_result)
        except:
            result = metrics.accuracy_score(labels.reshape(-1), self.assignments_result)
        return result
    def jaccard_smililarity_score(self,labels):
        try:
            result=metrics.jaccard_similarity_score(labels, self.assignments_result)
        except:
            result = metrics.jaccard_similarity_score(labels.reshape(-1), self.assignments_result)
        return result
    def train(self,vectors,noofclusters):
        self.no_of_clusters=noofclusters
        """
        K-Means Clustering using TensorFlow.
        'vectors' should be a n*k 2-D NumPy array, where n is the number
        of vectors of dimensionality k.
        'noofclusters' should be an integer.
        """

        noofclusters = int(noofclusters)
        assert noofclusters < len(vectors)

        # Find out the dimensionality
        dim = len(vectors[0])

        # Will help select random centroids from among the available vectors
        vector_indices = list(range(len(vectors)))
        shuffle(vector_indices)

        # GRAPH OF COMPUTATION
        # We initialize a new graph and set it as the default during each run
        # of this algorithm. This ensures that as this function is called
        # multiple times, the default graph doesn't keep getting crowded with
        # unused ops and Variables from previous function calls.

        graph = tf.Graph()

        with graph.as_default():

            # SESSION OF COMPUTATION

            sess = tf.Session()

            ##CONSTRUCTING THE ELEMENTS OF COMPUTATION

            ##First lets ensure we have a Variable vector for each centroid,
            ##initialized to one of the vectors from the available data points
            centroids = [tf.Variable((vectors[vector_indices[i]]))
                         for i in range(noofclusters)]
            ##These nodes will assign the centroid Variables the appropriate
            ##values
            centroid_value = tf.placeholder(tf.float32, [dim])
            cent_assigns = []
            for centroid in centroids:
                cent_assigns.append(tf.assign(centroid, centroid_value))

            ##Variables for cluster assignments of individual vectors(initialized
            ##to 0 at first)
            assignments = [tf.Variable(0) for i in range(len(vectors))]
            ##These nodes will assign an assignment Variable the appropriate
            ##value
            assignment_value = tf.placeholder("int32")
            cluster_assigns = []
            for assignment in assignments:
                cluster_assigns.append(tf.assign(assignment,
                                                 assignment_value))

            ##Now lets construct the node that will compute the mean
            # The placeholder for the input
            mean_input = tf.placeholder(tf.float32, [None, dim])
            # The Node/op takes the input and computes a mean along the 0th
            # dimension, i.e. the list of input vectors
            mean_op = tf.reduce_mean(mean_input, 0)

            ##Node for computing Euclidean distances
            # Placeholders for input
            v1 = tf.placeholder(tf.float32, [dim])
            v2 = tf.placeholder(tf.float32, [dim])
            sm=similarity()
            distane_function = sm.euclid_distance(v1,v2)

            ##This node will figure out which cluster to assign a vector to,
            ##based on Euclidean distances of the vector from the centroids.
            # Placeholder for input
            centroid_distances = tf.placeholder(tf.float32, [noofclusters])
            cluster_assignment = tf.argmin(centroid_distances, 0)

            ##INITIALIZING STATE VARIABLES

            ##This will help initialization of all Variables defined with respect
            ##to the graph. The Variable-initializer should be defined after
            ##all the Variables have been constructed, so that each of them
            ##will be included in the initialization.
            init_op = tf.global_variables_initializer()
            # Initialize all variables
            sess.run(init_op)

            ##CLUSTERING ITERATIONS

            # Now perform the Expectation-Maximization steps of K-Means clustering
            # iterations. To keep things simple, we will only do a set number of
            # iterations, instead of using a Stopping Criterion.
            for iteration_n in range(self.epoch):

                ##EXPECTATION STEP
                ##Based on the centroid locations till last iteration, compute
                ##the _expected_ centroid assignments.
                # Iterate over each vector
                for vector_n in range(len(vectors)):
                    vect = vectors[vector_n]
                    # Compute Euclidean distance between this vector and each
                    # centroid. Remember that this list cannot be named
                    # 'centroid_distances', since that is the input to the
                    # cluster assignment node.
                    distances = [sess.run(distane_function, feed_dict={
                        v1: vect, v2: sess.run(centroid)})
                                 for centroid in centroids]
                    # Now use the cluster assignment node, with the distances
                    # as the input
                    assignment = sess.run(cluster_assignment, feed_dict={
                        centroid_distances: distances})
                    # Now assign the value to the appropriate state variable
                    sess.run(cluster_assigns[vector_n], feed_dict={
                        assignment_value: assignment})

                ##MAXIMIZATION STEP
                # Based on the expected state computed from the Expectation Step,
                # compute the locations of the centroids so as to maximize the
                # overall objective of minimizing within-cluster Sum-of-Squares
                for cluster_n in range(noofclusters):
                    # Collect all the vectors assigned to this cluster
                    assigned_vects = [vectors[i] for i in range(len(vectors))
                                      if sess.run(assignments[i]) == cluster_n]
                    # Compute new centroid location
                    new_location = sess.run(mean_op, feed_dict={
                        mean_input: np.array(assigned_vects)})
                    # Assign value to appropriate variable
                    sess.run(cent_assigns[cluster_n], feed_dict={
                        centroid_value: new_location})
                print "iteration : ",iteration_n

            # Return centroids and assignments
            self.centroids_result = np.array(sess.run(centroids))
            self.assignments_result = np.array(sess.run(assignments))

import dataset_reader,utils

data = dataset_reader.read_cifar10('cifar-10-batches-py/', one_hot=False)
train_x=data['train_x']
train_y=data['train_y']
test_x=data['test_x']
test_y=data['test_y']
text_labels=data['text_labels']
print train_x.shape,train_y.shape,test_x.shape,test_y.shape
no_of_clusters=10
model=KMeans()
model.set_epoch(10)
model.train(np.float32(train_x), no_of_clusters)
utils.save_model(model,"k-means-000")
model=utils.load_model("k-means-000")
print "No Of Cluster : ",model.no_of_clusters
print "homogeneity score : ",model.homogeneity_score(train_y)
print "Completeness : " ,model.completeness_score(train_y)
print "V-measure : " ,model.v_measure_score(train_y)
print "Adjusted Rand Index : ", model.adjusted_rand_score(train_y)
print "Adjusted Mutual Information : ",model.adjusted_mutual_info_score(train_y)
print "Silhouette Coefficient : ", model.silhouette_score(train_y)